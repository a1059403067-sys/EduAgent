# BGE-M3 向量检索 
# backend/core/knowledge_base.py（阶段版：仅含 BGEMEmbedder + 数据类，5.6 补充 KnowledgeBaseClient）
# BGE-M3支持8192最大token, 覆盖大多数chunk, 维度1024维，大多数模型只能输出稠密向量，它可以一次推理同时输出稠密向量和稀疏向量
# ollama企业里面最好不要用，本地部署的是压缩之后的模型，而魔搭下载的是纯正的完整模型，可以下载完整模型之后用ollama部署
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)
backend_path = os.path.dirname(os.path.dirname(__file__))


# ──────────────────────────────────────────────────────────────
# BGE-M3 本地嵌入模型（进程内单例，dense + sparse 双输出）
# ──────────────────────────────────────────────────────────────

class BGEMEmbedder:
    """
    BGE-M3 本地嵌入模型单例。

    一次推理同时输出：
      - dense 向量（1024 维浮点数组，用于语义相似度检索）
      - sparse 向量（{token_id: weight} 字典，用于关键词精确检索）

    进程内单例：首次调用 get_instance() 时加载模型（约5-15秒），
    后续调用直接返回同一实例，不重复加载。

    用法：
        embedder = BGEMEmbedder.get_instance()
        dense, sparse = embedder.encode_query("什么是 Spring IOC？")
    """

    _instance: Optional["BGEMEmbedder"] = None   # 单例持有

    def __init__(self, model_path: str):
        # ── 兼容性补丁1：FlagEmbedding 1.3.x 依赖 transformers 内部函数 ──
        # transformers>=5.0 移除了 is_torch_fx_available，
        # 但当前锁定 transformers==4.51.0 不受影响。
        # 此补丁作为保险，避免未来升级时报 ImportError。
        import importlib.util as _ilu
        from transformers.utils import import_utils as _tf_iu
        if not hasattr(_tf_iu, "is_torch_fx_available"):
            _tf_iu.is_torch_fx_available = (
                lambda: _ilu.find_spec("torch.fx") is not None
            )
        #
        # ── 兼容性补丁2：修复 FlagEmbedding 的 dtype 参数在 transformers 4.51 不生效的问题 ──
        # FlagEmbedding 1.4.x 通过 from_pretrained(dtype=torch.float16) 传精度，
        # 但 transformers 4.51 只认 torch_dtype，dtype 会被转传给
        # XLMRobertaModel.__init__()（不接受该参数 → TypeError）。
        # 这里把 dtype 转换为 torch_dtype：既避免 TypeError，又让权重真正按 fp16 加载。
        # 若只是丢弃 dtype，内层 XLMRobertaModel 会是 fp32，而
        # colbert_linear / sparse_linear 是 fp16，前向时报
        # "mat1 and mat2 must have the same dtype"，且被 FlagEmbedding 的
        # except RuntimeError 吞掉后表现为诡异的 IndexError。
        from transformers.models.xlm_roberta import modeling_xlm_roberta as _xlm
        _OriginalXLMRoberta = _xlm.XLMRobertaModel

        class _PatchedXLMRobertaModel(_OriginalXLMRoberta):
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                dtype = kwargs.pop("dtype", None)
                if dtype is not None and "torch_dtype" not in kwargs:
                    kwargs["torch_dtype"] = dtype
                return super().from_pretrained(*args, **kwargs)

            def __init__(self, config, **kwargs):
                kwargs.pop("dtype", None)  # 保险：防止其他路径传入 dtype 报 TypeError
                super().__init__(config, **kwargs)

        _xlm.XLMRobertaModel = _PatchedXLMRobertaModel

        import torch
        from FlagEmbedding import BGEM3FlagModel

        logger.info("bge_m3.loading", model_path=model_path)

        # ── BGEM3的fp16 仅在 CUDA 上启用，MPS（Apple M系列）不启用 ──
        # MPS 在 BGE-M3 attention 矩阵乘法上会触发 LLVM ERROR，
        # CPU 模式下用 fp32，速度稍慢但稳定。
        _use_fp16 = torch.cuda.is_available()

        self._model = BGEM3FlagModel(
            model_name_or_path=model_path,
            use_fp16=_use_fp16,
        )
        logger.info("bge_m3.loaded", use_fp16=_use_fp16)

    @classmethod
    def get_instance(cls) -> "BGEMEmbedder":
        """获取单例（首次调用时加载模型，后续复用），工厂方法或单例模式的写法，类里面写一个方法，让它返回自己的实例"""
        if cls._instance is None:
            bge3_path = os.path.join(backend_path, get_settings().bge_m3_model_path)
            cls._instance = BGEMEmbedder(bge3_path)
        return cls._instance

    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
    ) -> tuple[list[list[float]], list[dict]]:
        """
        向量化用户问题，返回 稠密向量 和 稀疏向量。
        批量编码文本，同时返回 dense 和 sparse 两种向量。
        传入文本和batch_size，转换输出的类型，然后返回 稠密向量list[list[float] 和 稀疏向量list[dict]，dict={"词索引": foalt}
        Args:
            texts:      待编码的文本列表
            batch_size: 单次推理批大小，越大速度越快但显存占用越多；
                        12 是 16GB 显存 / 统一内存下的经验值

        Returns:
            (dense_vecs, sparse_vecs)
              dense_vecs:  list of 1024-dim float 向量，每项对应 texts[i]
              sparse_vecs: list of {token_id: weight} 字典，每项对应 texts[i]
        """
        valid_texts = [t for t in texts if t and t.strip()]

        if not valid_texts:
            logger.warning("bge_m3.encode_empty", original_count=len(texts))
            return [], []  # 返回空结果，不崩溃

        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,            # BGE-M3 支持最长 8192 token，覆盖大多数 chunk
            return_dense=True,          # 输出稠密语义向量
            return_sparse=True,         # 输出稀疏关键词向量
            return_colbert_vecs=False,  # ColBERT 多向量表示，本项目不用
        )

        dense_vecs = output["dense_vecs"].tolist()   # numpy → Python list

        # sparse: numpy.float16 → Python float
        # 必须转换！LangGraph MemorySaver 用 msgpack 序列化 State，
        # msgpack 不支持 numpy.float16，会在运行时抛 TypeError。
        sparse_vecs = [
            {int(k): float(v) for k, v in d.items()}
            for d in output["lexical_weights"]
        ]
        return dense_vecs, sparse_vecs

    def encode_query(self, text: str) -> tuple[list[float], dict]:
        """
        编码单条查询，返回 (dense_vec, sparse_vec)。

        查询时调用此方法（而非 encode），batch_size=1 避免不必要的 padding。

        Returns:
            (dense_vec, sparse_vec)
              dense_vec:  1024-dim float 列表
              sparse_vec: {token_id: weight} 字典
        """
        dense_list, sparse_list = self.encode([text], batch_size=1)
        return dense_list[0], sparse_list[0]        # 看似是返回两个元素，实际是返回一个元组


# ──────────────────────────────────────────────────────────────
# 数据类：DocumentChunk（建库写入 Milvus 的数据结构）
# ──────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """
    @dataclass语法糖，相当于省略了__init__方法，直接给类定义属性
    准备写入 Milvus 的单个文档块，字段与 Milvus Schema 一一对应。

    id:               全局唯一 ID（MD5 of content + document_id + chunk_index）
    content:          chunk 文本（Contextual RAG 模式下含 LLM 生成的上下文描述前缀）
    embedding:        Dense 向量（BGE-M3，1024 维），稠密向量
    sparse_embedding: Sparse 向量（{token_id: weight}，BGE-M3 lexical weights），稀疏向量
    source_name:      来源标注（检索结果展示用，如 "Java讲义 > 第3章 > 3.1 IOC"）
    """
    id:               str
    content:          str
    embedding:        list[float]
    sparse_embedding: dict
    course_id:        str
    document_id:      str
    source_name:      str
    chunk_type:       str                  # "text" / "code" / "table"
    chunk_index:      int
    version:          str
    tenant_id:        str = "tenant_default"
    updated_at:       int = field(default_factory=lambda: int(time.time()))


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def generate_chunk_id(content: str, document_id: str, chunk_index: int) -> str:
    """
    生成 chunk 全局唯一 ID（MD5 散列）。

    用 document_id + chunk_index + content 前缀组合，确保：
    - 同一文档不同位置的 chunk 不冲突
    - 内容不变时 ID 稳定（幂等重建时不会重复插入）

    注意：此函数后续会作为 KnowledgeBaseClient 的静态方法重新出现（5.6），
    届时 build_knowledge_base.py 会改为调用 KnowledgeBaseClient.generate_chunk_id()。
    """
    raw = f"{document_id}_{chunk_index}_{content[:50]}"
    return hashlib.md5(raw.encode()).hexdigest()


import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

from backend.config import get_settings
from backend.core.logger import get_logger,configure_logging
configure_logging()
logger = get_logger(__name__)

COLLECTION_NAME = "knowledge_domain"


class KnowledgeBaseClient:
    """
    Milvus 知识库客户端（MilvusClient 版）。

    单 Collection 设计（knowledge_domain），按 tenant_id 字段过滤实现多租户隔离。
    本节实现写入方法，5.6 节追加检索方法。

    单例连接：_client 是类变量，整个进程只创建一次 MilvusClient 连接。
    """

    _client: Optional["MilvusClient"] = None
    _loaded: bool = False

    # HNSW 搜索时的候选集大小，精度/速度平衡点（5.6 节 _hybrid_search 使用）
    ANN_EF = 64

    def __init__(self):
        if KnowledgeBaseClient._client is None:
            settings = get_settings()
            uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
            KnowledgeBaseClient._client = MilvusClient(uri=uri)
            logger.info("milvus.connected", uri=uri)

        if not KnowledgeBaseClient._loaded:
            try:
                KnowledgeBaseClient._client.load_collection(COLLECTION_NAME)
            except Exception:
                pass   # init_milvus.py 尚未运行时忽略
            KnowledgeBaseClient._loaded = True

    # ── 写入：批量 Upsert ────────────────────────────────────

    def upsert_chunks(self, chunks: list) -> int:
        """
        批量写入文档块（Upsert：primary key 存在则更新，不存在则插入）。

        MilvusClient 行格式写入：每行一个 dict，key = 字段名，字段顺序无关。
        """
        if not chunks:
            return 0
        # print(f'len(chunks):{len(chunks)}')
        # print(f'chunks[0]:{chunks[0]}')
        # 一行样本是一个entity，代表一个实体
        data = [
            {
                "id":               c.id,
                "embedding":        c.embedding,
                "sparse_embedding": c.sparse_embedding,
                "content":          c.content[:4096],
                "chunk_index":      c.chunk_index,
                "document_id":      c.document_id,
                "course_id":        c.course_id,
                "tenant_id":        c.tenant_id,
                "source_name":      c.source_name,
                "chunk_type":       c.chunk_type,
                "version":          c.version,
                "updated_at":       c.updated_at,
            }
            for c in chunks
        ]

        self._client.upsert(collection_name=COLLECTION_NAME, data=data)
        logger.info("knowledge_base.chunks_upserted", count=len(chunks))
        return len(chunks)

        # ── 写入：删除指定文档的所有 chunk ──────────────────────

    def delete_document_chunks(self, document_id: str) -> None:
        """
        删除指定文档的所有 chunk（文档更新时先删后插，幂等重建）。
        对 document_id 转义，防止 filter 表达式注入。
        """
        safe_id = document_id.replace('"', '\\"')
        self._client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'document_id == "{safe_id}"',
        )
        logger.info("knowledge_base.document_deleted", document_id=document_id)

    @staticmethod
    def generate_chunk_id(content: str, document_id: str, chunk_index: int) -> str:
        """生成 chunk 唯一 ID（MD5）。内容+位置不变则 ID 不变，支持幂等 upsert。"""
        raw = f"{document_id}_{chunk_index}_{content[:50]}"
        return hashlib.md5(raw.encode()).hexdigest()

        # ── 检索配置 ─────────────────────────────────────────────

    VECTOR_TOP_K = 10  # Hybrid 召回的候选数量，传给 Reranker 精排

    def _hybrid_search(
            self,
            query_embedding: list[float],
            query_sparse: dict,
            top_k: int,
            filters: Optional[str] = None,
    ) -> list[dict]:
        """
        对 knowledge_domain 做 Hybrid 检索（Dense + Sparse → WeightedRanker 融合）。

        两个 AnnSearchRequest 分别构造 Dense 和 Sparse 检索请求，
        由 Milvus 在服务端并行执行后，用 WeightedRanker 加权融合排序。

        Args:
            query_embedding: Dense Query 向量（1024 维，来自 encode_query）
            query_sparse:    Sparse Query 向量（{token_id: weight}，来自 encode_query）
            top_k:           每路召回数量（融合后同样取 top_k）
            filters:         Milvus bool 表达式，如 'tenant_id == "xxx"'

        Returns:
            候选文档列表，每项含 "content" / "score" / "metadata"。
            score 是 WeightedRanker 的加权排序信号，不是概率，
            直接交给 5.7 节的 Reranker 做精细打分。
        """
        try:
            # ── Dense ANN 检索请求 ─────────────────────────────────────
            # COSINE 度量匹配 BGE-M3 dense 向量（L2 归一化后等价于余弦相似度）
            # ef=64：HNSW 搜索时的候选集大小，越大精度越高，64 是精度/速度平衡点
            dense_req = AnnSearchRequest(
                data=[query_embedding],
                anns_field="embedding",
                param={
                    "metric_type": "COSINE",
                    "params": {"ef": self.ANN_EF},
                },
                limit=top_k,
                expr=filters,
            )

            # ── Sparse 关键词检索请求 ──────────────────────────────────
            # IP（内积）是 BGE-M3 lexical_weights 的标准度量
            sparse_req = AnnSearchRequest(
                data=[query_sparse],
                anns_field="sparse_embedding",
                param={"metric_type": "IP"},
                limit=top_k,
                expr=filters,
            )

            output_fields = [
                "content", "source_name", "chunk_type",
                "course_id", "document_id", "chunk_index",
            ]

            # ── WeightedRanker(0.7, 0.3) ──────────────────────────────
            # 第一个权重对应第一个请求（Dense），第二个对应第二个请求（Sparse）
            # 两路结果在 Milvus 服务端并行检索，融合后返回
            results = self._client.hybrid_search(
                collection_name=COLLECTION_NAME,
                reqs=[dense_req, sparse_req],
                ranker=WeightedRanker(0.7, 0.3),
                limit=top_k,
                output_fields=output_fields,
            )
            # print(f'results: {results}')
            # print(f'results: {type(results)}')
            # print(f'results: {results[0]}')
            # print(f'results: {len(results[0])}')
            candidates = []
            for hit in results[0]:
                # print(f'hit-->{hit}')
                candidates.append({
                    "content": hit["entity"].get("content") or "",
                    "score":   hit.get("distance") or 0.0,
                    "metadata": {
                        "source_name": hit["entity"].get("source_name") or "",
                        "chunk_type":  hit["entity"].get("chunk_type")  or "text",
                        "course_id":   hit["entity"].get("course_id")   or "",
                        "document_id": hit["entity"].get("document_id") or "",
                        "chunk_index": hit["entity"].get("chunk_index") or 0,
                    },
                })
            logger.info(
                "knowledge_base.hybrid_search_done",
                candidates=len(candidates),
            )
            return candidates
        except Exception as e:
            logger.error("knowledge_base.hybrid_search_failed", error=str(e))
            return []

    @staticmethod
    def _build_filter(tenant_id: str, course_id: Optional[str] = None) -> str:
        """
        构建 Milvus bool 过滤表达式。
        对 tenant_id / course_id 做转义，防止 filter 表达式注入。
        方式：将 tenant_id / course_id 中的双引号替换为 \"，防止中间引号和左侧引号组合成闭合的字符串
        防止注入：" or "1"="1"，若不转义course_id = "" or "1"="1"，相当于不过滤，转义之后course_id = "\" or \"1\"=\"1"
        """
        safe_tenant = tenant_id.replace('"', '\\"')
        expr = f'tenant_id == "{safe_tenant}"'
        if course_id:
            safe_course = course_id.replace('"', '\\"')
            expr += f' and course_id == "{safe_course}"'
        return expr


if __name__ == '__main__':
    # bge_model = BGEMEmbedder(model_path = "/Users/ligang/Desktop/LiveEduAgent/backend/models/embedding/bge-m3")
    # print(bge_model._model)
    # # 验证设备
    # device = next(bge_model._model.model.parameters()).device
    # print(f'device: {device}')

    BGEMEmbedder._instance = None
    model = BGEMEmbedder.get_instance()
    dense, sparse = model.encode_query(text="商品聚合多模态大模型项目主要讲的是什么内容")

    # print(f'result: {len(result[0])}')
    # print("*"*80)
    # print(f'result: {result[1]}')
    # print(generate_chunk_id(content="AI是人工智能", document_id="AI", chunk_index=0))

    kb = KnowledgeBaseClient()
    results = kb._hybrid_search(dense, sparse, top_k=5)
    if results:
        print(f'results[0]: {results[0]}')
    else:
        print('无检索结果：knowledge_domain 集合为空，请先运行 scripts/build_knowledge_base.py 建库')

    # from scripts.build_knowledge_base import load_document,split_documents, embed_chunks
    # file_path = "E:\workspace\EduAgent\EduAgent\samples\sample2.md"
    # # 加载文档
    # docs = load_document(file_path)
    # # 切分文档
    # chunks = split_documents(docs, file_path)
    # # 变成向量
    # embed_docs = embed_chunks(chunks, course_id="01", document_id="02")
    # # 存储到集合中
    # kb.upsert_chunks(embed_docs)
    # # kb.delete_document_chunks(document_id="02")


# 导入知识库文档 
# scripts/build_knowledge_base_old.py（阶段版：仅文档加载部分）
# 5.3 补充分块，5.4 补充嵌入，5.5 完整版

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")


def load_pdf(file_path: str) -> list[Document]:
    """
    加载 PDF 文档，每页返回一个 Document。

    只提取文字层内容；图片/扫描件页面 page_content 为空字符串，
    不报错（在 5.3 分块时会过滤掉空页）。

    Args:
        file_path: PDF 文件的本地路径

    Returns:
        list[Document]，每个 Document 对应一页
        metadata 包含 source（文件路径）和 page（页码，从 0 开始）
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()  # 返回一个列表，列表中每个Document代表一页，可以统计列表长度，也可以通过metadata元数据获取页数
    char_count = 0
    for page in pages:
        char_count += len(page.page_content)
    print(f"  [PDF] 加载完成：共{len(pages)} 页，{char_count} 字符 ← {Path(file_path).name}")
    return pages


def load_markdown(file_path: str) -> list[Document]:
    """
    加载 Markdown 文档，整个文件作为一个 Document 返回。

    不在这里做标题切分——那是 5.3 分块步骤的工作。
    这里只负责把文件内容读进内存。

    Args:
        file_path: Markdown 文件的本地路径（.md 或 .markdown）

    Returns:
        list[Document]，只有一个元素，page_content 为文件全文
        metadata 包含 source（文件路径）
    """
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()  # 返回一个列表，列表中每个Document代表一页，可以统计列表长度，也可以通过metadata元数据获取页数
    char_count = 0
    for page in docs:
        char_count += len(page.page_content)
    print(f"  [MD]  加载完成：共{len(docs)} 页，{char_count} 字符 ← {Path(file_path).name}")
    return docs


def load_document(file_path: str) -> list[Document]:
    """
    统一文档加载入口。

    根据文件扩展名自动选择加载器：
        .pdf            → PyPDFLoader（文字层提取）
        .md / .markdown → TextLoader（纯文本读取）

    Args:
        file_path: 文档本地路径

    Returns:
        list[Document]

    Raises:
        ValueError: 不支持的文件类型
        FileNotFoundError: 文件不存在
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    ext = path.suffix.lower()  # 获取文件扩展名并转换为小写
    if ext == ".pdf":  # 如果扩展名是 .pdf，则是 PDF 文件
        return load_pdf(file_path)  # 调用 load_pdf 函数加载 PDF 文件
    elif ext in (".md", ".markdown"):  # 如果扩展名是 .md 或 .markdown，则是 Markdown 文件
        return load_markdown(file_path)  # 调用 load_markdown 函数加载 Markdown 文件
    else:
        raise ValueError(
            f"不支持的文件类型：{ext}\n"  # 输出不支持的文件类型
            f"当前支持：.pdf / .md / .markdown\n"  # 输出当前支持的文件类型
            f"提示：可用 markitdown 将 Word/PPT 转换为 .md 后再导入"  # 输出提示信息
        )


if __name__ == '__main__':
    doc = load_document("E:\workspace\EduAgent\EduAgent\samples\sample2.md")
    print(doc)

# 简历审查 
import asyncio
import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import text

from backend.agents.resume.graph import build_resume_graph
from backend.core.logger import get_logger
from backend.dependencies import AsyncSessionLocal, get_current_user
from backend.core.llm_factory import set_llm_user_id  # LLM 限流：注入当前学员 ID

router = APIRouter()
logger = get_logger(__name__)

# ── 线程本地图：每个线程一份独立的图实例，避免并发竞争 ──
import threading

_graph_local = threading.local()


def _get_graph():
    """获取线程本地的图实例（当前线程没有就编译一个）。langgraph本身不会出现这个问题，但是其他模块可能有，此处学习这个处理方法"""
    # 检查当前线程的_graph_local上有没有graph属性。不用给代码加锁，利用线程自带的隔离机制。可以考虑线程本地图的形式，给每一个线程一份独立的图实例，这样就不会造成并发竞争的现象了。另一个方法使用互斥锁，只有一把钥匙，谁抢到钥匙谁干活
    # thread_id字段也能实现线程隔离
    if not hasattr(_graph_local, "graph"):
        _graph_local.graph = build_resume_graph()
    return _graph_local.graph


# ── GC 保护：模块级集合持有后台任务的强引用，防止被垃圾回收 ──
_background_tasks: set[asyncio.Task] = set()
RESUME_REVIEW_TIMEOUT_SECONDS = 10 * 60  # 审查超时阈值：10 分钟
REAP_INTERVAL_SECONDS = 120  # 后台回收任务扫描间隔：2 分钟
_reaper_task: asyncio.Task | None = None  # 定时回收循环的任务引用（shutdown 时取消）


async def _reap_stale_processing() -> int:
    """把超过 10 分钟仍处于 processing 的记录批量置为 failed（后台定时调用）。

    覆盖「服务重启中断了后台审查任务」的场景：任务没了，记录会永远卡在
    processing，前端轮询永远拿不到结果；这里用 SQL 端时间做比较（NOW()），
    与数据库时区一致，不受应用服务器时钟影响。
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                UPDATE resume_reviews
                SET status = 'failed',
                    error_msg = '审查任务超时（超过10分钟未完成），请重新上传。',
                    updated_at = NOW()
                WHERE status = 'processing'
                  AND COALESCE(updated_at, created_at) < NOW() - make_interval(mins => :minutes)
            """),
            {"minutes": RESUME_REVIEW_TIMEOUT_SECONDS // 60},
        )
        await session.commit()
        return result.rowcount


async def _stale_reaper_loop() -> None:
    """后台循环：定期回收卡在 processing 的孤儿记录。启动时先扫一次再周期性扫描。"""
    while True:
        try:
            count = await _reap_stale_processing()
            if count:
                logger.info("resume.stale_processing_reaped", count=count)
        except Exception as e:  # 扫描失败不影响下一轮（DB 抖动时下轮再试）
            logger.error("resume.stale_reap_failed", error=str(e))
        await asyncio.sleep(REAP_INTERVAL_SECONDS)


@router.on_event("startup")
async def _start_reaper() -> None:
    """应用启动时拉起定时回收任务（挂在 router 上，不依赖具体 main 文件）。"""
    global _reaper_task
    if _reaper_task is None or _reaper_task.done():
        _reaper_task = asyncio.create_task(_stale_reaper_loop())
        logger.info("resume.stale_reaper_started", interval_seconds=REAP_INTERVAL_SECONDS)


@router.on_event("shutdown")
async def _stop_reaper() -> None:
    """应用关闭时取消回收任务。"""
    global _reaper_task
    if _reaper_task and not _reaper_task.done():
        _reaper_task.cancel()
    logger.info("resume.stale_reaper_stopped")


async def _mark_review_failed(review_id: str, error_msg: str) -> None:
    """把仍处于 processing 的记录标记为 failed（幂等：仅当当前是 processing 才改）。"""
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                UPDATE resume_reviews
                SET status = 'failed', error_msg = :error_msg, updated_at = NOW()
                WHERE id = :review_id AND status = 'processing'
            """),
            {"review_id": review_id, "error_msg": error_msg[:1000]},
        )
        await session.commit()


@router.post("/upload", status_code=202)  # 202 Accepted：已接受、正在处理
async def upload_resume(
        file: UploadFile = File(...),  # 上传的文件
        current_user: dict = Depends(get_current_user),  # 鉴权：拿到当前用户
):
    """上传 PDF 简历，触发异步审查，立即返回 review_id。"""
    if not file.filename.lower().endswith(".pdf"):  # 只收 PDF
        raise HTTPException(status_code=400, detail="仅支持 PDF 格式")

    review_id = str(uuid.uuid4())
    student_id = current_user["user_id"]
    tenant_id = current_user["tenant_id"]

    # 1. 读取并校验文件（用 await 读，避免阻塞事件循环）
    MAX_PDF_SIZE = 20 * 1024 * 1024  # 20MB 上限
    file_bytes = await file.read()
    if len(file_bytes) > MAX_PDF_SIZE:
        raise HTTPException(status_code=413, detail="文件过大，最大支持 20MB")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="上传文件为空")

    # 暂存到临时目录
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), f"{review_id}_upload.pdf")  # D:\Temp\{review_id}_upload.pdf
    with open(tmp_path, "wb") as f:  # 清空并以二进制形式写入
        f.write(file_bytes)
    logger.info("upload_resume.file_saved", review_id=review_id, tmp_path=tmp_path)

    # 2. 写入 resume_reviews 初始记录（status=processing）
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                INSERT INTO resume_reviews (id, tenant_id, student_id, pdf_minio_path, status)
                VALUES (:id, :tenant_id, :student_id, :pdf_minio_path, 'processing')
            """),
            {"id": review_id, "tenant_id": tenant_id, "student_id": student_id,
             "pdf_minio_path": f"resumes/{student_id}/{review_id}.pdf"},
        )
        await session.commit()

    # 3. 准备初始 State，后台启动图执行
    initial_state = {
        "messages": [], "student_id": student_id, "tenant_id": tenant_id,
        "review_id": review_id, "pdf_minio_path": "", "pdf_local_path": tmp_path,
        "raw_text": "", "page_count": 0, "structured": None,
        "dimension_scores": [], "weighted_score": 0.0, "issues": [],
        "summary": None, "fallback_used": False, "structured_output": None,  # 新增字段
    }

    def _on_task_done(t: asyncio.Task):
        """任务结束回调：移除引用、清理临时文件、失败则标记 failed。"""
        _background_tasks.discard(t)  # 从集合移除（释放强引用）
        if os.path.exists(tmp_path):  # 清理临时 PDF
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        mark_failed_msg = None
        if t.cancelled():  # 任务被取消（如服务重启）
            mark_failed_msg = "审查任务被服务重启中断，请重试。"
            logger.warning("resume.background_task_cancelled", review_id=review_id)
        else:
            exc = t.exception()  # 任务抛了异常
            if exc:
                mark_failed_msg = f"审查任务执行失败：{exc}"
                logger.error("resume.background_task_failed", review_id=review_id,
                             error=str(exc), exc_info=exc)
        if mark_failed_msg:  # 异步把记录标记为 failed
            try:
                asyncio.create_task(_mark_review_failed(review_id, mark_failed_msg))  # _mark_review_failed是异步函数可以直接用
                # loop = asyncio.get_running_loop()
                # loop.create_task(_mark_review_failed(review_id, mark_failed_msg))       # 在异步函数里面跑同步阻塞代码，用run_in_executor来执行
            except RuntimeError:
                pass  # 循环已关闭：下次查询走超时兜底

    # create_task 前注入 user_id：后台 task 创建时复制当前 context，限流生效
    set_llm_user_id(student_id)
    graph = _get_graph()  # 拿线程本地图
    task = asyncio.create_task(graph.ainvoke(initial_state))  # 后台执行，ainvoke不止可以执行大模型，它按照图的边，按顺序执行节点
    _background_tasks.add(task)  # GC 保护：使用集合持有强引用，不被python的垃圾回收机制回收
    task.add_done_callback(_on_task_done)  # 注册完成回调，当任务完成时自动执行回调函数，需要的移除引用、清理临时文件、失败则标记 failed，而不仅仅是移除引用
    logger.info("upload_resume.task_started", review_id=review_id)

    return {  # 立即返回（不等审查完成）
        "review_id": review_id, "status": "processing",
        "message": "简历已上传，正在审查中，预计 30-60 秒完成。",
    }


@router.get("/reviews/{review_id}")
async def get_review(review_id: str, current_user: dict = Depends(get_current_user)):
    """查询审查状态/结果。processing / done / failed / 404。"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT id, status, scores, issues, summary, error_msg, created_at, updated_at
                FROM resume_reviews
                WHERE id = :review_id AND student_id = :student_id
            """),
            {"review_id": review_id, "student_id": current_user["user_id"]},
        )
        row = result.mappings().fetchone()

    if not row:  # 不存在或不属于自己
        raise HTTPException(status_code=404, detail="审查记录不存在")

    if row["status"] == "processing":
        # 超时兜底：防止后台任务中断后长期卡 processing
        last_ts = row["updated_at"] or row["created_at"]
        if isinstance(last_ts, datetime):
            elapsed = (datetime.now(timezone.utc) - last_ts).total_seconds()
            if elapsed >= RESUME_REVIEW_TIMEOUT_SECONDS:
                timeout_msg = "审查任务超时或被中断，请重新上传后重试。"
                await _mark_review_failed(review_id, timeout_msg)
                return {"review_id": review_id, "status": "failed", "error_msg": timeout_msg}
        return {"review_id": review_id, "status": "processing"}

    if row["status"] == "failed":
        return {"review_id": review_id, "status": "failed",
                "error_msg": row.get("error_msg") or "审查任务失败，请重新上传。"}

    # status == done：JSONB 列已被 asyncpg 自动反序列化为 dict/list，无需 json.loads
    def _to_dict(v):
        if v is None:
            return {}
        if isinstance(v, (dict, list)):
            return v
        import json as _json
        return _json.loads(v)

    scores_data = _to_dict(row["scores"])
    return {
        "review_id": review_id, "status": "done",
        "weighted_score": scores_data.get("weighted_score", 0),
        "dimension_scores": scores_data.get("dimension_scores", []),
        "issues": _to_dict(row["issues"]) if not isinstance(row["issues"], list) else row["issues"],
        "summary": _to_dict(row["summary"]),
    }


@router.delete("/reviews/{review_id}", status_code=204)
async def delete_review(review_id: str, current_user: dict = Depends(get_current_user)):
    """删除审查记录（WHERE 带 student_id，只能删自己的）。"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                DELETE FROM resume_reviews
                WHERE id = :review_id AND student_id = :student_id
            """),
            {"review_id": review_id, "student_id": current_user["user_id"]},
        )
        await session.commit()
    if result.rowcount == 0:  # 没删到任何行 = 不存在/无权限
        raise HTTPException(status_code=404, detail="记录不存在")


@router.get("/reviews")
async def list_reviews(current_user: dict = Depends(get_current_user)):
    """列出本人历史审查记录（摘要，按时间倒序）。"""
    # 转换为jsonb类型，从json中获取weighted_score字段的值，并转换为float类型
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT id, status, created_at,
                       (scores::jsonb ->> 'weighted_score')::float AS weighted_score    
                FROM resume_reviews
                WHERE student_id = :student_id
                ORDER BY created_at DESC
                LIMIT 50
            """),
            {"student_id": current_user["user_id"]},
        )
        rows = result.mappings().all()

    items = [
        {"review_id": row["id"], "status": row["status"],
         "created_at": row["created_at"].isoformat() if row["created_at"] else None,
         "weighted_score": row["weighted_score"]}
        for row in rows
    ]
    return {"items": items, "total": len(items)}

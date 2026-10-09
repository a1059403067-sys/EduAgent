import os
import sys

import uvicorn
from fastapi import FastAPI
from backend.api.v1 import resume

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path = [p for p in sys.path if os.path.abspath(p) != _BACKEND_DIR]

from backend.api.v1 import qa
from backend.api.v1 import auth
from backend.api.v1 import exam
from backend.api.v1 import interview
from backend.mcp.knowledge_base_server import mcp as kb_mcp
from backend.mcp.web_search_server import mcp as ws_mcp
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # app.mount() 不会自动传播子应用 lifespan，必须在这里手动启动
    # _session_manager.run() 初始化 anyio task group，MCP 请求依赖它
    async with ws_mcp._session_manager.run():
        async with kb_mcp._session_manager.run():
            yield


app = FastAPI(lifespan=lifespan)

# CORS：浏览器直连 8006（VITE_API_BASE_URL），必须放行前端开发端口
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://localhost:5173", "http://localhost:8080",
        "http://127.0.0.1:3000", "http://127.0.0.1:5173", "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router, prefix="/api/v1/auth")
app.include_router(qa.router, prefix="/api/v1/qa")
app.include_router(exam.router, prefix="/api/v1/exam")
app.include_router(interview.router, prefix="/api/v1/interview")
app.include_router(resume.router, prefix="/api/v1/resume")
app.mount("/mcp/kb", kb_mcp.streamable_http_app())
app.mount("/mcp/web-search", ws_mcp.streamable_http_app())

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8006)

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.src.config.settings import settings
from backend.src.database.db import init_db
from backend.src.routes.api import router as api_router
from api.routes import analysis as v1_analysis
from api.routes import trading as v1_trading
from api.routes import news as v1_news
from api.routes import dashboard as v1_dashboard
from api.routes import backtest as v1_backtest
from backend.src.services.market.data_manager import market_data_manager
from backend.src.services.portfolio.position_monitor import position_monitor
from backend.src.services.orchestration.state_machine import orchestrator
from backend.src.services.websocket.broadcaster import ws_broadcaster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("server")

# Background ticker for frontend live streams
async def websocket_stream_task():
    while True:
        try:
            scan_info = market_scanner_info()
            # Broadcast live state to all connected frontends
            await ws_broadcaster.broadcast("heartbeat", {
                "active_position": execution_engine_pos(),
                "portfolio": execution_engine_portfolio(),
                "app_state": orchestrator.current_state,
                "best_opportunity": scan_info.get("best_opportunity"),
                "candidates": scan_info.get("candidates", []),
                "current_step": scan_info.get("current_step", "IDLE"),
                "activity_log": scan_info.get("activity_log", []),
                "is_scanning": scan_info.get("is_scanning", False)
            })
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug(f"WS stream error: {e}")
            await asyncio.sleep(1.0)

def execution_engine_pos():
    from backend.src.services.execution.engine import execution_engine
    return execution_engine.active_position

def execution_engine_portfolio():
    from backend.src.services.execution.engine import execution_engine
    return execution_engine.get_portfolio_status()

def market_scanner_info():
    from backend.src.services.scanner.scanner import market_scanner
    return {
        "best_opportunity": market_scanner.best_opportunity,
        "candidates": market_scanner.latest_candidates,
        "current_step": market_scanner.current_step,
        "activity_log": market_scanner.scan_activity_log,
        "is_scanning": market_scanner._is_scanning
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Database...")
    await init_db()

    logger.info("Starting Market Data Manager (Binance Feed)...")
    await market_data_manager.start()

    logger.info("Starting Position Monitor...")
    await position_monitor.start()

    logger.info("Starting Orchestrator State Machine...")
    await orchestrator.start()

    ws_task = asyncio.create_task(websocket_stream_task())

    # Pre-run an initial scan in background
    asyncio.create_task(orchestrator.run_pipeline_once(
        amount_pkr=settings.USER_AMOUNT_PKR,
        min_profit_pkr=settings.MIN_DESIRED_PROFIT_PKR,
        auto_enter=False
    ))

    yield

    logger.info("Shutting down background services...")
    ws_task.cancel()
    await orchestrator.stop()
    await position_monitor.stop()
    await market_data_manager.stop()

app = FastAPI(
    title="AI Crypto Scalping Trading Bot API",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(v1_analysis.router, prefix="/api/v1/analysis", tags=["Market Analysis"])
app.include_router(v1_trading.router, prefix="/api/v1/trading", tags=["Paper Trading & Setup"])
app.include_router(v1_news.router, prefix="/api/v1/news", tags=["News Engine"])
app.include_router(v1_dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard Metrics"])
app.include_router(v1_backtest.router, prefix="/api/v1/backtest", tags=["Backtesting"])

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_broadcaster.connect(websocket)
    try:
        while True:
            # Keep reading from client in case of messages or ping
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_broadcaster.disconnect(websocket)
    except Exception:
        ws_broadcaster.disconnect(websocket)

# Mount frontend build static files if present
frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)

import asyncio
import json
import logging
from typing import Set
from fastapi import WebSocket

logger = logging.getLogger("ws_broadcaster")

class WebSocketBroadcaster:
    """Manages active frontend WebSocket client connections and multiplexes real-time messages."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Frontend client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"Frontend client disconnected. Remaining: {len(self.active_connections)}")

    async def broadcast(self, message_type: str, data: any):
        if not self.active_connections:
            return

        payload = json.dumps({
            "type": message_type,
            "data": data
        })

        dead_connections = set()
        for conn in self.active_connections:
            try:
                await conn.send_text(payload)
            except Exception:
                dead_connections.add(conn)

        for dead in dead_connections:
            self.active_connections.discard(dead)

ws_broadcaster = WebSocketBroadcaster()

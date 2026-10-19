# API Documentation

## Base URL
- Production Root: `http://127.0.0.1:8000`
- API Prefix: `/api`
- WebSocket Endpoint: `ws://127.0.0.1:8000/ws`

---

## REST Endpoints

### 1. `GET /api/status`
Returns connectivity health across Binance API, WebSocket stream, AI engine, ML model, database, and risk engine.
```json
{
  "binance_api": "CONNECTED",
  "websocket": "CONNECTED",
  "market_data": "CONNECTED",
  "ai": "CONNECTED",
  "ml": "CONNECTED",
  "database": "CONNECTED",
  "execution": "CONNECTED",
  "risk_engine": "ACTIVE",
  "auto_trading": false,
  "paper_trading": true,
  "auto_close": true,
  "active_tickers_count": 757,
  "app_state": "IDLE"
}
```

### 2. `GET /api/portfolio`
Returns virtual / live balances (USDT & PKR), daily realized PnL, win rate, consecutive losses, and pause state.

### 3. `GET /api/scanner`
Returns top scanned candidates and the current #1 best opportunity.

### 4. `POST /api/scan`
Triggers an immediate hierarchical market scan across all Binance USDT pairs.

### 5. `GET /api/position`
Returns the active open scalp position with real-time unrealized PnL, duration, and progress percentage.

### 6. `POST /api/trade/execute`
Manually approves and executes the current best opportunity.

### 7. `POST /api/trade/close`
Closes the active scalp position at market price and settles PnL.

### 8. `POST /api/risk/emergency-stop`
Immediate emergency stop: closes open positions, halts new entries, and disables Auto Trading.

### 9. `POST /api/risk/resume`
Resumes trading and clears risk-paused circuit breakers.

### 10. `POST /api/backtest`
Runs historical backtest simulation on 1m candles.
**Request Body**:
```json
{
  "symbol": "BTCUSDT",
  "candles": 500,
  "amount_pkr": 5000.0
}
```

### 11. `GET /api/settings` and `POST /api/settings`
Retrieves or updates user amount (PKR), target profit, API credentials, and execution toggles.

### 12. `POST /api/paper/reset`
Resets virtual balance to `$1,000.00 USDT`.

---

## WebSocket Stream (`/ws`)
Broadcasts real-time events to connected frontend clients:
- `heartbeat`: Active position status, portfolio balances, and state machine updates.

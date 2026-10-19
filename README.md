# Nexus Quant AI Crypto Scalping Trading Bot (Binance)

Production-grade, high-frequency cryptocurrency scalping platform integrated with Binance. Built using a **multi-layer AI + quantitative signal + real-time market-data architecture** designed specifically for ultra-short holding periods (2–3 minutes, 5 minutes maximum).

---

## ⚡ Primary Directive
**HIGH-QUALITY TRADE SELECTION + FAST EXECUTION + STRICT CAPITAL PRESERVATION + SHORT HOLDING TIME**

The bot strictly prioritizes **No trade is better than a bad trade**. It aggressively rejects low-confidence setups, illiquid order books, wide spreads, and conflicting indicator regimes. It never uses Martingale or averaging down.

---

## 🏛️ System Architecture

```
                  BINANCE REAL-TIME WEBSOCKET FEED
                                ↓
                     IN-MEMORY MARKET STATE
                                ↓
                   LEVEL 1: ULTRA-FAST FILTER
           (USDT Pairs, >$3M Vol, Tight Spread <0.08%, Active Trades)
                                ↓
             LEVEL 2: MULTI-TIMEFRAME TECHNICAL ANALYSIS
              (1m, 3m, 5m: EMA 9/21/50, VWAP, RSI, MACD, ATR, BB)
                                ↓
              LEVEL 3: QUANTITATIVE SCORING + ML MODEL
       (100-pt Weighted Quant Score + HistGradientBoosting P(up)/P(down))
                                ↓
                 LEVEL 4: AI VALIDATION & ROUTER
       (OpenAI GPT-4o-mini / GPT-4o with Deterministic Quant Fallback)
                                ↓
               LEVEL 5: RISK VALIDATOR & PRE-ENTRY CHECK
           (Dynamic ATR SL/TP, R:R >= 1.5, Fee & Slippage Friction)
                                ↓
                TRADE APPROVED  ───or───  NO TRADE
                                ↓
                     EXECUTION ENGINE (PAPER / LIVE)
                                ↓
                   POSITION MONITOR (500ms LOOP)
             (Auto-TP, Auto-SL, 3-min Time Exit, Emergency Stop)
                                ↓
                   NET PNL & AUDIT PERSISTENCE
```

---

## 💰 Capital Management & Currency Settings
- **User Amount in PKR**: The user decides the trade amount (e.g. `PKR 5,000`). The bot never risks an arbitrary percentage of user funds.
- **Real-Time FX Converter**: Converts PKR to USDT using live rate (default: `1 USDT = 278.50 PKR`).
- **Full Friction Transparency**: Displays entry fee, exit fee, estimated slippage, net profit, and max controlled loss.
- **Target Feasibility Guard**: If requested profit requires an unrealistic short-term move (>1.5% in 2-3 mins), the trade is rejected with an explicit feasibility warning.

---

## 🛡️ Capital Preservation Safeguards
- **3-Consecutive Loss Circuit Breaker**: If 3 consecutive trades exit at a loss, Auto Trading immediately pauses. User manual action is required to resume.
- **Daily Drawdown Limit**: Automatically halts if daily net loss reaches `$20.00 USDT`.
- **One Open Position Limit**: Prevents overlapping risk exposure.
- **Single-Click Emergency Stop**: Immediately closes active position at market, cancels pending orders, and stops trading.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.11+
- Node.js 18+ and npm

### 2. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your credentials:
```env
OPENAI_API_KEY=your_key_here
BINANCE_API_KEY=your_binance_key (optional for live trading)
BINANCE_API_SECRET=your_binance_secret (optional for live trading)
PKR_USDT_RATE=278.50
USER_AMOUNT_PKR=5000.0
MIN_DESIRED_PROFIT_PKR=50.0
PAPER_TRADING=true
```

### 3. Run Backend Server
```bash
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```
The server will start at `http://127.0.0.1:8000` with the production dashboard accessible at root `/`.

### 4. Run Frontend (Development Mode)
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🧪 Running Automated Tests
Run the comprehensive test suite (technical indicators, profit calculator, risk engine, execution, ML probability):
```bash
python -m pytest tests/ -v
```

---

## 📊 Dashboard Modules
1. **Dashboard**: Live KPI cards (Balance, Today's Net PnL, Win Rate, Trades Today, Loss Streak), Capital Calculator, Best Opportunity, Active Scalp Monitor, Market Scanner Table, and Recent Trades.
2. **Market Scanner**: Live stream of 750+ Binance USDT pairs filtered into top high-frequency scalping candidates.
3. **Active Trades**: Real-time position tracker with live PnL, duration countdown, progress bar, and manual market exit.
4. **Trade History**: Full audit trail of completed scalps with gross PnL, fee breakdown, net PnL, and exit reason.
5. **Analytics**: Cumulative PnL charts, win rate analytics, average duration, and exit trigger breakdown.
6. **AI Analysis**: 5-layer pipeline breakdown, ML directional probability distributions, and AI reasoning logs.
7. **Risk Management**: Circuit breaker controls, daily drawdown monitors, and emergency stop protocol.
8. **Backtesting**: 1-minute historical candle simulation on any Binance symbol.
9. **Settings**: Binance credentials, OpenAI key, exchange rate, and paper balance reset.
10. **System Logs**: Live state machine event stream.

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, Boolean
from backend.src.database.db import Base

class TradeRecord(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    symbol = Column(String(20), index=True, nullable=False)
    direction = Column(String(10), nullable=False) # LONG or SHORT
    mode = Column(String(10), nullable=False, default="PAPER") # PAPER or LIVE
    status = Column(String(15), nullable=False, default="OPEN") # OPEN, CLOSED, CANCELLED

    amount_pkr = Column(Float, nullable=False)
    amount_usdt = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)

    gross_pnl_usdt = Column(Float, default=0.0)
    entry_fee_usdt = Column(Float, default=0.0)
    exit_fee_usdt = Column(Float, default=0.0)
    slippage_usdt = Column(Float, default=0.0)
    net_pnl_usdt = Column(Float, default=0.0)
    net_pnl_pkr = Column(Float, default=0.0)

    entry_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    exit_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, default=0)

    signal_score = Column(Float, default=0.0)
    ml_probability = Column(Float, default=0.0)
    ai_confidence = Column(Float, default=0.0)
    exit_reason = Column(String(30), nullable=True) # TAKE_PROFIT, STOP_LOSS, TIME_EXIT, EMERGENCY_STOP, MANUAL
    reconciliation_order_id = Column(String(64), nullable=True)

class SignalRecord(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    symbol = Column(String(20), index=True, nullable=False)
    technical_score = Column(Float, nullable=False)
    ml_probability_up = Column(Float, nullable=False)
    ml_probability_down = Column(Float, nullable=False)
    ai_decision = Column(String(15), nullable=False)
    ai_confidence = Column(Float, nullable=False)
    final_quality_score = Column(Float, nullable=False)
    market_regime = Column(String(30), nullable=False)
    action = Column(String(15), nullable=False) # APPROVED, REJECTED
    rejection_reason = Column(String(100), nullable=True)

class RiskEventRecord(Base):
    __tablename__ = "risk_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    event_type = Column(String(50), nullable=False) # CONSECUTIVE_LOSSES_PAUSE, DAILY_LIMIT_REACHED, EMERGENCY_STOP
    details = Column(Text, nullable=False)

class AuditLogRecord(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    level = Column(String(10), nullable=False) # INFO, WARN, ERROR
    module = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)

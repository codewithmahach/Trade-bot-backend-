import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

class MT5Connector:
    def __init__(self):
        self.connected = False
        
    def connect(self) -> bool:
        """Initialize connection to MT5 terminal."""
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            return False
        self.connected = True
        return True

    def get_historical_data(self, symbol: str, timeframe: int, num_candles: int) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        if not self.connected:
            self.connect()
            
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
        
    def execute_trade(self, symbol: str, side: str, lot_size: float) -> dict:
        """Execute a REAL live market order on MT5."""
        if not self.connected:
            self.connect()
            
        # Prepare the request
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {"status": "error", "message": f"{symbol} not found in MT5"}
            
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return {"status": "error", "message": f"Failed to select {symbol}"}
                
        order_type = mt5.ORDER_TYPE_BUY if side.upper() == 'BUY' else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "AI AutoTrade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"status": "error", "message": f"MT5 Execution Failed: {result.comment}"}
            
        return {"status": "success", "order_id": result.order, "price": result.price}

    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False

mt5_connector = MT5Connector()

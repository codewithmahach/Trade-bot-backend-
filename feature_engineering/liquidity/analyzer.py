import pandas as pd
import numpy as np

class LiquidityAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def detect_fvg(self):
        """
        Detects Fair Value Gaps (Bullish and Bearish).
        A Bullish FVG occurs when Candle 1 High < Candle 3 Low.
        """
        self.df['bullish_fvg'] = False
        self.df['bearish_fvg'] = False
        
        for i in range(2, len(self.df)):
            c1_high = self.df['high'].iloc[i-2]
            c3_low = self.df['low'].iloc[i]
            
            c1_low = self.df['low'].iloc[i-2]
            c3_high = self.df['high'].iloc[i]
            
            # Bullish FVG
            if c3_low > c1_high:
                self.df.at[self.df.index[i], 'bullish_fvg'] = True
                
            # Bearish FVG
            if c3_high < c1_low:
                self.df.at[self.df.index[i], 'bearish_fvg'] = True
                
        return self.df
        
    def detect_liquidity_sweeps(self, swing_highs: pd.Series, swing_lows: pd.Series):
        """
        Detects if current price action sweeps a previous high/low but closes within range (Rejection).
        """
        self.df['liquidity_sweep'] = "NONE"
        
        # Simplified: check if current high breaks previous swing high but closes below it
        for i in range(1, len(self.df)):
            # Fakeout / Sweep of High
            if self.df['high'].iloc[i] > swing_highs.iloc[i-1] and self.df['close'].iloc[i] < swing_highs.iloc[i-1]:
                self.df.at[self.df.index[i], 'liquidity_sweep'] = "SWEEP_HIGH"
                
            # Fakeout / Sweep of Low
            if self.df['low'].iloc[i] < swing_lows.iloc[i-1] and self.df['close'].iloc[i] > swing_lows.iloc[i-1]:
                self.df.at[self.df.index[i], 'liquidity_sweep'] = "SWEEP_LOW"
                
        return self.df

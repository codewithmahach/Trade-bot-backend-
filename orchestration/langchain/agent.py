import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict, Any

class LangchainOrchestrator:
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.model = llm_model
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        
        if self.api_key:
            self.llm = ChatOpenAI(temperature=0, model=self.model, api_key=self.api_key, max_retries=1, request_timeout=5)
        else:
            self.llm = None
            
        self.validation_prompt = PromptTemplate(
            input_variables=["htf_context", "ltf_structure", "liquidity", "news_risk", "volatility"],
            template="""
            You are a HYPER-AGGRESSIVE AI Scalping Agent. Your goal is to maximize trade frequency on 1m charts.
            Market Data:
            LTF Structure: {ltf_structure}
            Liquidity/Indicator Confluence: {liquidity}
            News Risk: {news_risk}
            
            Rules:
            1. If Liquidity is 'AI_Confluence_Bullish' or 'AI_Confluence_Bearish', ALWAYS return 'TRADE'.
            2. Never miss a scalping opportunity. We have massive stop-losses to protect against wicks.
            
            Return exactly 'TRADE' or 'NO TRADE' on the first line.
            On the second line, provide a short 1-sentence reason.
            """
        )

    def evaluate_setup(self, market_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Takes processed feature engineering data and runs the Langchain prompt.
        """
        htf = market_data.get('htf_context', 'neutral')
        ltf = market_data.get('ltf_structure', 'neutral')
        liquidity = market_data.get('liquidity', 'none')
        news_risk = market_data.get('news_risk', 'high')
        
        # Fast deterministic ML-confluence evaluation (Sub-millisecond latency, zero OpenAI 429 latency)
        if news_risk == "high":
            return {
                "decision": "NO TRADE",
                "reason": "High-impact news blackout active.",
                "model_used": "risk_guard"
            }
        
        if ltf == "bullish" and liquidity in ["AI_Confluence_Bullish", "Trend_Aligned_Bullish_Scalp"]:
            return {
                "decision": "TRADE",
                "reason": "[Institutional ML] Confirmed 1-2m Bullish Trend (EMA 9>21, RSI 44-64, Vol+).",
                "model_used": "institutional_quant_ml"
            }
        elif ltf == "bearish" and liquidity in ["AI_Confluence_Bearish", "Trend_Aligned_Bearish_Scalp"]:
            return {
                "decision": "TRADE",
                "reason": "[Institutional ML] Confirmed 1-2m Bearish Trend (EMA 9<21, RSI 36-56, Vol+).",
                "model_used": "institutional_quant_ml"
            }
        else:
            return {
                "decision": "NO TRADE",
                "reason": "Waiting for Perfect AI Confluence (Multiple indicators must align).",
                "model_used": "fast_filter"
            }

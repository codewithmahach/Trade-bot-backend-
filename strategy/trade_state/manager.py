from typing import Optional, Dict, Any

class TradeStateManager:
    """
    Implements the ONE-TRADE-AT-A-TIME Engine logic.
    """
    def __init__(self):
        self.active_position = False
        self.current_trade: Optional[Dict[str, Any]] = None
        
    def check_can_trade(self) -> bool:
        """Returns True if the system is allowed to open a new trade."""
        return not self.active_position
        
    def open_trade(self, setup: Dict[str, Any]) -> bool:
        """Registers a new trade if allowed."""
        if self.active_position:
            return False
            
        self.active_position = True
        self.current_trade = setup
        self.current_trade['status'] = 'open'
        return True
        
    def close_trade(self, reason: str = "target_reached") -> bool:
        """Closes the current trade and frees up the engine."""
        if not self.active_position:
            return False
            
        self.current_trade['status'] = 'closed'
        self.current_trade['exit_reason'] = reason
        self.active_position = False
        self.current_trade = None
        return True

    def get_state(self) -> Dict[str, Any]:
        return {
            "active_position": self.active_position,
            "current_trade": self.current_trade
        }

trade_state_manager = TradeStateManager()

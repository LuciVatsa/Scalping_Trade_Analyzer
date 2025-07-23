from abc import ABC, abstractmethod
import logging
from typing import Tuple, List
from app.core.config import TradingConfig, ScalpingConstants
from app.core.models.signals import ScalpingInputs

class ScalpingStrategy(ABC):
    """Abstract base class for all scalping strategies."""
    def __init__(self, config: TradingConfig, constants: ScalpingConstants):
        self.config = config
        self.constants = constants
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def calculate_score(self, inputs: ScalpingInputs) -> Tuple[float, List[str]]:
        """Calculates a strategy-specific score."""
        pass
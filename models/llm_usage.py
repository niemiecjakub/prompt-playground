from decimal import Decimal
from utils import CostCalculator

class LlmUsage:
    def __init__(self, tokens: int, cost: Decimal):
        self.tokens = tokens
        self.cost = cost

    @classmethod
    def from_tokens_and_price(cls, tokens: int, price: Decimal) -> "LlmUsage":
        cost = CostCalculator.calculate(tokens, price)
        return cls(tokens, cost)

    @classmethod
    def from_tokens_and_cost(cls, tokens: int, cost: Decimal) -> "LlmUsage":
        return cls(tokens, cost)

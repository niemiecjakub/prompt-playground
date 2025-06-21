from decimal import Decimal, ROUND_HALF_UP

class CostCalculator:

    @staticmethod
    def calculate(tokens_used: int, token_price: Decimal) -> Decimal:
        cost = (Decimal(tokens_used) / Decimal("1000000")) * token_price
        return cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

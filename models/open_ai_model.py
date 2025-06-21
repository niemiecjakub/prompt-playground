from decimal import Decimal

class OpenAIModel:
    def __init__(self, model_id: str, input_token_price: Decimal, output_token_price: Decimal) -> None:
        self.model_id = model_id
        self.input_token_price = input_token_price
        self.output_token_price = output_token_price

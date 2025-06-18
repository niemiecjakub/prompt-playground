class OpenAIModel:
    def __init__(self, model_id: str, input_token_price: float, output_token_price: float) -> None:
        self.model_id = model_id
        self.input_token_price = input_token_price
        self.output_token_price = output_token_price

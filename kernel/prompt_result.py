from models import LlmUsage
from .open_ai_models import OpenAIModels

class PromptResult:
    def __init__(self, model_id: str, answer: str, input_tokens: int, output_tokens: int):
        self.answer = answer
        model = OpenAIModels.get_model_by_id(model_id=model_id)
        if model is None:
            raise ValueError(f"Model with id '{model_id}' not found.")
        self.input = LlmUsage.from_tokens_and_price(input_tokens, model.input_token_price)
        self.output = LlmUsage.from_tokens_and_price(output_tokens, model.output_token_price)
        self.total = LlmUsage.from_tokens_and_cost(
            tokens=self.input.tokens +self.output.tokens, 
            cost=self.input.cost+self.output.cost
        )

from models import OpenAIModel
from decimal import Decimal

class OpenAIModels:
    _models: list[OpenAIModel] = [
        OpenAIModel(model_id="gpt-4.1", input_token_price=Decimal("2.0"), output_token_price=Decimal("8.0")),
        OpenAIModel(model_id="gpt-4.1-mini", input_token_price=Decimal("0.4"), output_token_price=Decimal("1.6")),
        OpenAIModel(model_id="gpt-4.1-nano", input_token_price=Decimal("0.1"), output_token_price=Decimal("0.4")),
        OpenAIModel(model_id="gpt-4", input_token_price=Decimal("30.0"), output_token_price=Decimal("60.0")),
        OpenAIModel(model_id="gpt-4o", input_token_price=Decimal("2.5"), output_token_price=Decimal("10.0")),
        OpenAIModel(model_id="gpt-4o-mini", input_token_price=Decimal("0.15"), output_token_price=Decimal("0.6")),
        OpenAIModel(model_id="gpt-4-turbo", input_token_price=Decimal("10.0"), output_token_price=Decimal("30.0")),
        OpenAIModel(model_id="gpt-3.5-turbo", input_token_price=Decimal("0.5"), output_token_price=Decimal("1.5")),
    ]

    @classmethod
    def get_model_ids(cls) -> list[str]:
        return [model.model_id for model in cls._models]

    @classmethod
    def get_model_by_id(cls, model_id: str) -> OpenAIModel | None:
        return next((model for model in cls._models if model.model_id == model_id), None)

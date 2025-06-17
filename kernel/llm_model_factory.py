import re
from openai import OpenAI

class LLMModelFactory:
    
    @classmethod
    def get_available_openai_models(cls, api_key: str, show_dated_and_preview_models: bool = False) -> list[str]:
        client = OpenAI(api_key=api_key)
        models = []
        for model in client.models.list():
            if cls.should_exclude_model(model.id, show_dated_and_preview_models):
                continue
            models.append(model.id)

        models.sort(key=cls.sort_key)
        return models
    
    @staticmethod
    def should_exclude_model(model_id: str, show_dated_and_preview_models: bool) -> bool:
        exclude_keywords = ["tts", "realtime", "audio", "whisper", "embedding", "davinci","image","babbage","dall-e","moderation","transcribe","instruct","search","codex"]
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        model_id_lower = model_id.lower()
        if any(keyword in model_id_lower for keyword in exclude_keywords):
            return True
        if model_id_lower.startswith("o"):
            return True
        if not show_dated_and_preview_models and (date_pattern.search(model_id_lower) or "preview" in model_id_lower):
            return True
        return False

    @staticmethod
    def sort_key(model_id: str):
        match = re.match(r"([a-zA-Z\-]+)([\d\.]+)?", model_id)
        if match:
            prefix = match.group(1)
            version_str = match.group(2)
            version = tuple(map(int, version_str.split('.'))) if version_str else ()
            return (prefix, tuple(-v for v in version))
        else:
           return (model_id, ())

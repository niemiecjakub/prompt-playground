from openai import OpenAI

class ApiKeyValidator:

    @staticmethod
    def validate(api_key: str) -> bool:
        try:
            OpenAI(api_key=api_key).models.list()
            return True
        except:
            return False
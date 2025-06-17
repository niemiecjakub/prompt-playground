from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

class KernelFactory:
    @staticmethod
    def create(api_key: str, models: list[str]) -> Kernel:
        kernel = Kernel()
        for model in models:
            kernel.add_service(OpenAIChatCompletion(
                api_key=api_key,
                ai_model_id=model,
                service_id=model,
            ))
        return kernel

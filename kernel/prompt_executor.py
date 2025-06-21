from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai import OpenAITextPromptExecutionSettings
from semantic_kernel.kernel import Kernel
from .prompt_result import PromptResult

class PromptExecutor:

    @staticmethod
    async def execute(kernel:Kernel, model_id: str, history: ChatHistory, settings: OpenAITextPromptExecutionSettings) -> PromptResult:
        chat_completion_service = kernel.get_service(model_id, ChatCompletionClientBase)
        response = await chat_completion_service.get_chat_message_content(chat_history=history, settings=settings)

        if response is None:
            return PromptResult(
                model_id=model_id,
                answer="",
                input_tokens=0,
                output_tokens=0
            )   

        input_tokens = response.metadata['usage'].prompt_tokens # type: ignore
        output_tokens = response.metadata['usage'].completion_tokens # type: ignore

        return PromptResult(
            model_id=model_id,
            answer=response.content,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

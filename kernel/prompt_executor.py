from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai import OpenAITextPromptExecutionSettings
from models import PromptResult

class PromptExecutor:

    @staticmethod
    async def execute(chat_completion_service:ChatCompletionClientBase, history: ChatHistory, settings: OpenAITextPromptExecutionSettings) -> PromptResult:
        response = await chat_completion_service.get_chat_message_content(chat_history=history, settings=settings)
        result = PromptResult()
        result.prompt_tokens = response.metadata['usage'].prompt_tokens # type: ignore
        result.completion_tokens = response.metadata['usage'].completion_tokens # type: ignore
        result.total_tokens = result.completion_tokens + result.prompt_tokens
        result.answer = response.content # type: ignore
        return result


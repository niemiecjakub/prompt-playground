import asyncio
from kernel.kernel_factory import KernelFactory
from kernel.prompt_executor import PromptExecutor
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from models.prompt_result import PromptResult
from semantic_kernel.connectors.ai.open_ai import OpenAITextPromptExecutionSettings

async def main():
    api_key = "api_key_placeholder"
    models = ["gpt-4.1-mini",  "gpt-4.1-nano"]

    history = ChatHistory()
    history.add_system_message("Kindly greet the user and introduce yourself")
    history.add_user_message("Hey, nice to meet you!")

    kernel = KernelFactory.create(api_key, models)
    settings = OpenAITextPromptExecutionSettings()
    settings.temperature = 0.5
    settings.max_tokens = 100
    
    tasks = []
    for model in models:
        chat_completion_service = kernel.get_service(model, ChatCompletionClientBase)
        tasks.append(PromptExecutor.execute(chat_completion_service, history, settings))

    answers = await asyncio.gather(*tasks)
    for model, answer in zip(models, answers):
        print(f"----{model}----")
        print(answer.answer)
        print(f"####################")


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import streamlit as st
from resources.spinner import spinner_html
from kernel.kernel_factory import KernelFactory
from kernel.prompt_executor import PromptExecutor
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai import OpenAITextPromptExecutionSettings
from kernel.llm_model_factory import LLMModelFactory

st.set_page_config(layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("## Credentials")
    api_key = st.text_input("Open AI API Key", type="password")

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("System message")
    system_message = st.text_area("System message", "You are a helpful assistant", label_visibility="collapsed")

    st.markdown("User message")
    user_message = st.text_area("User message", "",label_visibility="collapsed")

    st.markdown("Models")
    include_dated_and_preview_models = st.checkbox("Show dated and preview models", value=False)
    models = st.multiselect("Models", LLMModelFactory.get_available_openai_models(api_key, include_dated_and_preview_models), label_visibility="collapsed")

    st.markdown("Temperature")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.7, label_visibility="collapsed")
    
    st.markdown("Max tokens")
    max_tokens = st.number_input("Max tokens", min_value=0, max_value=1000000, step=100, value=1500, label_visibility="collapsed")

    run_button = st.button("Run", use_container_width=True)
        

if run_button and api_key and models and user_message:

    async def run_prompt():
        history = ChatHistory()
        history.add_system_message(system_message)
        history.add_user_message(user_message)

        kernel = KernelFactory.create(api_key, models)
        settings = OpenAITextPromptExecutionSettings()
        settings.temperature = temperature
        settings.max_tokens = max_tokens
        
        tasks = []
        for model in models:
            chat_completion_service = kernel.get_service(model, ChatCompletionClientBase)
            tasks.append(PromptExecutor.execute(chat_completion_service, history, settings))

        with col2:
            spinner_placeholder = st.empty()
            spinner_placeholder.markdown(spinner_html, unsafe_allow_html=True)
            answers = await asyncio.gather(*tasks)

        spinner_placeholder.empty()

        with col2:
            cols = st.tabs(models)
            if answers:
                for col, model, answer in zip(cols, models, answers):
                    with col:
                        st.markdown(answer.answer)
                        st.code(f"""
                        Prompt tokens: {answer.prompt_tokens}
                        Completion tokens: {answer.completion_tokens}
                        Total tokens: {answer.total_tokens}
                        """)

    asyncio.run(run_prompt())

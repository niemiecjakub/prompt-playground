import asyncio
import streamlit as st
from resources import spinner_html
from kernel import KernelFactory, OpenAIModels, PromptExecutor, PromptResult
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai import OpenAITextPromptExecutionSettings

st.set_page_config(layout="wide")

with st.sidebar:
    st.markdown("## Secrets & API keys")
    st.write("Your API keys are stored in your browser and used solely to communicate directly to services")
    api_key = st.text_input("Open AI API Key", type="password")

col1, col2 = st.columns([2, 3])

with col1:
    system_message = st.text_area("System message", "You are a helpful assistant", )
    user_message = st.text_area("User message", "",) 
    models = st.multiselect("Models", OpenAIModels.get_model_ids())
    max_tokens = st.number_input("Max tokens", min_value=0, max_value=1000000, step=100, value=1500, )
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, step=0.1, value=0.7 )
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, step=0.1, value=1.0 )
    frequency_penalty = st.slider("Frequency penalty", min_value=-2.0, max_value=2.0, step=0.1, value=0.0 )
    presence_penalty = st.slider("Presence penalty", min_value=-2.0, max_value=2.0, step=0.1, value=0.0 )
    run_button = st.button("Run", use_container_width=True)
        

if run_button:
    if api_key and models and user_message:

        async def run_prompt():
            history = ChatHistory()
            history.add_system_message(system_message)
            history.add_user_message(user_message)

            kernel = KernelFactory.create(api_key, models)
            settings = OpenAITextPromptExecutionSettings(
                temperature=temperature, 
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            tasks = []
            for model in models:
                tasks.append(PromptExecutor.execute(kernel,model, history, settings))

            with col2:
                spinner_placeholder = st.empty()
                spinner_placeholder.markdown(spinner_html, unsafe_allow_html=True)
                answers : list[PromptResult] = await asyncio.gather(*tasks)

            spinner_placeholder.empty()

            with col2:
                cols = st.tabs(models)
                if answers:
                    for col, model, answer in zip(cols, models, answers):
                        with col:
                            st.markdown(answer.answer)
                            st.code(f"""
                            Input: {answer.input.tokens} tokens (${answer.input.cost})
                            Output: {answer.output.tokens} tokens (${answer.output.cost})
                            Total: {answer.total.tokens} tokens (${answer.total.cost})
                            """)

        asyncio.run(run_prompt())

    else:
        with col2:
            if not api_key:
                st.warning("Please enter your OpenAI API key.")
            if not models:
                st.warning("Please select at least one model.")
            if not user_message:
                st.warning("Please enter a user message.")

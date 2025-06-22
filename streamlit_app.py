import asyncio
import streamlit as st
from utils import spinner_html
from kernel import KernelFactory, OpenAIModels, PromptExecutor, PromptResult
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai import OpenAITextPromptExecutionSettings

st.set_page_config(
    page_title="Prompt playground",
    page_icon="📊",
    initial_sidebar_state="expanded",
    layout="wide",
)

st.markdown("""
    # Prompt playground
                    
    📊 Try your prompts with various models and configurations
    """, unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("## 🔐 Secrets & API keys")
    st.write("Your API keys are stored in your browser and used solely to communicate directly to services")
    api_key = st.text_input("Open AI API Key", type="password")

col1, col2 = st.columns([2, 3])

with col1:
    system_message = st.text_area("💬 System message", "You are a helpful assistant", placeholder="System message")
    user_message = st.text_area("💬 User message", "", placeholder="User message")
    models = st.multiselect("🤖 Models", OpenAIModels.get_model_ids())
    expander = st.expander("Settings ", icon="🛠️")
    max_tokens = expander.number_input("Max tokens", min_value=0, max_value=1000000, step=100, value=1500)
    temperature = expander.slider("Temperature", min_value=0.0, max_value=2.0, step=0.1, value=0.7)
    top_p = expander.slider("Top P", min_value=0.0, max_value=1.0, step=0.1, value=1.0)
    frequency_penalty = expander.slider("Frequency penalty", min_value=-2.0, max_value=2.0, step=0.1, value=0.0)
    presence_penalty = expander.slider("Presence penalty", min_value=-2.0, max_value=2.0, step=0.1, value=0.0)
    run_button = st.button("Run", use_container_width=True)

with col2:
    results_info = st.empty()
    spinner_placeholder = st.empty()

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
    tasks = [PromptExecutor.execute(kernel, model, history, settings) for model in models]
    spinner_placeholder.markdown(spinner_html, unsafe_allow_html=True)
    results: list[PromptResult] = await asyncio.gather(*tasks)
    spinner_placeholder.empty()
    st.session_state["results"] = results

if run_button:
    if api_key and models and user_message:
        try:
            asyncio.run(run_prompt())
            st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)
        except Exception as e:
            spinner_placeholder.empty()
            results_info.error(f"❌ Error occurred: {str(e)}")
    else:
        with col2:
            if not api_key:
                st.warning("Please enter your OpenAI API key.", icon="🔐")
            if not models:
                st.warning("Please select at least one model.", icon="🤖")
            if not user_message:
                st.warning("Please enter a user message.", icon="💬")


if "results" in st.session_state:
    results: list[PromptResult] = st.session_state["results"]
    with col2:
        cols = st.tabs([result.model_id for result in results])
        for col, result in zip(cols, results):
            with col:
                st.markdown(result.answer)
                st.code(f"""
                Input: {result.input.tokens} tokens (${result.input.cost})
                Output: {result.output.tokens} tokens (${result.output.cost})
                Total: {result.total.tokens} tokens (${result.total.cost})
                """)
else:
    results_info.info("Your prompt results will be displayed here ...")
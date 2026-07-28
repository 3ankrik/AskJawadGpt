import os
import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

#streamlit run RagAnyAnswer.py

#model_path = hf_hub_download(
#    repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
#    filename="Llama-3.2-1B-Instruct-Q8_0.gguf"
#)
model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    filename="qwen2.5-0.5b-instruct-q4_0.gguf"
)

###repo_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
###filename = "qwen2.5-0.5b-instruct-q4_0.gguf"

#LOCAL_QWEN_GGUF = "C:/Users/jawad/PycharmProjects/rag310/models/Llama-3.2-1B-Instruct-Q8_0.gguf"
LOCAL_QWEN_GGUF = model_path

st.set_page_config(page_title="JawadGpt", page_icon="🤖")
st.title("🤖 JawadGpt")
st.caption("Ask me anything.")

@st.cache_resource
def load_model():
    return Llama(
        model_path=LOCAL_QWEN_GGUF,
        n_ctx=2048,
        verbose=False
    )

#####llm = load_model()
@st.cache_resource
def load_model():
    return Llama(
        model_path=model_path,
        n_ctx=2048,
        verbose=False
    )

llm = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Ask me anything."
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            full_prompt = f"""

You are a helpful AI assistant.



{prompt}


"""

            output = llm(
                full_prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stop=[""]
            )

            answer = output["choices"][0]["text"].strip()

            st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
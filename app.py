
import uuid
import streamlit as st
from chatbot import ask_constitution

# Page configuration
st.set_page_config(
    page_title="Constitution Chatbot",
    page_icon="⚖️",
    layout="centered"
)

# Initialize conversation
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Application heading
st.title("⚖️ Constitution of India")
st.caption(
    "Ask questions about constitutional Articles, rights, and provisions."
)

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "Your AI assistant for learning about the "
        "Constitution of India in simple language."
    )

    st.divider()

    st.caption("📚 Knowledge Base Answer — based on retrieved PDF passages.")
    st.caption("✨ AI-Generated Answer — generated without sufficient retrieved evidence.")

    if st.button("New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

# Display previous messages, including source labels
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            if message.get("source_type"):
                st.caption(f'📌 {message["source_type"]}')

            for source in message.get("sources", []):
                if source.get("page") is not None:
                    st.caption(
                        f'{source["document"]} — '
                        f'Page {source["page"]}'
                    )
                else:
                    st.caption(source["document"])

# Chat input
question = st.chat_input("Ask about the Indian Constitution...")

if question:
    # Save and display the current question
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Previous messages, excluding the current question
    previous_history = st.session_state.chat_history[:-1]

    with st.chat_message("assistant"):
        with st.spinner(
            "Searching the knowledge base and generating answer..."
        ):
            try:
                result = ask_constitution(
                    question,
                    st.session_state.thread_id,
                    previous_history
                )

                answer = result["answer"]
                source_type = result.get(
                    "source_type", "AI-Generated Answer"
                )
                sources = result.get("sources", [])

                # Display answer
                st.markdown(answer)
                st.caption(f"📌 {source_type}")

                # Display source references
                for source in sources:
                    if source.get("page") is not None:
                        st.caption(
                            f'{source["document"]} — '
                            f'Page {source["page"]}'
                        )
                    else:
                        st.caption(source["document"])

                # Save answer and source information
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "source_type": source_type,
                    "sources": sources
                })

            except Exception as e:
                st.error(
                    "Sorry, the chatbot failed to respond. "
                    "Please check the terminal for the error."
                )
                print(f"Chatbot error: {e}")

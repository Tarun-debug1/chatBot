
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# --------------------------------------------------
# 1. Gemini LLM: answer generation
# --------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_retries=1,
    timeout=60,
)

SYSTEM_PROMPT = """
You are an assistant specializing in the Constitution of India.

Instructions:
- Answer constitutional questions accurately and in simple language.
- Use the supplied knowledge-base context when it is relevant.
- Never invent Articles, legal provisions, quotations, or citations.
- If the supplied context is insufficient, acknowledge the limitation.
- Answer follow-up questions using the conversation history.
- If the user asks to summarize previous content, summarize that content
  instead of searching for a new, unrelated answer.
- Politely reject questions unrelated to the Indian Constitution.
"""

# --------------------------------------------------
# 2. MiniLM: local embeddings
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# --------------------------------------------------
# 3. Load ChromaDB knowledge base
# --------------------------------------------------
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

# Initial threshold; validate it against test questions.
DISTANCE_THRESHOLD = 1.0


def format_history(chat_history: list) -> str:
    """Format recent Streamlit conversation history."""
    if not chat_history:
        return "No previous conversation."

    recent_messages = chat_history[-8:]

    return "\n".join(
        f'{message["role"].upper()}: {message["content"]}'
        for message in recent_messages
    )


def ask_constitution(
    question: str,
    thread_id: str,
    chat_history: list = None,
) -> dict:
    """
    Answer using retrieved Constitution passages when relevant.
    Otherwise, use Gemini's general knowledge with a distinct label.
    """

    chat_history = chat_history or []
    history_text = format_history(chat_history)

    # --------------------------------------------------
    # 4. Detect follow-up questions
    # --------------------------------------------------
    follow_up_prompt = f"""
Given the conversation and the latest question, rewrite the latest
question as a standalone question that includes necessary context.

If the user asks to summarize, shorten, explain, or elaborate on a
previous answer, preserve that intent.

CONVERSATION:
{history_text}

LATEST QUESTION:
{question}

Return only the rewritten question.
"""

    try:
        standalone_response = llm.invoke([
            SystemMessage(
                content="Rewrite follow-up questions clearly and concisely."
            ),
            HumanMessage(content=follow_up_prompt),
        ])
        search_question = standalone_response.content

        if isinstance(search_question, list):
            search_question = "\n".join(
                item["text"]
                for item in search_question
                if isinstance(item, dict) and item.get("type") == "text"
            )

        search_question = str(search_question).strip() or question

    except Exception:
        # Continue with the original question if rewriting fails.
        search_question = question

    # --------------------------------------------------
    # 5. Retrieve relevant passages
    # --------------------------------------------------
    scored_docs = vector_store.similarity_search_with_score(
        search_question,
        k=4,
    )

    relevant_docs = [
        (doc, distance)
        for doc, distance in scored_docs
        if distance < DISTANCE_THRESHOLD
    ]

    sources = []
    context_parts = []

    for doc, distance in relevant_docs:
        page = doc.metadata.get("page")

        # PyPDFLoader normally stores page numbers starting at zero.
        page_number = page + 1 if isinstance(page, int) else None

        if page_number is not None:
            source_label = f"Constitution PDF, Page {page_number}"
        else:
            source_label = "Constitution PDF, page unavailable"

        context_parts.append(
            f"[{source_label}]\n{doc.page_content}"
        )

        source = {
            "document": "Constitution PDF",
            "page": page_number,
        }

        # Avoid duplicate page references.
        if source not in sources:
            sources.append(source)

    context = "\n\n".join(context_parts)

    # --------------------------------------------------
    # 6. Prepare prompt and source label
    # --------------------------------------------------
    if relevant_docs:
        source_type = "Knowledge Base Answer"

        prompt = f"""
Use the retrieved Constitution passages to answer the latest question.

Rules:
- Answer the actual question, not just repeat the passages.
- Use simple language and be concise unless more detail is requested.
- Use previous conversation context for follow-up questions.
- Summarize previous answers when the user requests a summary.
- Do not invent facts or claim the passages support something they do not.
- If the retrieved material is insufficient, state that clearly.
- Do not invent citations. Source references will be displayed by the app.

PREVIOUS CONVERSATION:
{history_text}

RETRIEVED CONSTITUTION PASSAGES:
{context}

LATEST USER QUESTION:
{question}
"""
    else:
        source_type = "AI-Generated Answer"
        sources = []

        prompt = f"""
Answer the user's latest question only if you can do so reliably.

Rules:
- You specialize in the Constitution of India.
- Use previous conversation context for follow-up questions.
- If the user requests a summary of a previous answer, summarize it.
- Do not invent Articles, legal provisions, or citations.
- If you are uncertain, say that you cannot verify the answer.
- Politely reject questions unrelated to the Indian Constitution.
- No knowledge-base passages were sufficiently relevant to this question.

PREVIOUS CONVERSATION:
{history_text}

LATEST USER QUESTION:
{question}
"""

    # --------------------------------------------------
    # 7. Generate answer using Gemini
    # --------------------------------------------------
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    answer = response.content

    if isinstance(answer, list):
        answer = "\n".join(
            item["text"]
            for item in answer
            if isinstance(item, dict) and item.get("type") == "text"
        )

    return {
        "answer": str(answer),
        "source_type": source_type,
        "sources": sources,
    }

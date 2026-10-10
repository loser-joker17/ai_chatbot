
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
import uuid
import re
import langgraph_backend

chatbot = langgraph_backend.chatbot
retrieve_all_threads = langgraph_backend.retrieve_all_threads
save_chat_title = langgraph_backend.save_chat_title


# ****************************************
# Utility Functions
# ****************************************

def generate_thread_id():
    return str(uuid.uuid4())


def generate_title(user_input):
    # Normalize whitespace
    text = user_input.strip().replace("\n", " ")

    # Remove common opening phrases
    text = re.sub(
        r"^(so\s+|hey\s+|hi\s+|please\s+|can you\s+|could you\s+|"
        r"tell me\s+|explain\s+|what are\s+the\s+|what is\s+|"
        r"how does\s+|how do\s+|i am learning\s+|i'm learning\s+)",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Keep the first few meaningful words
    words = re.findall(r"\b[\w+#.-]+\b", text)

    # Skip common filler words
    stop_words = {
        "the", "a", "an", "is", "are", "in", "of",
        "for", "to", "and", "what", "important"
    }

    meaningful_words = [
        word for word in words
        if word.lower() not in stop_words
    ]

    title_words = meaningful_words[:3]

    if not title_words:
        return "New Chat"

    title = " ".join(title_words)

    # Capitalize the title for display
    return title[0].upper() + title[1:]


def add_thread(thread_id, title="New Chat"):
    for thread in st.session_state["chat_threads"]:
        if thread["thread_id"] == thread_id:
            return

    st.session_state["chat_threads"].append({
        "thread_id": thread_id,
        "title": title
    })



def reset_chat():
    # If the current chat is still empty, reuse it
    if not st.session_state["message_history"]:
        return

    # Otherwise, create a new conversation
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    add_thread(thread_id)


def load_conversation(thread_id):
    try:
        state = chatbot.get_state(
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        return state.values.get("messages", [])

    except Exception:
        return []


def switch_conversation(thread_id):
    st.session_state["thread_id"] = thread_id

    messages = load_conversation(thread_id)

    temp_messages = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, (AIMessage, AIMessageChunk)):
            role = "assistant"
        else:
            # Skip tool and other unsupported message types
            continue

        if isinstance(msg.content, str):
            content = msg.content
        else:
            # Basic handling for non-string message content
            content = str(msg.content)

        temp_messages.append({
            "role": role,
            "content": content
        })

    st.session_state["message_history"] = temp_messages


# ****************************************
# Session Setup
# ****************************************

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

# Register the current conversation
add_thread(st.session_state["thread_id"])


# ****************************************
# Sidebar UI
# ****************************************

st.sidebar.title("AI Chatbot")

if st.sidebar.button("＋ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

for thread in reversed(st.session_state["chat_threads"]):
    thread_id = thread["thread_id"]
    title = thread["title"]

    # Highlight the currently selected conversation
    if thread_id == st.session_state["thread_id"]:
        button_label = f"💬 {title}  ✓"
    else:
        button_label = f"💬 {title}"

    if st.sidebar.button(
        button_label,
        key=f"thread_{thread_id}",
        use_container_width=True
    ):
        switch_conversation(thread_id)
        st.rerun()


# ****************************************
# Main Chat UI
# ****************************************

st.title("Welcome to the AI Chatbot")

# Display the selected conversation's history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# User input
user_input = st.chat_input("Type here...")


if user_input:
    current_thread_id = st.session_state["thread_id"]

    for thread in st.session_state["chat_threads"]:
        if thread["thread_id"] == current_thread_id:
            if thread["title"] == "New Chat":
                new_title = generate_title(user_input)

                thread["title"] = new_title

                save_chat_title( current_thread_id,new_title)
            break

    # Add user message to UI history
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # LangGraph thread configuration
    CONFIG = {
        "configurable": {
            "thread_id": current_thread_id
        }
    }

    # ****************************************
    # Stream Assistant Response
    # ****************************************

    with st.chat_message("assistant"):

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages"
            ):

                # Handle streamed assistant message chunks
                if isinstance(
                    message_chunk,
                    (AIMessage, AIMessageChunk)
                ):
                    # Ignore non-text content blocks
                    if isinstance(message_chunk.content, str):
                        if message_chunk.content:
                            yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    # Save assistant response in UI history
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })
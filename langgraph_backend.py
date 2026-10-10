from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import sqlite3


load_dotenv()
llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0,
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

#Creation of database 
connection = sqlite3.connect(database='chatbot.db',check_same_thread=False) #bydefault is true and we are using multiple threads

# Create table for storing chat titles
connection.execute("""
    CREATE TABLE IF NOT EXISTS chat_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT
    )
""")

connection.commit()

# Checkpointer
checkpointer = SqliteSaver(connection)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def save_chat_title(thread_id, title):
    connection.execute(
        "INSERT OR REPLACE INTO chat_titles (thread_id, title) VALUES (?, ?)",
        (thread_id, title)
    )
    connection.commit()


def get_chat_title(thread_id):
    result = connection.execute(
        "SELECT title FROM chat_titles WHERE thread_id = ?",
        (thread_id,)
    ).fetchone()

    if result:
        return result[0]

    return "New Chat"

#Extracting or retrieving all the unique threads 
def retrieve_all_threads():

    all_threads = []

    # Get all saved checkpoints from LangGraph
    for checkpoint in checkpointer.list(None):

        # Get the thread ID from the checkpoint
        thread_id = checkpoint.config["configurable"]["thread_id"]

        # Check if we already added this thread
        if thread_id not in all_threads:
            all_threads.append(thread_id)

    # Convert each thread ID into a dictionary
    thread_list = []

    for thread_id in all_threads:
        thread = {
            "thread_id": thread_id,
            "title": get_chat_title(thread_id)
        }

        thread_list.append(thread)

    return thread_list

#For Streaming instead of calling .invoke function will call .stream  and stream object provide two thing 
#message_chunk and metadata 

# thread_id = 1;
# stream = chatbot.stream(
#     {'messages': [HumanMessage(content = 'What is the receipe of Pasta')]},
#     config = {"configurable" : {'thread_id':thread_id}},
#     stream_mode='messages'
# )

# for message_chunk, metadata in stream: 
#     if(message_chunk.content):
#         print(message_chunk.content,end=" ",flush=True)

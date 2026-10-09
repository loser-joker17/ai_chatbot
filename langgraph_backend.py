from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

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

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

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
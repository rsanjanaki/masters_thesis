# chat_basic.py
import os
from dotenv import load_dotenv; load_dotenv()
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)   # requires OPENAI_API_KEY

history = [
    SystemMessage(content="You are a friendly, concise assistant.")
]
while True:
    q = input("You: ")
    history.append(HumanMessage(content=q))
    reply = llm.invoke(history)
    print("Bot:", reply.content)
    history.append(reply)

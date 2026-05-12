from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage , SystemMessage , AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V4-Flash",
task ="text-generation")

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about LangChain.")
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
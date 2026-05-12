from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage , SystemMessage , AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V4-Flash",
task ="text-generation")

model = ChatHuggingFace(llm=llm)

chat_history = [SystemMessage(content="You are a helpful AI assistant.")]

while True:
    user_input = input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat...")
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:", result.content)

print ("Chat history:" , chat_history)
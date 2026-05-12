from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V4-Flash",
task ="text-generation")

model = ChatHuggingFace(llm=llm)
result = model.invoke("how many matches rcb has won against chennai super kings?")

print(result)
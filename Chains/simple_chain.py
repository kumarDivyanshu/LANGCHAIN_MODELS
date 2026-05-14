from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template= "generate 5 interestinf fats about {topic}",
    input_variables=["topic"]
)

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

parser = StrOutputParser()

chain = prompt1 | model | parser

result = chain.invoke({"topic": "Bihar"})

# print("Facts:", result)

chain.get_graph().print_ascii()

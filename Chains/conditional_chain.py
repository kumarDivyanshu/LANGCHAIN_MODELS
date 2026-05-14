from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

model2 = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= "classify the sentiment of the following text into positive and negative \n {text}",
    input_variables=["text"]
)

classifier_chain = prompt1 | model | parser

result = classifier_chain.invoke({"text": "This is a terrible smartphone with a very bad battery life."})

print("Classification Result:", result)


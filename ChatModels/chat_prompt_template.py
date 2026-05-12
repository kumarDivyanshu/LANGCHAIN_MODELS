from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage , SystemMessage 

chat_template = ChatPromptTemplate([
    ("system", "You are a {domain} expert."),
    ("human", "Explain in simple terms, what is {query}?")
])

prompt = chat_template.invoke({"domain": "cricket" , "query": "no ball"})

print(prompt)
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V4-Flash",
task ="text-generation")

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="name of the person")
    age: int = Field(description="age of the person" , gt=18)
    city: str = Field(description="name of the city of the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template= "generate the name , age and city of a fictional {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# prompt = template.invoke({"place": "Indian"})

# result = model.invoke(prompt)

# parsed_result = parser.parse(result.content)

# print("Parsed Result:", parsed_result)

chain = template | model | parser

final_result = chain.invoke({"place": "Nepalese"})

print("Parsed Result:", final_result)
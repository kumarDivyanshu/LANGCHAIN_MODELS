from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser
# Note: ResponseSchema may still require langchain.output_parsers depending on the build
from langchain.output_parsers import ResponseSchema
#there is some import issure with StructuredOutputParser and ResponseSchema in the latest build, if you face an import error try installing langchain version 1.x+ or above

load_dotenv()

llm = HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-V4-Flash",
task ="text-generation")

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact1", description="fact1 about the topic"),
    ResponseSchema(name="fact2", description="fact2 about the topic"),
    ResponseSchema(name="fact3", description="fact3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template= "write 3 facts about the topic {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# chain = template | model | parser
# result = chain.invoke({"topic": "black hole"})

prompt = template.invoke({"topic": "black hole"})

result = model.invoke(prompt)

parsed_result = parser.parse(result.content)

print("Parsed Result:", parsed_result)
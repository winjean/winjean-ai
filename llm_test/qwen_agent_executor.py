from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_react_agent

class Person(BaseModel):
    name: str = Field(description="The person's name")
    height: float = Field(description="The person's height")
    hair_color: str = Field(description="The person's hair color")


model = OllamaFunctions(model="qwen2.5:1.5b", base_url="http://localhost:11434", format="json", verbose=True, temperature=0)
model_structured = model.with_structured_output(Person)
prompt = PromptTemplate.from_template(
    """Alex is 5 feet tall. 
    Claudia is 1 feet taller than Alex and jumps higher than him. 
    Claudia is a brunette and Alex is blonde.
    
    Human: {question}
    AI: 
    """
)

# parser = PydanticOutputParser(pydantic_object=Person)

chain = prompt | model_structured
response = chain.invoke("Describe Alex")
print(response)





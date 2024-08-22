from langchain_community.llms import OCIGenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

import oci

#Setup model
endpoint = 'Put your endpoint URL here'

llm = OCIGenAI(
    model_id = "cohere.command",
    service_endpoint = endpoint,
    compartment_id = "Put your OCID here", #replace with OCID
    model_kwargs = {"max_tokens":100}
)



response = llm.invoke("Tell me one fact about earth",temperature = 0.7)
print("Case1 Response - > " + response)

template = """You are a chatbot havinga conversation with a human. 
Human: {human_input} + {city}
"""
#Create a Prompt using the template
prompt = PromptTemplate(input_variables=["human_input","city"], template=template)

prompt_val = prompt.invoke({"human_input":"Tell us in a exciting tone about", "city":"Las Vegas"})
print("Prompt String is ->")
print(prompt_val.to_string())

#Declare chain that begins with a prompt. next llm and finally output parser
chain = prompt | llm

#Invoke a chaing and provide input question
respone = chain.invoke({"human_input":"Tell us in a exciting tone about", "city":"Las Vegas"})


print("Case2 Response ->" + response)

#Use chat Message Prompt to accept text input. Create a chat template and use HumanMessage and SystemMessage
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a chatbot that explains in steps."),
        ("ai","I shall explain in steps"),
        ("human","{input}")
    ]
)

chain = prompt | llm
respone = chain.invoke({"input":"What's the New York culture like?"})
print("Case3 Response ->" + response)

from langchain_community.llms import OCIGenAI

from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.chains import LLMChain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.prompts import PromptTemplate

#Setup model
endpoint = ' https://inference.generativeai.us-chicago-1.oci.oraclecloud.com'

llm = OCIGenAI(
    model_id = "cohere.command",
    service_endpoint = endpoint,
    compartment_id = "ocid1.compartment.oc1..aaaaaaaah3o77etbcfg2o25jxks2pucmyrz6veg26z5lgpx3q355nikleemq", #replace with OCID
    model_kwargs = {"max_tokens":100}
)

#Create a prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot who explain in steps"
        ),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

#Create a memory to remember our chat with the llm
memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
summary_memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history")

#Create a conversation chain using llm, prompt and memory
conversation = LLMChain(llm=llm,prompt=prompt,verbose=True,memory=memory) #can change which memory is being used to summary_memory as well

#Invoke a chain. Just pass in the 'question' variable = 'chat history' gets populated by memory
conversation.invoke({"question":"What is the capital of the U.S"})

#print all messages in the memory. Repeating invokes will continue printing items in memory
print(memory.chat_memory.messages)
print(summary_memory.chat_memory.messages)
print("Summary of conversation is ->"+summary_memory.buffer)


# Create a history with a key "chat messages". StreamlitChatMessageHistory will store messages in 
#Streamlit session state at the specified key=. Given StreamlitchatMessageHistory will NOT be persisted or shared across user sessions
history = StreamlitChatMessageHistory(key="chat_messages")

#Create memory object
streamLitMemory = ConversationBufferMemory(chat_memory=history)

#create template and prompt to accept question
template = """You are an AI chatbot having a conversation with a human.
Human: {human_input}
AI: 
"""
prompt = PromptTemplate(input_variables=["human_inuput"],template=template)

#create chain object
llm_chain = LLMChain(llm,prompt,streamLitMemory)

#use streamlit to print all messages in memory. create text input, run chain and question and responsne is automatically put in history
import streamlit as st

st.title('Welcome to the ChatBot')
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if x := st.chat_input():
    st.chat_message("human").write(x)
    response = llm_chain.run(x)
    st.chat_message("ai").write(response)

 
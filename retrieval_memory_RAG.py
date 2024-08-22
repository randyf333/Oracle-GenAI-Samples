from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import chromadb
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

from langchain_community.llms import OCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings

import os
from uuid import uuid4
unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Test111 = {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "Put your langsmith endpoint here"
os.environ["LANGCHAIN_API_KEY"] = "Put your API key here"

#Setup llm
llm = OCIGenAI(
    model_id = "cohere.command",
    service_endpoint = "Put endpoint here",
    compartment_id = "Put your OCID here", 
    model_kwargs = {"max_tokens":400}
)

#connect to chromadb server
clinet = chromadb.HttpClient(host = "Put host ip address here",settings=Settings(allow_reset = True))

#Create embeddings
embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="Put service endpoint here",
    compartment_id="Put OCID here"
)

#Create retriever that gets relevant documents
db = Chroma(client=clinet,embedding_function=embeddings)
retv = db.as_retriever(search_type="similarity",search_kwargs={"k":8})

#Create memory to remeber chat messages
memory = ConversationBufferMemory(llm=llm,memory_key = "chat_history",return_messages=True,output_key="answer")

#Create chain that uses llm, retriever, and memory
qa = ConversationalRetrievalChain.from_llm(llm=llm,retriever=retv,memory=memory,return_source_documents=True)

response = qa.invoke({"question":"Tell us about Oracle Cloud Infrastructure AI Foundations course"})
print(memory.chat_memory.messages)

response = qa.invoke({"question":"Which module of the course is relevant to the LLMs and Transformers"})
print(memory.chat_memory.messages)

print(response)
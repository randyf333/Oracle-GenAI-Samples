from langchain.chains import RetrievalQA
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings

#Setup OCI GenAI llm

llm = OCIGenAI(
    model_id = "cohere.command-light",
    service_endpoint = "Put endpoint here",
    compartment_id = "Put your OCID here", 
    model_kwargs = {"max_tokens":100}
)

#Connect to chromadb server
client = chromadb.HttpClient(host="Put host ip address here")

#Create embeddings
embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="Put service endpoint here",
    compartment_id="Put OCID here"
)

#Create a retriver that gets relecant documents
db = Chroma(client=client,embedding_function=embeddings)
retv = db.as_retriever(search_type = "similarity", search_kwargs = {"k":5})

#Create retrieval chain that takes llm, retriever object and invoke it to get response
chain = RetrievalQA.from_chain_type(llm=llm,retriever=retv, return_source_documents=True)

response = chain.invoke("Tell us which module is most relevant to LLMs and Generative AI")

print(response)
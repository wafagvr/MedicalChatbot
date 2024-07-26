from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app=Flask(__name__)


load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
#Initialize Pinecone Client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"


#Download Hugging face embeddings
embeddings = download_hugging_face_embeddings()

#Define Index
index_name="medical-chatbot"
index = pc.Index(index_name)
#Initialize vector store
text_field = "text"  
vectorstore = PineconeVectorStore( index, embeddings, text_field)  

# Define your query text
query_text = "What are Allergies?"

# Perform the similarity search using the Pinecone index
search_results = vectorstore.similarity_search(query_text, k=3)

print(search_results)

#Define Prompt Template
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    print(msg)
    result=qa({"query":msg})
    print(f"Response : ",result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(debug=True)
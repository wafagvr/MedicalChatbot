from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import time
from pinecone import Pinecone, ServerlessSpec
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
print("length of my chunk:", len(text_chunks))
embeddings = download_hugging_face_embeddings()

#Initialize Pinecone Client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

#Define Index
index_name="medical-chatbot"

existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

if index_name not in existing_indexes:
    print(f"Index {index_name} doesn't exist. Creating new..",index_name)
    pc.create_index(
        name=index_name,
        dimension=384, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )

    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(index_name)

time.sleep(1)

index.describe_index_stats()

vectorstore_from_texts = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks],
    index_name=index_name,
    embedding=embeddings
)
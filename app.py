from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import io
import pdfplumber
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain.schema import Document
import re
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Suppress PyPDF warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)

app = FastAPI()

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your allowed origins if necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get MongoDB URI and API key from environment variables
mongo_uri = os.getenv("MONGO_URI")
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not mongo_uri or not gemini_api_key:
    raise ValueError("MONGO_URI and GEMINI_API_KEY must be set in the .env file.")

# Connect to your MongoDB Atlas cluster
client = MongoClient(mongo_uri)
collection = client["smartfile"]["test"]

# Load the embedding model
model = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True})

# Initialize vector_store variable
vector_store = None

def regex_split_with_overlap(text, pattern, overlap_size):
    chunks = re.split(pattern, text)
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            overlap = chunks[i - 1][-overlap_size:]  # Add overlap from the previous chunk
            chunk = overlap + chunk
        overlapped_chunks.append(chunk)
    return overlapped_chunks

@app.get("/")
async def hello():
    return "Hello!"

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    global vector_store  # Declare vector_store as global
    if file.content_type == "text/plain":
        # For .txt files
        content = await file.read()
        text = content.decode("utf-8")
    elif file.content_type == "application/pdf":
        # For PDFs
        text = ""
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    elif file.content_type.startswith("image/"):
        # For images
        image = Image.open(io.BytesIO(await file.read()))
        text = pytesseract.image_to_string(image)
    else:
        return {"error": "Unsupported file type"}

    # Split text into chunks with regex
    pattern = r"\n\s*\n"  # Split on double newlines (paragraphs)
    overlap_size = 0  # No overlap
    text_chunks = regex_split_with_overlap(text, pattern, overlap_size)

    # Convert text chunks into Langchain Document objects
    docs = [Document(page_content=chunk, metadata={}) for chunk in text_chunks]

    # Store the data as vector embeddings in MongoDB Atlas
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=model,
        collection=collection,
        index_name="vect_index"
    )

    return {"extracted_text": text}

# Instantiate Atlas Vector Search as a retriever
def get_retriever():
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}  # Adjust 'k' to retrieve more documents
    )

def make_rag_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below\
                 but in case relevant information does not exist then display information from the internet and other sources. 
                Do not mention anything about relevancy about the passage.
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and conversational tone. 
    If the passage is irrelevant or doesn't directly answer the question, focus on the keywords in the query to generate a more accurate and helpful response. If needed, reformulate the query to better match the available information.\
    Always generate at least 5 points for the question unless answer is a one-liner or a keyword and in case 5 points are not available then add relevant points from the internet.\
    Prefer generating 10 points in case points are available or if many words are available provide in paragraph format.\
    In case there is a lot of information that is more than 5 points then format it well and display as many as possible\
    If the question is not relevant to the text provided then answer from internet.\
    Do not mention that the question is irrelevant to the topic in case it is.\
    Always prefer giving answers in points
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'
    
    ANSWER:
    """).format(query=query, relevant_passage=escaped)
    return prompt

def generate_answer1(prompt):
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    answer = model.generate_content(prompt)
    return answer.text

def generate_answer(query):
    # Retrieve top 2 relevant text chunks
    retriever = get_retriever()
    # Run a sample query in order of relevance
    documents = retriever.invoke(query)

    # Extract only the page content from the retrieved documents
    relevant_text = [doc.page_content for doc in documents]
    prompt = make_rag_prompt(query, relevant_passage="".join(relevant_text))  # joining the relevant chunks to create a single passage
    answer = generate_answer1(prompt)

    return answer

class SearchRequest(BaseModel):
    prompt: str

@app.post("/search")
async def search(prompt_request: SearchRequest):
    try:
        # Get the prompt from the request
        prompt = prompt_request.prompt

        # Process the prompt using the defined function
        result = generate_answer(prompt)

        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)

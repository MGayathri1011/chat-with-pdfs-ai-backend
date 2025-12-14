from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from utils.rag import ingest_pdf, chat_with_pdf
from pydantic import BaseModel
class ChatRequest(BaseModel):
    question: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    ingest_pdf(content)
    return {"message": "PDF uploaded successfully"}

@app.post("/chat")
async def chat(data: ChatRequest):
    answer = chat_with_pdf(data.question)
    return {"answer": answer}
@app.get("/")
def root():
    return {"status":"backend is running"}
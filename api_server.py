from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag_qa import generate_rag_answer

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    answer = generate_rag_answer(req.question)
    return {"question": req.question, "answer": answer}

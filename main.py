import requests
import os
from pydantic import BaseModel
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
print("DEBUG HF_API_KEY is None?", HF_API_KEY is None)

app = FastAPI()


class ChatRequest(BaseModel):
    mensagem: str


class ChatResponse(BaseModel):
    resposta: str


@app.post("/ia/chat", response_model=ChatResponse)
def chat_ia(req: ChatRequest):
    url = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
        {"role": "user", "content": req.mensagem}
        ],
        "max_tokens": 300
    }

    response = requests.post(url, json=payload, headers=headers)

    print("HF STATUS:", response.status_code)
    print("HF RAW TEXT:", response.text[:500], "\n")

    if response.status_code != 200:
        return ChatResponse(
            resposta=f"Erro ao chamar Hugging Face: {response.status_code} - {response.text}"
        )

    try:
        data = response.json()
        resposta_texto = data["choices"][0]["message"]["content"]
    except Exception as e:
        return ChatResponse(
            resposta=f"Erro ao processar resposta da IA: {e}\n\nCorpo recebido: {response.text}"
        )

    return ChatResponse(resposta=resposta_texto)

@app.on_event("startup")
def startup_event():
    print("\nAPI rodando em: http://localhost:3000/docs")
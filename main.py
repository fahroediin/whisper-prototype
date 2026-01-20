from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
from engine import engine

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Buat folder jika belum ada
os.makedirs("data/enrollment", exist_ok=True)
os.makedirs("data/sessions", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/enroll")
async def enroll(name: str = Form(...), file: UploadFile = File(...)):
    path = f"data/enrollment/{name}.wav"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    msg = engine.enroll_user(name, path)
    return {"status": msg}

@app.post("/process_meeting")
async def process_meeting(file: UploadFile = File(...)):
    path = f"data/sessions/meeting_live.wav"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Jalankan AI
    transcript = engine.process_meeting(path)
    return {"transcript": transcript}

if __name__ == "__main__":
    import uvicorn
    # Jalankan di 0.0.0.0 agar bisa diakses HP lewat IP Laptop
    uvicorn.run(app, host="0.0.0.0", port=8000)
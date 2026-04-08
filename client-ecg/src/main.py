from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app  import router as ecg_router
from chatbot import router as chatbot_router

app = FastAPI(
    title="ECG Medical API",
    description="ECG prediction + Cardiology chatbot",
    version="1.0.0"
)

# CORS (React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(ecg_router, prefix="/ecg")
app.include_router(chatbot_router, prefix="/chatbot")

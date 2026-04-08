"""
FastAPI ECG Chatbot Router
Multi-language support: English, French, Arabic
"""

from fastapi import HTTPException, UploadFile, File, APIRouter
from pydantic import BaseModel
from typing import Optional, List
import logging
from datetime import datetime
import numpy as np
from enum import Enum
import os
from dotenv import load_dotenv

# ================= ENV =================

load_dotenv(dotenv_path=".env", verbose=True)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in your .env file.")

# ================= LOGGING =================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= ROUTER =================

router = APIRouter(
    tags=["ECG Cardiology Chatbot"]
)

# ================= ENUMS & MODELS =================

class LanguageEnum(str, Enum):
    ENGLISH = "en"
    FRENCH = "fr"
    ARABIC = "ar"

class ECGCategoryEnum(str, Enum):
    ARRHYTHMIAS = "arrhythmias"
    ISCHEMIA = "ischemia"
    HYPERTROPHY = "hypertrophy"
    CONDUCTION = "conduction"
    GENERAL = "general"
    SYMPTOMS = "symptoms"
    RISK_FACTORS = "risk_factors"

class ChatRequest(BaseModel):
    message: str
    language: LanguageEnum = LanguageEnum.ENGLISH
    category: Optional[ECGCategoryEnum] = None
    conversation_history: Optional[List[dict]] = None

class ChatResponse(BaseModel):
    id: str
    role: str
    content: str
    category: Optional[ECGCategoryEnum]
    language: LanguageEnum
    timestamp: str
    sources: Optional[List[str]] = None
    confidence: Optional[float] = None

class DiagnosisResponse(BaseModel):
    patient_id: str
    predicted_condition: str
    confidence: float
    abnormalities: List[str]
    recommendations: str
    language: LanguageEnum

# ================= ECG KNOWLEDGE =================

ECG_KNOWLEDGE = {
    "arrhythmias": {
        "en": {"description": "Irregular heartbeats detected"},
        "fr": {"description": "Battements cardiaques irréguliers détectés"},
        "ar": {"description": "تم الكشف عن عدم انتظام في نبضات القلب"},
    },
    "ischemia": {
        "en": {"description": "Reduced blood flow to the heart"},
        "fr": {"description": "Réduction du flux sanguin vers le cœur"},
        "ar": {"description": "انخفاض تدفق الدم إلى القلب"},
    },
}

# ================= PLACEHOLDER DIAGNOSIS =================

async def call_nlp_diagnosis_model(ecg_data: np.ndarray) -> dict:
    return {
        "condition": "normal_sinus_rhythm",
        "confidence": 0.92,
        "abnormalities": []
    }

# ================= LLM CHAT =================

async def query_llm_chat(
    message: str,
    language: LanguageEnum,
    category: Optional[ECGCategoryEnum] = None,
    conversation_history: Optional[List[dict]] = None
) -> dict:

    from groq import Groq

    try:
        client = Groq()

        language_name = {
            LanguageEnum.ENGLISH: "English",
            LanguageEnum.FRENCH: "French",
            LanguageEnum.ARABIC: "Arabic"
        }[language]

        category_text = f" in the {category.value} category" if category else ""

        system_prompt = f"""
You are an expert cardiology assistant.
Respond in {language_name}{category_text}.
Provide short, clear medical explanations.
Always remind users to consult healthcare professionals.
Do not show internal reasoning.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]

        completion = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=messages,
        )

        answer = completion.choices[0].message.content

        response = completion.choices[0].message.content
        
        raw = completion.choices[0].message.content or ""

        # Parse thinking/answer (same as React app)
        import re
        thinking = ""
        answer = raw

        # Extract <think>...</think> block
        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        return {
            "response": answer,
            "thinking": None,  # Don't expose thinking
            "confidence": 0.85
        }

    except Exception as e:
        logger.error(f"LLM Error: {str(e)}")
        return {
            "response": "Unable to process request right now.",
            "confidence": 0.0
        }

# ================= ROUTES =================

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    try:
        message_id = f"{datetime.now().isoformat()}-{abs(hash(request.message)) % 10000}"

        llm_result = await query_llm_chat(
            message=request.message,
            language=request.language,
            category=request.category,
            conversation_history=request.conversation_history
        )

        return ChatResponse(
            id=message_id,
            role="assistant",
            content=llm_result["response"],
            category=request.category,
            language=request.language,
            timestamp=datetime.now().isoformat(),
            sources=["Cardiology Guidelines"],
            confidence=llm_result["confidence"]
        )

    except Exception as e:
        logger.error(f"Chat Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_ecg(
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    language: LanguageEnum = LanguageEnum.ENGLISH
):

    try:
        contents = await file.read()
        ecg_data = np.array([float(x) for x in contents.decode().split(",")])

        diagnosis = await call_nlp_diagnosis_model(ecg_data)

        return DiagnosisResponse(
            patient_id=patient_id or f"patient_{abs(hash(file.filename)) % 100000}",
            predicted_condition=diagnosis["condition"],
            confidence=diagnosis["confidence"],
            abnormalities=diagnosis["abnormalities"],
            recommendations="Please consult a cardiologist for full diagnosis.",
            language=language
        )

    except Exception as e:
        logger.error(f"Diagnosis Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

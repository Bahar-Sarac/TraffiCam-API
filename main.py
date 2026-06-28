# ==========================================
# 1. BÖLÜM: KÜTÜPHANELERİ ÇAĞIRMA
# ==========================================
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.responses import StreamingResponse # İşlenen resmi geri dönmek için
import cv2
import numpy as np
import io

# ==========================================
# 2. BÖLÜM: API VE GÜVENLİK AYARLARI
# ==========================================
app = FastAPI(
    title="TraffiCam-API",
    description="YOLOv8 ve OCR Tabanlı Akıllı Trafik İzleme Sistemi Backend Altyapısı",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 3. BÖLÜM: VERİ DOĞRULAMA ŞEMALARI (PYDANTIC)
# ==========================================
class TrafficAnalysisConfig(BaseModel):
    model_type: str = Field(
        "yolov8n",
        min_length=2,
        description="Kullanılacak nesne tespiti model boyutu"
    )
    confidence_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Modelin nesneleri algılama güven eşiği"
    )
    enable_ocr: bool = Field(
        True,
        description="OCR aktif edilsin mi?"
    )


# ==========================================
# 4. BÖLÜM: GET İSTEĞİ (SAĞLIK KONTROLÜ)
# ==========================================
# SORUNUN CEVABI TAM OLARAK BURASI: app tanımının altında, istediğin bir yere ekleyebilirsin.
@app.get("/", status_code=status.HTTP_200_OK)
async def system_status():
    """
    Sistemin anlık durumunu ve donanım bilgisini dönen sağlık kontrolü endpoint'i.
    """
    return {
        "status": "active",
        "system": "TraffiCam Analyzer Engine",
        "device": "CPU (Geliştirme Modu)",
        "version": "1.0.0"
    }


# ==========================================
# 5. BÖLÜM: POST İSTEĞİ (KAMERA BAŞLATMA)
# ==========================================
@app.post("/initialize-camera/", status_code=status.HTTP_201_CREATED)
async def initialize_camera(config: TrafficAnalysisConfig):
    """
    Sahadaki akıllı trafik kamerasını belirli konfigürasyonlarla başlatan endpoint.
    """
    supported_models = ["yolov8n", "yolov8s", "yolov8m"]

    if config.model_type.lower() not in supported_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Geçersiz model! Desteklenen trafik analiz modelleri: {supported_models}"
        )

    return {
        "status": "initialized",
        "message": f"TraffiCam akıllı kamera motoru {config.model_type} modeli ile başarıyla senkronize edildi.",
        "applied_configuration": config
    }

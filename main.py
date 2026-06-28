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

# ======================================================
# 6. BÖLÜM: NUMPY MATRİSİNE ÇEVİRME VE OPENCV İŞLEMLERİ
# ======================================================
@app.post("/process-image/", status_code=status.HTTP_200_OK)
async def process_image(
        file: UploadFile = File(...),
        low_threshold: int = 100,  # Dinamik parametre 1
        high_threshold: int = 200  # Dinamik parametre 2
):
    """
    Gelişmiş güvenlik filtreli, dinamik eşik değer destekli ve
    bellek korumalı OpenCV görüntü işleme endpoint'i.
    """
    # 1. KORUMA KALKANI: Dosya tipi kontrolü
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Geçersiz dosya tipi! Lütfen sadece görsel yükleyin."
        )

    # 2. KORUMA KALKANI: Dosya boyutu kontrolü (Maksimum 5 MB)
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 Megabayt
    # Dosyanın boyutunu okumak için imleci sona götürüp boyutu alıyoruz
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)  # İmleci tekrar başa alıyoruz ki okuma hatası olmasın

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_PAYLOAD_TOO_LARGE,
            detail="Dosya boyutu çok büyük! Maksimum sınır 5 MB'tır."
        )

    try:
        # 3. VERİ OKUMA VE DÖNÜŞÜM
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 4. KORUMA KALKANI: Bozuk/Kırık görsel kontrolü
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Görsel kodu çözülemedi. Dosya bozuk veya okunamaz durumda."
            )

        # 5. DİNAMİK OPENCV İŞLEME
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Kullanıcının URL'den gönderdiği low ve high threshold değerlerini kullanıyoruz
        edges = cv2.Canny(gray_img, low_threshold, high_threshold)

        # 6. TRANSFER VE ÇIKTI
        _, encoded_img = cv2.imencode(".png", edges)
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sistem hatası: {str(e)}"
        )
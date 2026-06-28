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
        low_threshold: int = 100,
        high_threshold: int = 200
):
    """
    Gelişmiş güvenlik filtreli, dinamik eşik değer destekli ve
    donanım maliyetlerini düşüren akıllı resize optimizasyonlu OpenCV görüntü işleme endpoint'i.
    """
    # 1. DOSYA TİPİ DOĞRULAMASI: Sadece görseller
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Geçersiz dosya tipi! Lütfen sadece görsel formatında bir dosya yükleyin."
        )

    # 2. BOYUT SINIRLANDIRMA: Maksimum 5 MB, RAM ve sunucu güvenliği için
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 Megabayt
    file.file.seek(0, 2)  # İmleci dosyanın sonuna götür
    file_size = file.file.tell()  # Toplam boyutu oku
    file.file.seek(0)  # İmleci tekrar başa al (Okuma hatasını önlemek için kritik)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_PAYLOAD_TOO_LARGE,
            detail="Dosya boyutu çok büyük! Sunucu güvenliği için maksimum sınır 5 MB'dır."
        )

    try:
        # 3. VERİ OKUMA: Ham byte yığınını okuma ve NumPy matrisine dönüştürme
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 4. BOZUK/KIRIK GÖRSEL KONTROLÜ
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Görsel kodu çözülemedi. Dosya bozuk veya okunamaz durumda."
            )

        # 5. PERFORMANS OPTİMİZASYONU: Downscaling
        # Devasa çözünürlüklerde (4K/2K) piksel yoğunluğunu azaltarak işlem hızını 3-4 kat artırıyoruz.
        h, w = img.shape[:2]
        max_width = 1280
        if w > max_width:
            ratio = max_width / float(w)
            new_dimensions = (max_width, int(h * ratio))
            # INTER_AREA: Küçültme işlemlerinde piksellerin bozulmasını önleyen en ideal interpolasyondur
            img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)

        # 6. OPENCV İLE GÖRÜNTÜ İŞLEME
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Kullanıcının URL'den (Query) gönderdiği dinamik threshold değerleri uygulanıyor
        edges = cv2.Canny(gray_img, low_threshold, high_threshold)

        # 7. TRANSFER VE CANLI AKIŞ: İşlenen matrisi bellek şişirmeden PNG olarak fırlatma
        _, encoded_img = cv2.imencode(".png", edges)
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Görsel işlenirken sunucu içi bir mimari hata oluştu: {str(e)}"
        )
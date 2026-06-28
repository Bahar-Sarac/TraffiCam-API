# ==========================================
# 1. BÖLÜM: KÜTÜPHANELERİ ÇAĞIRMA
# ==========================================
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse  # İşlenen resmi geri dönmek için
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager  # Lifespan yönetimi için eklendi
from ultralytics import YOLO
import cv2
import numpy as np
import io

# ======================================================
# 7. BÖLÜM: LIFESPAN / AI MODEL BELLEK YÖNETİMİ
# ======================================================
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sunucu başlatıldığında model RAM'e bir kez yüklenir ve sıcak tutulur
    ml_models["yolo"] = YOLO("yolov8n.pt")
    print("Yapay Zeka Modeli (YOLOv8) RAM'e başarıyla yüklendi ve tetikte bekliyor!")
    yield
    # Sunucu kapatıldığında RAM temizlenir
    ml_models.clear()
    print("Sunucu kapandı, RAM temizlendi!")


# ==========================================
# 2. BÖLÜM: API VE GÜVENLİK AYARLARI
# ==========================================
app = FastAPI(
    title="TraffiCam-API",
    description="YOLOv8 ve OCR Tabanlı Akıllı Trafik İzleme Sistemi Backend Altyapısı",
    version="1.0.0",
    lifespan=lifespan  # EKSİK BURADAYDI: Lifespan mekanizması API'ye başarıyla bağlandı
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
def validate_and_decode_image(file: UploadFile) -> np.ndarray:
    """
    Hem OpenCV hem de YOLO katmanları için ortak doğrulama, boyut sınırlama ve
    görüntü kodu çözme işlemlerini yürüten merkezi merkezi koruma fonksiyonu.
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

    # 3. VERİ OKUMA: Ham byte yığınını okuma ve NumPy matrisine dönüştürme
    image_bytes = file.file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 4. BOZUK/KIRIK GÖRSEL KONTROLÜ
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Görsel kodu çözülemedi. Dosya bozuk veya okunamaz durumda."
        )

    return img


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
    try:
        # Ortak doğrulama mekanizması çağrılıyor
        img = validate_and_decode_image(file)

        # 5. PERFORMANS OPTİMİZASYONU: Downscaling
        # Devasa çözünürlüklerde (4K/2K) piksel yoğunluğunu azaltarak işlem hızını 3-4 kat artırıyoruz.
        h, w = img.shape[:2]
        max_width = 1280
        if w > max_width:
            ratio = max_width / float(w)
            new_dimensions = (max_width, int(h * ratio))
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


# ======================================================
# 8. BÖLÜM: YOLOv8 YAPAY ZEKA ENDPOINT'İ
# ======================================================
@app.post("/analyze-traffic/", status_code=status.HTTP_200_OK)
def analyze_traffic(file: UploadFile = File(...)):
    """
    Görseli RAM'deki sıcak YOLOv8 modeline besleyerek nesne tespiti (araç, yaya, tabela) yapan,
    araçların etrafına kutu çizip işlenmiş resmi dönen senkron (CPU-bound) endpoint.
    """
    try:
        # Ortak kalkan ve matris dönüşüm fonksiyonu çağrılıyor
        img = validate_and_decode_image(file)

        # RAM'de hazır bekleyen modeli çağır ve tahmini (inference) başlat
        model = ml_models["yolo"]
        results = model(img)

        # Modelin bulduğu tüm nesnelerin koordinatlarını ve güven skorlarını resmin üzerine çiz
        annotated_img = results[0].plot()

        # İşlenmiş yeni yapay zeka resmini PNG formatına encode et
        _, encoded_img = cv2.imencode(".png", annotated_img)

        # Canlı akış olarak istemciye fırlat
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Yapay zeka çıkarım pipeline hatası: {str(e)}"
        )
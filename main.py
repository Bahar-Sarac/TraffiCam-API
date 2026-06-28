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
async def process_image(file: UploadFile = File(...)):
    """
    Kameradan gelen görseli alan, doğrplayan, OpenCV piksel matrisine çevirip
    kenar tespiti (Canny Edge) yaptıktan sonra yeni resmi tarayıcıya dönen endpoint.
    """

    # 1. KORUMA KALKANI: Gelen dosya gerçekten bir resim mi?
    # Kullanıcı kazaen bir pdf veya txt dosyası yüklerse sistemi yormadan kapıda eliyoruz.
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Geçersiz dosya! Lütfen sadece JPG veya PNG formatında bir trafik görseli yükleyin."
        )

    try:
        # 2. VERİ OKUMA: İnternet kablosundan gelen ham binary (0 ve 1) veriyi okuyoruz
        image_bytes = await file.read()

        # 3. MATRİS SİHRİ:
        # Ham byte yığınını, bilgisayarın anlayacağı 1 boyutlu bir NumPy sayı dizisine çeviriyoruz
        nparr = np.frombuffer(image_bytes, np.uint8)

        # OpenCV devreye giriyor: "Bu sayı dizisini al ve renkli bir görsel matrisine (BGR) dönüştür"
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 4. OPENCV İLE GÖRÜNTÜ İŞLEME (Gürültü Temizliği):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Canny Edge algoritması ile görseldeki arabaların ve tabelaların geometrik sınır çizgilerini buluyoruz
        # 100 ve 200 değerleri eşik değerleridir; çizgilerin ne kadar keskin olacağını belirler
        edges = cv2.Canny(gray_img, 100, 200)

        # 5. DIŞ DÜNYAYA RESİM FIRLATMA:
        # Hafızadaki işlenmiş matrisi (edges) tekrar internetten taşınabilir PNG formatına paketliyoruz
        _, encoded_img = cv2.imencode(".png", edges)

        # Bytes verisini FastAPI'nin tarayıcıya "canlı resim" olarak basabilmesi için bir akışa sarıyoruz
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")

    except Exception as e:
        # Kodun içinde beklenmedik bir matematiksel hata olursa sunucunun çökmesini engelle diyoruz
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Görsel işlenirken sunucu içi hata oluştu: {str(e)}"
        )
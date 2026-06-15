"""
app.py
======
BASİT FLASK WEB ARAYÜZÜ

Akış:
-----
1. Kullanıcı ana sayfada bir görüntü yükler.
2. Sunucu görüntüyü kaydeder ve inference.py'deki AnomalyDetector ile skorlar.
3. Sonuç sayfasında gösterilir:
   - yüklenen orijinal görüntü
   - karar (Normal / Anomali)
   - anomali skoru (güven göstergesi)
   - ısı haritalı (heatmap) görüntü

Çalıştırma:
-----------
    python src/app.py
    -> tarayıcıda  http://127.0.0.1:5000  adresini aç.
"""

from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# inference.py içindeki dedektör sınıfını kullanıyoruz.
from inference import AnomalyDetector, DEFAULT_MODEL


# --- Klasör yolları ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULTS_DIR = STATIC_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

# Flask uygulaması. static/templates klasörleri proje kökünde olduğu için
# yollarını açıkça belirtiyoruz.
app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(PROJECT_ROOT / "templates"),
)
app.secret_key = "kalite-kontrol-demo"  # flash mesajları için (demo amaçlı)

# Dedektör nesnesini tembel (lazy) yüklüyoruz: model dosyası yoksa uygulama
# yine de açılır, kullanıcıya anlamlı bir uyarı gösteririz.
_detector: AnomalyDetector | None = None


def get_detector() -> AnomalyDetector:
    """Dedektörü ilk istekte bir kez yükler, sonra önbellekten döndürür."""
    global _detector
    if _detector is None:
        _detector = AnomalyDetector(model_path=DEFAULT_MODEL)
    return _detector


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Ana sayfa: görüntü yükleme formu."""
    return render_template("index.html", model_exists=Path(DEFAULT_MODEL).exists())


@app.route("/predict", methods=["POST"])
def predict():
    """Yüklenen görüntüyü işler ve sonuç sayfasını döndürür."""
    # 1) Dosya kontrolü
    if "image" not in request.files or request.files["image"].filename == "":
        flash("Lütfen bir görüntü dosyası seçin.")
        return redirect(url_for("index"))

    file = request.files["image"]
    if not allowed_file(file.filename):
        flash("Geçersiz dosya türü. PNG, JPG, JPEG veya BMP yükleyin.")
        return redirect(url_for("index"))

    # 2) Yüklenen dosyayı kaydet
    filename = secure_filename(file.filename)
    upload_path = UPLOAD_DIR / filename
    file.save(str(upload_path))

    # 3) Model ile tahmin yap
    try:
        detector = get_detector()
        result = detector.predict(upload_path, results_dir=RESULTS_DIR)
    except FileNotFoundError as exc:
        flash(str(exc))
        return redirect(url_for("index"))

    # 4) Şablona göndermek için web yollarını (static/...) hazırla
    uploaded_url = url_for("static", filename=f"uploads/{filename}")
    heatmap_name = Path(result["heatmap_path"]).name
    heatmap_url = url_for("static", filename=f"results/{heatmap_name}")

    return render_template(
        "result.html",
        label=result["label"],
        is_anomaly=result["is_anomaly"],
        score=result["score"],
        score_pct=round(result["score"] * 100, 1),
        uploaded_url=uploaded_url,
        heatmap_url=heatmap_url,
        filename=filename,
    )


if __name__ == "__main__":
    # debug=True: kod değişince otomatik yeniden başlar (geliştirme için).
    app.run(host="127.0.0.1", port=5000, debug=True)

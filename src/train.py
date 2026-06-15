"""
train.py
========
Üretim hattı görsel kalite kontrol sistemi - MODEL EĞİTİM SCRIPTI

Ne yapar?
---------
1. MVTec AD veri setinden seçilen kategoriyi (varsayılan: "bottle") yükler.
2. PatchCore anomali tespit modelini SADECE "normal" (kusursuz) görüntülerle
   eğitir. (PatchCore tek-sınıf / one-class bir yöntemdir: hatalı örnek görmeden
   öğrenir, test sırasında "normalden ne kadar uzak?" sorusuna bakar.)
3. Modeli test verisiyle değerlendirir ve metrikleri (AUROC vb.) ekrana basar.
4. Eğitilmiş modeli, Flask uygulamasının kullanabileceği tek bir ".pt" dosyası
   olarak  ../models/  klasörüne kaydeder (export eder).

Neden PatchCore?
----------------
- Önceden eğitilmiş bir CNN (wide_resnet50_2) kullanır; sıfırdan ağ eğitmez,
  bu yüzden CPU'da bile çalışabilir.
- Geri yayılım (backpropagation) yapmaz -> eğitim tek "epoch"tur ve hızlıdır.
- Hatalı bölgeyi gösteren ısı haritası (anomaly map) üretir -> görselleştirme kolay.

Çalıştırma:
-----------
    python src/train.py                      # varsayılan: bottle
    python src/train.py --category screw     # başka kategori
    python src/train.py --sampling-ratio 0.05  # daha hızlı/hafif (daha düşük doğruluk)
"""

import argparse
import shutil
from pathlib import Path

# --- anomalib bileşenleri ---
from anomalib.data import MVTecAD          # MVTec AD veri seti modülü
from anomalib.models import Patchcore       # PatchCore modeli
from anomalib.engine import Engine          # eğitim/test/export motoru
from anomalib.deploy import ExportType      # export formatı (TORCH, ONNX, OPENVINO)


# Proje kök dizini (bu dosya src/ içinde olduğu için bir üst klasör).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    """Komut satırı argümanlarını okur."""
    parser = argparse.ArgumentParser(description="PatchCore ile anomali tespit modeli eğit.")
    parser.add_argument(
        "--category",
        default="bottle",
        help="Eğitilecek MVTec AD kategorisi (örn. bottle, screw, hazelnut).",
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "data" / "MVTecAD"),
        help="MVTec AD veri setinin bulunduğu (veya indirileceği) klasör.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(PROJECT_ROOT / "models"),
        help="Eğitilmiş modelin kaydedileceği klasör.",
    )
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=0.1,
        help="Coreset örnekleme oranı. Düşürmek eğitimi hızlandırır ama doğruluğu düşürebilir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(f" PatchCore eğitimi başlıyor | kategori: {args.category}")
    print("=" * 60)

    # 1) VERİ MODÜLÜ ---------------------------------------------------------
    # root klasöründe veri yoksa anomalib otomatik indirmeye çalışır.
    # Küçük batch'ler CPU bellek kullanımını düşük tutar.
    datamodule = MVTecAD(
        root=args.data_root,
        category=args.category,
        train_batch_size=8,   # CPU için küçük tutuldu
        eval_batch_size=8,
        num_workers=0,        # Windows'ta 0 en güvenli ayardır
    )

    # 2) MODEL --------------------------------------------------------------
    # Varsayılan backbone "wide_resnet50_2"; layer2+layer3 özelliklerini kullanır.
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=args.sampling_ratio,
        num_neighbors=9,
    )

    # 3) MOTOR (Engine) -----------------------------------------------------
    # accelerator="auto": GPU varsa kullanır, yoksa otomatik CPU'ya düşer.
    # PatchCore tek epoch ile eğitilir (model bunu kendi içinde zorlar).
    engine = Engine(accelerator="auto")

    # 4) EĞİT ---------------------------------------------------------------
    # PatchCore "öğrenme" = normal görüntülerin özelliklerini bellek bankasına
    # toplamak ve coreset ile küçültmek demektir.
    print("\n[1/3] Model eğitiliyor (normal görüntülerden bellek bankası kuruluyor)...")
    engine.fit(datamodule=datamodule, model=model)

    # 5) TEST ---------------------------------------------------------------
    # Test seti hem normal hem hatalı görüntüler içerir; metrikler hesaplanır.
    print("\n[2/3] Model test ediliyor (AUROC vb. metrikler)...")
    test_results = engine.test(datamodule=datamodule, model=model)
    print("Test sonuçları:", test_results)

    # 6) EXPORT (kaydet) ----------------------------------------------------
    # Modeli, tek başına çalışabilen bir TorchScript-benzeri .pt dosyasına aktarır.
    # Bu .pt dosyası ön-işleme + son-işleme adımlarını da içerir, böylece
    # inference.py ve Flask uygulaması ek ayar yapmadan kullanabilir.
    print("\n[3/3] Model export ediliyor (.pt)...")
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    exported_path = engine.export(
        model=model,
        export_type=ExportType.TORCH,
        export_root=str(models_dir),
    )

    # anomalib export'u  models/weights/torch/model.pt  gibi bir yola yazar.
    # Uygulamanın kolay bulması için sabit bir isme kopyalıyoruz.
    final_model_path = models_dir / f"patchcore_{args.category}.pt"
    if exported_path is not None and Path(exported_path).exists():
        shutil.copy(str(exported_path), str(final_model_path))
        print(f"\n[OK] Model kaydedildi: {final_model_path}")
        print("   (inference.py ve app.py bu dosyayı kullanacak)")
    else:
        print("\n[UYARI] Export yolu bulunamadı. 'models/' klasörünü kontrol edin.")

    print("\nEğitim tamamlandı.")


if __name__ == "__main__":
    main()

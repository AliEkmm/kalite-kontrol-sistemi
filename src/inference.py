"""
inference.py
============
TEK GÖRÜNTÜ İÇİN TAHMİN + ISI HARİTASI (HEATMAP)

Ne yapar?
---------
1. Eğitilmiş PatchCore modelini (.pt) bir kez yükler.
2. Verilen bir görüntüyü modele sokar ve şunları üretir:
   - karar: "Normal" mı yoksa "Anomali" mi?
   - skor : 0-1 arası anomali skoru (yüksek = daha şüpheli)
   - anomaly_map: her piksel için "ne kadar anormal" değeri (ısı haritası)
3. Isı haritasını renklendirip orijinal görüntünün üzerine bindirir (overlay)
   ve sonucu bir PNG dosyası olarak kaydeder.

Hem komut satırından tek başına, hem de Flask uygulamasından (app.py)
içe aktarılarak (import) kullanılabilir.

Komut satırı kullanımı:
-----------------------
    python src/inference.py --image yol/gorsel.png
    python src/inference.py --image yol/gorsel.png --model models/patchcore_bottle.pt
"""

import argparse
import os
from pathlib import Path

# anomalib 2.x, .pt model dosyalarını pickle ile yüklemeden önce güvenlik
# nedeniyle onay ister. Burada eğittiğimiz kendi modelimizi yüklediğimiz için
# bu izni veriyoruz (TorchInferencer çağrılmadan önce ayarlanmalı).
os.environ.setdefault("TRUST_REMOTE_CODE", "1")

import cv2
import numpy as np

from anomalib.deploy import TorchInferencer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "patchcore_capsule.pt"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "static" / "results"


def _to_numpy(value):
    """torch.Tensor veya numpy gelebilir; her durumda numpy'a çevirir."""
    if hasattr(value, "detach"):          # torch.Tensor ise
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class AnomalyDetector:
    """
    PatchCore modelini bir kez yükleyip tekrar tekrar tahmin yapmayı sağlayan
    sınıf. Flask uygulaması bu sınıftan tek bir nesne oluşturur; böylece her
    istek için modeli yeniden yüklemek gerekmez (hız için önemli).
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, device: str = "cpu"):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model bulunamadı: {model_path}\n"
                "Önce 'python src/train.py' ile modeli eğitmeniz gerekir."
            )
        # TorchInferencer .pt dosyasını yükler; ön/son işleme adımları içindedir.
        self.inferencer = TorchInferencer(path=str(model_path), device=device)

    def predict(self, image_path: str | Path, results_dir: str | Path = DEFAULT_RESULTS_DIR) -> dict:
        """
        Tek bir görüntü için tahmin yapar ve ısı haritalı sonucu kaydeder.

        Dönüş (dict):
            label        : "Anomali" veya "Normal"
            is_anomaly   : bool
            score        : float (0-1 arası anomali skoru)
            heatmap_path : üretilen overlay görselinin dosya yolu
        """
        image_path = Path(image_path)
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # --- 1) Modelden tahmin al ---
        prediction = self.inferencer.predict(image=str(image_path))

        # pred_label True/1 -> anomali, False/0 -> normal.
        is_anomaly = bool(int(_to_numpy(prediction.pred_label).reshape(-1)[0]))
        score = float(_to_numpy(prediction.pred_score).reshape(-1)[0])

        # --- 2) Isı haritasını orijinal görüntüye bindir ---
        anomaly_map = _to_numpy(prediction.anomaly_map).squeeze()  # (H, W)
        heatmap_path = results_dir / f"result_{image_path.stem}.png"
        self._save_overlay(image_path, anomaly_map, heatmap_path)

        return {
            "label": "Anomali" if is_anomaly else "Normal",
            "is_anomaly": is_anomaly,
            "score": score,
            "heatmap_path": str(heatmap_path),
        }

    @staticmethod
    def _save_overlay(image_path: Path, anomaly_map: np.ndarray, out_path: Path) -> None:
        """Anomali haritasını renkli ısı haritası olarak orijinal resmin üzerine bindirir."""
        # Orijinal görüntüyü oku (BGR formatında gelir).
        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError(f"Görüntü okunamadı: {image_path}")
        h, w = original.shape[:2]

        # Anomali haritasını 0-1 aralığına normalize et (min-max).
        amap = anomaly_map.astype(np.float32)
        amin, amax = float(amap.min()), float(amap.max())
        if amax - amin > 1e-8:
            amap = (amap - amin) / (amax - amin)
        else:
            amap = np.zeros_like(amap)

        # Haritayı orijinal görüntü boyutuna büyüt ve 0-255'e ölçekle.
        amap = cv2.resize(amap, (w, h))
        amap_uint8 = (amap * 255).astype(np.uint8)

        # JET renk paleti: mavi=normal, kırmızı=anomali bölgesi.
        heatmap = cv2.applyColorMap(amap_uint8, cv2.COLORMAP_JET)

        # Orijinal görüntü + ısı haritası (ağırlıklı karışım).
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        cv2.imwrite(str(out_path), overlay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tek görüntü için anomali tahmini yap.")
    parser.add_argument("--image", required=True, help="Tahmin yapılacak görüntü yolu.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Eğitilmiş .pt model yolu.")
    args = parser.parse_args()

    detector = AnomalyDetector(model_path=args.model)
    result = detector.predict(args.image)

    print("=" * 50)
    print(f" Görüntü : {args.image}")
    print(f" Karar   : {result['label']}")
    print(f" Skor    : {result['score']:.4f}  (yüksek = daha şüpheli)")
    print(f" Heatmap : {result['heatmap_path']}")
    print("=" * 50)


if __name__ == "__main__":
    main()

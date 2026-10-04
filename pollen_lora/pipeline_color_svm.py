"""Pipeline B: descriptores de color hechos a mano + SVM.

Hipotesis que pone a prueba: si la clasificacion de polen corbicular depende
sobre todo del color (el supuesto detras de augment_data.py y
train_classifier.py), un modelo que SOLO mira estadisticas de color del pellet
deberia andar igual o mejor que una CNN cuando hay tan pocas fotos (~65
reales), porque tiene muchisimos menos parametros que ajustar y no puede
sobreajustarse a forma, textura o fondo.

Que hace:
  1. Separa el pellet del fondo negro que deja el recorte de remove.bg.
  2. Saca ~55 numeros por imagen: histogramas de tono/saturacion/brillo (HSV)
     y media, desvio y percentiles en Lab y RGB.
  3. Entrena un SVM (kernel RBF) sobre esos numeros.

Entrena en segundos en CPU y en la Raspberry solo necesita OpenCV + scikit-learn
(no NCNN): ver predict_bgr() mas abajo.

Uso:
    python pipeline_color_svm.py                       # entrena con train+valid, evalua en holdout, guarda el modelo
    python compare_pipelines.py --pipelines color_svm   # validacion cruzada (lo que conviene mirar para comparar)
"""
import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cv_common import load_samples, print_summary, summarize

FEATURE_SIZE = 128              # se reduce la imagen antes de sacar estadisticas: el color medio no necesita mas resolucion
BACKGROUND_THRESHOLD = 25       # un pixel con sus 3 canales por debajo de esto se toma como fondo negro (algo mas alto que en prepare_data.py por los artefactos JPG del borde)
MIN_PELLET_FRACTION = 0.05      # si queda menos que esto de pellet (ej. recorte de la Raspberry sin fondo negro) se usa la imagen entera
MIN_SATURATION_FOR_HUE = 30     # el tono de un pixel casi gris es ruido, no se cuenta en el histograma de tono
HUE_BINS, SV_BINS = 18, 8


def color_features(rgb: np.ndarray) -> np.ndarray:
    """Descriptor de color de una imagen RGB uint8 (H, W, 3)."""
    rgb = cv2.resize(rgb, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    mask = rgb.max(axis=2) >= BACKGROUND_THRESHOLD
    if mask.mean() < MIN_PELLET_FRACTION:
        mask[:] = True

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[mask].astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[mask].astype(np.float32)
    pix = rgb[mask].astype(np.float32)

    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    chromatic = s >= MIN_SATURATION_FOR_HUE
    hue_hist = np.histogram(h[chromatic], bins=HUE_BINS, range=(0, 180))[0].astype(np.float32)
    hue_hist /= max(hue_hist.sum(), 1.0)
    s_hist = np.histogram(s, bins=SV_BINS, range=(0, 256))[0] / len(s)
    v_hist = np.histogram(v, bins=SV_BINS, range=(0, 256))[0] / len(v)

    lab_stats = np.concatenate([
        lab.mean(0), lab.std(0), *np.percentile(lab, [10, 50, 90], axis=0),
    ])
    rgb_stats = np.concatenate([pix.mean(0), pix.std(0)])
    return np.concatenate([hue_hist, s_hist, v_hist, lab_stats, rgb_stats, [chromatic.mean()]]).astype(np.float32)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


class ColorSVMPipeline:
    name = "color_svm"

    def __init__(self, seed: int = 42, C: float = 10.0):
        self.seed, self.C = seed, C
        self._cache: dict[Path, np.ndarray] = {}

    def _features(self, paths) -> np.ndarray:
        for p in paths:
            if p not in self._cache:
                self._cache[p] = color_features(load_rgb(p))
        return np.stack([self._cache[p] for p in paths])

    def fit(self, paths, labels):
        self.model = make_pipeline(
            StandardScaler(),
            # Calibrado para que predict_proba() de una confianza usable en la Raspberry
            CalibratedClassifierCV(SVC(C=self.C, gamma="scale", class_weight="balanced"), ensemble=False),
        )
        self.model.fit(self._features(paths), labels)
        return self

    def predict(self, paths) -> np.ndarray:
        return self.model.predict(self._features(paths))

    def predict_proba(self, paths) -> np.ndarray:
        return self.model.predict_proba(self._features(paths))


def predict_bgr(bundle: dict, recorte_bgr: np.ndarray) -> tuple[str, float]:
    """Para usar en la Raspberry en lugar de clasificar_tipo_polen() de detector.py:

        bundle = joblib.load("color_svm.joblib")
        tipo, conf = predict_bgr(bundle, recorte_corbicula_bgr)
    """
    feats = color_features(cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB))[None]
    probs = bundle["model"].predict_proba(feats)[0]
    idx = int(probs.argmax())
    return bundle["classes"][idx], float(probs[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dirs", nargs="+", default=["data/train_instance", "data/valid"])
    ap.add_argument("--test_dir", default="data/holdout_test")
    ap.add_argument("--output_dir", default="models/color_svm")
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train, class_names = load_samples(args.train_dirs)
    test, test_classes = load_samples([args.test_dir])
    if test_classes != class_names:
        raise SystemExit("Las clases de train y test no coinciden entre si, revisa las carpetas.")

    print(f"=== Entrenando color_svm ({len(train)} imagenes, {sum(not s.is_aug for s in train)} reales) ===")
    pipe = ColorSVMPipeline(seed=args.seed, C=args.C).fit([s.path for s in train], [s.label for s in train])

    test = [s for s in test if not s.is_aug]
    summary = summarize([s.label for s in test], pipe.predict([s.path for s in test]), class_names)
    print_summary(f"color_svm (holdout, {len(test)} fotos)", summary, class_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipe.model, "classes": class_names}, output_dir / "color_svm.joblib")
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "holdout": summary}, f, indent=2, ensure_ascii=False)
    print(f"\nModelo guardado en {output_dir / 'color_svm.joblib'}")
    print("Ojo: el holdout tiene muy pocas fotos; para comparar pipelines usa compare_pipelines.py.")


if __name__ == "__main__":
    main()

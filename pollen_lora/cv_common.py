"""Utilidades compartidas para comparar pipelines de clasificacion con
validacion cruzada en vez de con el holdout fijo.

Por que: el holdout tiene 7 fotos (2 aromo, 2 cerezo, 3 yuyo). Con eso, UNA
foto mal clasificada mueve la accuracy 14 puntos -- no alcanza para decidir
que pipeline es mejor. Aca se juntan TODAS las fotos reales (train + valid +
holdout, 65 en total) y se hace k-fold, asi cada foto real se usa exactamente
una vez como test y la comparacion se apoya en 65 predicciones en vez de 7.

Dos cuidados para que el k-fold no infle las metricas:
  * Las copias de augment_data.py ("_augNN") viajan siempre con su foto
    original: si la original esta en el fold de test, sus copias NO se usan
    para entrenar ese fold. Y nunca se evalua sobre una copia aumentada.
  * Los recortes-hermanos (varios pellets de la misma foto, ver
    prepare_data.py) van todos al mismo fold (StratifiedGroupKFold).
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from augment_data import AUG_MARKER
from prepare_data import IMG_EXTS, source_key

DEFAULT_DIRS = ("data/train_instance", "data/valid", "data/holdout_test")


@dataclass
class Sample:
    path: Path
    label: int
    group: str
    is_aug: bool


def group_key(path: Path, class_name: str) -> str:
    """Foto de origen de una imagen: saca el indice que agrega prepare_data.py
    ("000123_") y el sufijo "_augNN" de augment_data.py, y despues aplica
    source_key() para juntar recortes-hermanos."""
    stem = re.sub(r"^\d{6}_", "", path.stem)
    stem = re.sub(rf"{AUG_MARKER}\d+$", "", stem)
    return f"{class_name}/{source_key(stem)}"


def load_samples(dirs=DEFAULT_DIRS) -> tuple[list[Sample], list[str]]:
    dirs = [Path(d) for d in dirs]
    class_names = sorted({p.name for d in dirs if d.exists() for p in d.iterdir() if p.is_dir()})
    if not class_names:
        raise SystemExit(f"No encontre carpetas de clase en {', '.join(map(str, dirs))}. Corre prepare_data.py primero.")

    samples = []
    for d in dirs:
        for label, cls in enumerate(class_names):
            cls_dir = d / cls
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    samples.append(Sample(p, label, group_key(p, cls), AUG_MARKER in p.stem))
    return samples, class_names


def make_folds(samples: list[Sample], n_splits: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Devuelve (indices_train, indices_test) por fold. El test son solo fotos
    originales; el train son las originales de los otros folds + sus copias
    aumentadas."""
    orig_idx = [i for i, s in enumerate(samples) if not s.is_aug]
    y = [samples[i].label for i in orig_idx]
    groups = [samples[i].group for i in orig_idx]

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for _, test_pos in splitter.split(orig_idx, y, groups):
        test_idx = [orig_idx[p] for p in test_pos]
        test_groups = {samples[i].group for i in test_idx}
        train_idx = [i for i, s in enumerate(samples) if s.group not in test_groups]
        folds.append((train_idx, test_idx))
    return folds


def split_train_val(samples: list[Sample], train_idx: list[int], n_splits: int, seed: int) -> tuple[list[int], list[int]]:
    """Separa una validacion (solo originales, ~1/n_splits de las fotos) dentro
    de los indices de entrenamiento de un fold, con el mismo cuidado de grupos
    y copias aumentadas que make_folds()."""
    sub = [samples[i] for i in train_idx]
    inner_train, val = make_folds(sub, n_splits, seed)[0]
    return [train_idx[i] for i in inner_train], [train_idx[i] for i in val]


def summarize(y_true, y_pred, class_names: list[str]) -> dict:
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return {"report": report, "confusion_matrix": cm.tolist()}


def print_summary(name: str, summary: dict, class_names: list[str]) -> None:
    report, cm = summary["report"], summary["confusion_matrix"]
    print(f"\n  --- {name}: precision / recall / f1-score por clase ---")
    print(f"  {'clase':20s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>8s}")
    for cls in [*class_names, "macro avg"]:
        m = report[cls]
        print(f"  {cls:20s} {m['precision']:10.3f} {m['recall']:10.3f} {m['f1-score']:10.3f} {m['support']:8.0f}")
    print(f"  accuracy: {report['accuracy']:.3f}")
    print("  matriz de confusion (filas = real, columnas = prediccion):")
    print("  " + " " * 20 + "".join(f"{c[:10]:>12s}" for c in class_names))
    for cls, row in zip(class_names, np.asarray(cm)):
        print(f"  {cls:20s}" + "".join(f"{v:12d}" for v in row))

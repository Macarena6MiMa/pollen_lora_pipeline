"""Compara pipelines de clasificacion bajo un protocolo identico para todos:

  resnet18_finetune         fine-tuning de ResNet18 (train_classifier.py)
  efficientnet_b0_finetune  fine-tuning de EfficientNet-B0 (train_classifier.py --model efficientnet_b0)
  color_svm                 descriptores de color + SVM (pipeline_color_svm.py)
  dinov2_linear             DINOv2 congelado + regresion logistica (pipeline_dinov2_linear.py)

Protocolo (lo unico que cambia entre pipelines es el pipeline):
  * Misma semilla, mismos datos (train_instance con sus copias _aug, ya
    corregidos por fuga) y mismos folds/splits para todos.
  * Las CNN usan EXACTAMENTE train_one() de train_classifier.py: mismas
    transformaciones (sin hue), AdamW, --epochs epocas (100 por default) y
    checkpoint en la mejor accuracy de validacion.
  * Mismo presupuesto de ajuste para todos: 3 valores del hiperparametro
    principal, elegido con la validacion (nunca con test). En las CNN es el
    lr {1e-4, 3e-4, 1e-3}, asi ninguna red pierde por estar mal configurada.
    Empates de accuracy de validacion se desempatan con la menor log-loss.

Dos protocolos de evaluacion:
  --protocol kfold  (default, recomendado) k-fold agrupado por foto de origen
                    sobre las 65 fotos reales (train + valid + holdout). Dentro
                    de cada fold se separa una validacion, tambien agrupada.
                    Cada foto real se usa exactamente una vez como test.
  --protocol seeds  splits fijos (train_instance / valid / holdout_test),
                    repetido con 5 semillas; reporta media +- desvio. Ojo: el
                    test sigue siendo de 7 fotos (1 foto = ~14 puntos).

Uso:
    python compare_pipelines.py
    python compare_pipelines.py --pipelines resnet18_finetune efficientnet_b0_finetune
    python compare_pipelines.py --protocol seeds
"""
import argparse
import contextlib
import copy
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score, log_loss
from torch.utils.data import DataLoader, Dataset

from cv_common import load_samples, make_folds, print_summary, split_train_val, summarize
from train_classifier import build_model, build_transforms, train_one

ALL_PIPELINES = ["resnet18_finetune", "efficientnet_b0_finetune", "color_svm", "dinov2_linear"]
LR_BUDGET = [1e-4, 3e-4, 1e-3]
C_BUDGET = {"color_svm": [1.0, 10.0, 100.0], "dinov2_linear": [0.1, 0.5, 2.0]}


class PathDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


class CNNFinetune:
    """Adaptador de train_classifier.py: la unica diferencia entre las dos CNN
    es `arch`. Se resiembra despues de crear el modelo y el orden de los
    batches sale de un generador propio, asi ambas redes ven las mismas
    imagenes en el mismo orden."""
    hp_name = "lr"
    budget = LR_BUDGET

    def __init__(self, arch, num_classes, epochs, batch_size, seed):
        self.arch, self.num_classes, self.epochs, self.batch_size, self.seed = arch, num_classes, epochs, batch_size, seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, train_p, train_y, val_p, val_y, lr):
        model = build_model(self.arch, self.num_classes).to(self.device)
        torch.manual_seed(self.seed)
        train_loader = DataLoader(PathDataset(train_p, train_y, build_transforms(train=True)),
                                  batch_size=self.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(self.seed))
        val_loader = DataLoader(PathDataset(val_p, val_y, build_transforms(train=False)), batch_size=self.batch_size)
        with contextlib.redirect_stdout(io.StringIO()):  # train_one imprime cada epoca
            self.model, _ = train_one(model, train_loader, val_loader, self.device, self.epochs, lr)
        return self

    @torch.no_grad()
    def predict_proba(self, paths):
        self.model.eval()
        loader = DataLoader(PathDataset(paths, [0] * len(paths), build_transforms(train=False)),
                            batch_size=self.batch_size)
        return np.concatenate([self.model(x.to(self.device)).softmax(1).cpu().numpy() for x, _ in loader])

    def predict(self, paths):
        return self.predict_proba(paths).argmax(1)


class SklearnFit:
    """Adaptador de pipeline_color_svm / pipeline_dinov2_linear: el
    hiperparametro es C. El extractor de descriptores se crea una sola vez y
    cachea, asi probar 3 valores de C no cuesta 3 veces."""
    hp_name = "C"

    def __init__(self, pipe, budget):
        self.pipe, self.budget = pipe, budget

    def fit(self, train_p, train_y, val_p, val_y, C):
        self.pipe.C = C
        self.pipe.fit(train_p, train_y)
        self.model = self.pipe.model
        return self

    def predict_proba(self, paths):
        self.pipe.model = self.model
        return self.pipe.predict_proba(paths)

    def predict(self, paths):
        return self.predict_proba(paths).argmax(1)


def build_pipeline(name, num_classes, args):
    if name == "resnet18_finetune":
        return CNNFinetune("resnet18", num_classes, args.epochs, args.batch_size, args.seed)
    if name == "efficientnet_b0_finetune":
        return CNNFinetune("efficientnet_b0", num_classes, args.epochs, args.batch_size, args.seed)
    if name == "color_svm":
        from pipeline_color_svm import ColorSVMPipeline
        return SklearnFit(ColorSVMPipeline(seed=args.seed), C_BUDGET[name])
    if name == "dinov2_linear":
        from pipeline_dinov2_linear import DinoV2LinearPipeline
        return SklearnFit(DinoV2LinearPipeline(seed=args.seed), C_BUDGET[name])
    raise ValueError(name)


def fit_with_budget(pipe, samples, train_idx, val_idx, n_classes):
    """Entrena con cada valor del presupuesto y se queda con el mejor en
    validacion: mayor accuracy, y a igual accuracy menor log-loss."""
    paths = lambda idx: [samples[i].path for i in idx]
    labels = lambda idx: [samples[i].label for i in idx]
    best, tried = None, []
    for hp in pipe.budget:
        pipe.fit(paths(train_idx), labels(train_idx), paths(val_idx), labels(val_idx), hp)
        proba = pipe.predict_proba(paths(val_idx))
        val_y = labels(val_idx)
        acc = float(np.mean(proba.argmax(1) == np.array(val_y)))
        loss = float(log_loss(val_y, proba, labels=list(range(n_classes))))
        tried.append({pipe.hp_name: hp, "val_acc": acc, "val_loss": loss})
        if best is None or (acc, -loss) > (best[0], -best[1]):
            best = (acc, loss, hp, copy.deepcopy(pipe.model))
    pipe.model = best[3]
    return best[2], tried


def make_splits(args, samples_by_dir):
    """Lista de (etiqueta, samples, train_idx, val_idx, test_idx) segun el protocolo."""
    if args.protocol == "kfold":
        samples = [s for group in samples_by_dir for s in group]
        splits = []
        for rep in range(args.repeats):
            for k, (train_idx, test_idx) in enumerate(make_folds(samples, args.folds, args.seed + rep)):
                inner_train, val = split_train_val(samples, train_idx, args.inner_splits, args.seed + rep)
                splits.append((f"rep {rep + 1} fold {k + 1}/{args.folds}", samples, inner_train, val, test_idx))
        return splits

    train, valid, test = samples_by_dir
    samples = train + valid + test
    train_idx = list(range(len(train)))
    val_idx = [len(train) + i for i, s in enumerate(valid) if not s.is_aug]
    test_idx = [len(train) + len(valid) + i for i, s in enumerate(test) if not s.is_aug]
    return [(f"semilla {args.seed + r}", samples, train_idx, val_idx, test_idx) for r in range(args.n_seeds)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="data/train_instance")
    ap.add_argument("--valid_dir", default="data/valid")
    ap.add_argument("--test_dir", default="data/holdout_test")
    ap.add_argument("--pipelines", nargs="+", default=ALL_PIPELINES, choices=ALL_PIPELINES)
    ap.add_argument("--protocol", default="kfold", choices=["kfold", "seeds"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner_splits", type=int, default=4, help="la validacion interna es ~1/N de las fotos de entrenamiento del fold")
    ap.add_argument("--repeats", type=int, default=1, help="(kfold) repetir el k-fold con distintas semillas")
    ap.add_argument("--n_seeds", type=int, default=5, help="(seeds) cantidad de semillas")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--output", default="models/comparison/results.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples_by_dir, class_names = [], None
    for d in (args.train_dir, args.valid_dir, args.test_dir):
        group, classes = load_samples([d])
        if class_names not in (None, classes):
            raise SystemExit("Las clases de train/valid/test no coinciden entre si, revisa las carpetas.")
        samples_by_dir.append(group)
        class_names = classes
    n_classes = len(class_names)

    splits = make_splits(args, samples_by_dir)
    _, samples, tr, va, te = splits[0]
    print(f"Protocolo '{args.protocol}': {len(splits)} corridas. Primera: {len(tr)} train "
          f"({sum(not samples[i].is_aug for i in tr)} reales), {len(va)} validacion, {len(te)} test. "
          f"Clases: {class_names}")
    if not torch.cuda.is_available() and any(p.endswith("_finetune") for p in args.pipelines):
        n_runs = len(splits) * len(LR_BUDGET) * sum(p.endswith("_finetune") for p in args.pipelines)
        print(f"[!] Sin GPU: las CNN hacen {n_runs} entrenamientos de {args.epochs} epocas en CPU. "
              "Puede tardar muchas horas; en un Colab con GPU es bastante mas rapido.")

    results = {}
    for name in args.pipelines:
        print(f"\n=== {name} ===")
        t0 = time.time()
        torch.manual_seed(args.seed)
        pipe = build_pipeline(name, n_classes, args)
        if args.protocol == "seeds" and not name.endswith("_finetune"):
            print("  (pipeline deterministico: las semillas solo cambian las CNN, se corre una vez)")
            run_splits = splits[:1]
        else:
            run_splits = splits

        y_true, y_pred, runs = [], [], []
        for label, samples, train_idx, val_idx, test_idx in run_splits:
            if isinstance(pipe, CNNFinetune):
                pipe.seed = args.seed + len(runs) if args.protocol == "seeds" else args.seed
            chosen, tried = fit_with_budget(pipe, samples, train_idx, val_idx, n_classes)
            pred = pipe.predict([samples[i].path for i in test_idx])
            true = [samples[i].label for i in test_idx]
            y_true.extend(true)
            y_pred.extend(pred.tolist())
            acc = float(np.mean(np.array(true) == pred))
            f1 = float(f1_score(true, pred, average="macro", labels=list(range(n_classes)), zero_division=0))
            runs.append({"run": label, f"chosen_{pipe.hp_name}": chosen, "tried": tried, "test_acc": acc, "test_macro_f1": f1})
            print(f"  {label}: {pipe.hp_name}={chosen:g} (val_acc={max(t['val_acc'] for t in tried):.3f})  "
                  f"test acc={acc:.3f}  macro_f1={f1:.3f}  [{time.time() - t0:.0f}s]")

        summary = summarize(y_true, y_pred, class_names)
        summary.update({"runs": runs, "seconds": time.time() - t0})
        results[name] = summary
        print_summary(f"{name} ({len(y_true)} predicciones de test acumuladas)", summary, class_names)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "protocol": vars(args), "results": results}, f, indent=2, ensure_ascii=False)

    print(f"\n=== Comparacion final (protocolo '{args.protocol}', solo fotos reales en test) ===")
    print(f"{'pipeline':26s} {'acc media +- desvio':>20s} {'macro-F1 media +- desvio':>25s} {'acc acumulada':>14s} {'tiempo':>8s}")
    stats = {}
    for name, r in results.items():
        accs = np.array([x["test_acc"] for x in r["runs"]])
        f1s = np.array([x["test_macro_f1"] for x in r["runs"]])
        stats[name] = (f1s.mean(), f1s.std())
        print(f"{name:26s} {accs.mean():11.3f} +- {accs.std():.3f} {f1s.mean():16.3f} +- {f1s.std():.3f} "
              f"{r['report']['accuracy']:14.3f} {r['seconds']:7.0f}s")

    ranked = sorted(stats, key=lambda n: stats[n][0], reverse=True)
    if len(ranked) > 1:
        best, second = ranked[0], ranked[1]
        gap = stats[best][0] - stats[second][0]
        noise = max(stats[best][1], stats[second][1])
        if gap < noise:
            print(f"\n[!] {best} le gana a {second} por {gap:.3f} de macro-F1 medio, menos que el desvio entre "
                  f"corridas ({noise:.3f}): con estos datos no se puede afirmar que uno sea mejor.")
        else:
            print(f"\nMejor: {best} (macro-F1 medio {gap:+.3f} sobre {second}).")
    print(f"\nResultados completos (hiperparametro elegido y validacion de cada corrida) en {output}")


if __name__ == "__main__":
    main()

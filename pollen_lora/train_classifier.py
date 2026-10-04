"""Entrena un clasificador de polen (transfer learning con una CNN preentrenada)
y compara "real_only" vs "real_plus_synthetic", evaluando ambos sobre el holdout
de test que nunca vio ni el LoRA ni el clasificador.

Corre esto DESPUES de generate_synthetic.py (aunque tambien funciona sin
sinteticas, solo entrena el caso "real_only" y avisa que falta la comparacion).

Uso:
    python train_classifier.py
    python train_classifier.py --epochs 20 --model resnet18
"""
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool, size: int = 224) -> transforms.Compose:
    ops = []
    if train:
        # Leve y sin hue-shift, igual criterio que augment_data.py: la clasificacion
        # de polen corbicular se basa sobre todo en color.
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.08, hue=0.0),
        ]
    ops += [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"modelo no soportado: {name}")
    return model


def evaluate_acc(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0


def train_one(model, train_loader, valid_loader, device, epochs, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_state, best_val_acc = None, -1.0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_acc = evaluate_acc(model, valid_loader, device)
        print(f"    epoch {epoch + 1:3d}/{epochs}  val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc


def evaluate_full(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            all_preds.extend(pred.tolist())
            all_labels.extend(y.tolist())
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    return report, cm


def print_report(name: str, report: dict, cm, class_names: list[str]) -> None:
    print(f"\n  --- {name}: precision / recall / f1-score por clase (holdout test) ---")
    print(f"  {'clase':20s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>8s}")
    for cls in class_names:
        m = report[cls]
        print(f"  {cls:20s} {m['precision']:10.3f} {m['recall']:10.3f} {m['f1-score']:10.3f} {m['support']:8.0f}")
    macro, weighted = report["macro avg"], report["weighted avg"]
    print(f"  {'macro avg':20s} {macro['precision']:10.3f} {macro['recall']:10.3f} {macro['f1-score']:10.3f} {macro['support']:8.0f}")
    print(f"  {'weighted avg':20s} {weighted['precision']:10.3f} {weighted['recall']:10.3f} {weighted['f1-score']:10.3f} {weighted['support']:8.0f}")

    print(f"  matriz de confusion (filas = real, columnas = prediccion):")
    header = "  " + " " * 20 + "".join(f"{c[:10]:>12s}" for c in class_names)
    print(header)
    for cls, row in zip(class_names, cm):
        print(f"  {cls:20s}" + "".join(f"{v:12d}" for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="data/train_instance")
    ap.add_argument("--synthetic_dir", default="data/synthetic")
    ap.add_argument("--valid_dir", default="data/valid")
    ap.add_argument("--test_dir", default="data/holdout_test")
    ap.add_argument("--output_dir", default="models/classifier")
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[!] No se detecto GPU. El clasificador entrena bastante mas rapido que el LoRA "
              "igual va a andar en CPU, pero contá con mas tiempo que en GPU.")

    real_dir, synth_dir = Path(args.real_dir), Path(args.synthetic_dir)
    valid_dir, test_dir = Path(args.valid_dir), Path(args.test_dir)
    for d in (real_dir, valid_dir, test_dir):
        if not d.exists():
            raise SystemExit(f"No existe {d}. Corre prepare_data.py / augment_data.py primero.")

    train_tf, eval_tf = build_transforms(train=True), build_transforms(train=False)

    real_train_ds = datasets.ImageFolder(real_dir, transform=train_tf)
    class_names = real_train_ds.classes
    valid_ds = datasets.ImageFolder(valid_dir, transform=eval_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    if valid_ds.classes != class_names or test_ds.classes != class_names:
        raise SystemExit("Las clases de train/valid/test no coinciden entre si, revisa las carpetas.")

    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    configs = {"real_only": [real_train_ds]}
    if synth_dir.exists() and any(synth_dir.rglob("*.*")):
        synth_ds = datasets.ImageFolder(synth_dir, transform=train_tf)
        if synth_ds.classes != class_names:
            raise SystemExit(f"Las clases de {synth_dir} no coinciden con las de {real_dir}.")
        configs["real_plus_synthetic"] = [real_train_ds, synth_ds]
    else:
        print(f"[!] No encontre imagenes en {synth_dir}. Corre generate_synthetic.py para poder "
              "comparar contra 'real_plus_synthetic'; por ahora solo entreno 'real_only'.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, ds_list in configs.items():
        combined = ConcatDataset(ds_list) if len(ds_list) > 1 else ds_list[0]
        print(f"\n=== Entrenando clasificador '{name}' ({len(combined)} imagenes) ===")
        train_loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True)

        model = build_model(args.model, len(class_names)).to(device)
        model, best_val_acc = train_one(model, train_loader, valid_loader, device, args.epochs, args.lr)

        test_report, test_cm = evaluate_full(model, test_loader, device, class_names)
        results[name] = {
            "best_val_acc": best_val_acc,
            "test_accuracy": test_report["accuracy"],
            "test_report": test_report,
            "confusion_matrix": test_cm.tolist(),
        }

        torch.save(model.state_dict(), output_dir / f"{name}.pt")
        print(f"  val_acc={best_val_acc:.3f}  test_acc={test_report['accuracy']:.3f}  "
              f"-> guardado en {output_dir / f'{name}.pt'}")
        print_report(name, test_report, test_cm, class_names)

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "results": results}, f, indent=2, ensure_ascii=False)

    print("\n=== Comparacion final (holdout test, nunca visto por LoRA ni clasificador) ===")
    print(f"{'config':25s} {'val_acc':>10s} {'test_acc':>10s} {'macro_f1':>10s} {'macro_rec':>10s}")
    for name, r in results.items():
        macro = r["test_report"]["macro avg"]
        print(f"{name:25s} {r['best_val_acc']:10.3f} {r['test_accuracy']:10.3f} "
              f"{macro['f1-score']:10.3f} {macro['recall']:10.3f}")

    if "real_plus_synthetic" in results:
        acc_delta = results["real_plus_synthetic"]["test_accuracy"] - results["real_only"]["test_accuracy"]
        f1_delta = (results["real_plus_synthetic"]["test_report"]["macro avg"]["f1-score"]
                    - results["real_only"]["test_report"]["macro avg"]["f1-score"])
        if acc_delta > 0 and f1_delta > 0:
            print(f"\nEl aumento con sinteticas mejoro accuracy ({acc_delta:+.3f}) y macro-F1 ({f1_delta:+.3f}) en test.")
        else:
            print(f"\n[!] El aumento con sinteticas NO mejoro de forma consistente en test "
                  f"(accuracy {acc_delta:+.3f}, macro-F1 {f1_delta:+.3f}). "
                  "No asumas que ayuda solo porque las imagenes generadas se ven bien -- revisa "
                  "data/synthetic/ a ojo, mira que clase esta bajando en el detalle por clase de arriba, "
                  "y considera reentrenar los LoRA antes de sumarlas.")

    print(f"\nResultados completos (por clase, matriz de confusion) en {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

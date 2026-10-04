"""Verifica que el clasificador exportado a NCNN prediga lo mismo que el
modelo PyTorch original, corriendo las mismas imagenes de test por ambos.

Corresponde a la Fase 6b de tu metodologia ("Evaluacion del despliegue"):
si las predicciones no coinciden, las metricas reportadas para el modelo
PyTorch no describen al modelo que realmente corre en la Raspberry Pi.

Requiere: pip install ncnn

Uso:
    python verify_classifier_ncnn.py
    python verify_classifier_ncnn.py --ncnn_param models/classifier/real_only.torchscript.ncnn.param
"""
import argparse
from pathlib import Path

import ncnn
import numpy as np
import torch
from torch import nn
from torchvision import datasets, models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"modelo no soportado: {name}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/classifier/real_only.pt")
    ap.add_argument("--ncnn_param", default="models/classifier/real_only.torchscript.ncnn.param")
    ap.add_argument("--ncnn_bin", default="models/classifier/real_only.torchscript.ncnn.bin")
    ap.add_argument("--test_dir", default="data/holdout_test")
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    for p in (args.weights, args.ncnn_param, args.ncnn_bin, args.test_dir):
        if not Path(p).exists():
            raise SystemExit(f"No existe {p}.")

    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_ds = datasets.ImageFolder(args.test_dir, transform=transform)
    class_names = test_ds.classes

    model = build_model(args.model, len(class_names))
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    net = ncnn.Net()
    net.load_param(args.ncnn_param)
    net.load_model(args.ncnn_bin)

    print(f"{'imagen':30s} {'real':10s} {'PyTorch':10s} {'NCNN':10s} {'max|diff|':>10s} {'coincide':>10s}")
    n_match = 0
    max_diff_overall = 0.0

    for i in range(len(test_ds)):
        img_tensor, label = test_ds[i]
        img_path = Path(test_ds.samples[i][0]).name

        with torch.no_grad():
            logits_pt = model(img_tensor.unsqueeze(0))[0].numpy()

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(img_tensor.numpy()).clone())
            _, out0 = ex.extract("out0")
            logits_ncnn = np.array(out0)

        pred_pt = int(logits_pt.argmax())
        pred_ncnn = int(logits_ncnn.argmax())
        max_diff = float(np.abs(logits_pt - logits_ncnn).max())
        max_diff_overall = max(max_diff_overall, max_diff)
        match = pred_pt == pred_ncnn
        n_match += match

        print(f"{img_path:30s} {class_names[label]:10s} {class_names[pred_pt]:10s} "
              f"{class_names[pred_ncnn]:10s} {max_diff:10.6f} {'si' if match else 'NO':>10s}")

    print(f"\n{n_match}/{len(test_ds)} predicciones coinciden entre PyTorch y NCNN "
          f"(diferencia maxima de logits observada: {max_diff_overall:.6f}).")
    if n_match < len(test_ds):
        print("[!] Hay discrepancias -- revisa el preprocesamiento (resize/normalize) "
              "o el paso de exportacion antes de confiar en las metricas reportadas "
              "para el modelo desplegado en la Raspberry Pi.")


if __name__ == "__main__":
    main()

"""Exporta el clasificador CNN (ResNet18) entrenado por train_classifier.py
al formato NCNN, para desplegarlo en la Raspberry Pi 4 junto a los dos
modelos YOLO del pipeline en cascada.

A diferencia de los modelos YOLO (que Ultralytics exporta directo con
`model.export(format="ncnn")`), un modelo de torchvision arbitrario como
ResNet18 no tiene ese atajo. El camino soportado por el propio proyecto
ncnn es: PyTorch -> TorchScript (torch.jit.trace) -> pnnx -> NCNN
(.param + .bin).

Instalar antes:
    pip install pnnx

Uso:
    python export_classifier_ncnn.py --weights models/classifier/real_only.pt
"""
import argparse
import shutil
import site
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import models

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


def find_pnnx() -> str:
    pnnx_exe = shutil.which("pnnx")
    if pnnx_exe is not None:
        return pnnx_exe

    # El ejecutable "pnnx" que instala pip puede quedar junto al interprete
    # (instalacion normal) o en la carpeta Scripts de usuario (pip install --user
    # en Windows: %APPDATA%\Python\PythonXY\Scripts), que no siempre esta en el PATH.
    candidates = [
        Path(sys.executable).parent / "Scripts" / "pnnx.exe",
        Path(sys.executable).parent / "pnnx",
        Path(site.getusersitepackages()).parent / "Scripts" / "pnnx.exe",
        Path(site.getuserbase()) / "Scripts" / "pnnx.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise SystemExit(
        "No encuentro el ejecutable 'pnnx'. Corre 'pip install pnnx' y despues "
        "'pnnx --help' en una terminal para confirmar que quedo en el PATH; si no, "
        "pasame la ruta donde pip lo instalo (normalmente junto a pip.exe) y ajusto "
        "el script para apuntar ahi directamente."
    )


def export_to_ncnn(model: nn.Module, torchscript_path: Path, size: int) -> tuple[Path, Path]:
    """PyTorch -> TorchScript -> pnnx -> NCNN. Devuelve (.param, .bin)."""
    example = torch.rand(1, 3, size, size)
    traced = torch.jit.trace(model, example)
    traced.save(str(torchscript_path))
    print(f"TorchScript guardado en {torchscript_path}")

    cmd = [find_pnnx(), str(torchscript_path), f"inputshape=[1,3,{size},{size}]"]
    print(f"Corriendo: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit("pnnx fallo, revisa el log de arriba.")

    return torchscript_path.with_suffix(".ncnn.param"), torchscript_path.with_suffix(".ncnn.bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/classifier/real_only.pt",
                     help="checkpoint guardado por train_classifier.py (state_dict)")
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"No encuentro {weights_path}. Corre train_classifier.py primero.")

    model = build_model(args.model, args.num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    ncnn_param, ncnn_bin = export_to_ncnn(model, weights_path.with_suffix(".torchscript.pt"), args.size)
    print(f"\nListo. Copia estos dos archivos a la Raspberry Pi:")
    print(f"  {ncnn_param}")
    print(f"  {ncnn_bin}")
    print("\nRecorda preprocesar igual que en el entrenamiento antes de pasarle la "
          f"imagen al modelo NCNN: redimensionar a {args.size}x{args.size} y normalizar "
          f"con mean={IMAGENET_MEAN} std={IMAGENET_STD} (en escala 0-255, no 0-1, si usas "
          "ncnn.Mat.substract_mean_normalize con esos valores multiplicados por 255).")


if __name__ == "__main__":
    main()

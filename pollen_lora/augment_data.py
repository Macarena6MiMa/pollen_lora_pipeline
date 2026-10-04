"""
Genera copias aumentadas (flip, rotacion, zoom leve, brillo/contraste MUY leves) de
las fotos reales de entrenamiento, para darle mas variedad visual al fine-tuning del LoRA.

Correr esto DESPUES de prepare_data.py y ANTES de run_training.py:

    python prepare_data.py
    python augment_data.py --copies 8
    python run_training.py

Solo toca data/train_instance/ -- nunca toca data/holdout_test/. Las copias quedan
en la misma carpeta que las reales, asi que run_training.py las usa automaticamente
sin que haya que tocar nada mas.

Ojo con el color: la identificacion de polen corbicular se basa sobre todo en color,
asi que el jitter de color/brillo/contraste se deja deliberadamente leve y sin
hue-shift -- no queremos que una especie termine "viendose" como otra. Los flips,
la rotacion y el zoom leve no tienen ese problema (la forma/orientacion del pellet
no es la senal de clasificacion), asi que ahi se puede ser mas generoso.

Uso:
    python augment_data.py --copies 8
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
AUG_MARKER = "_aug"  # asi identificamos (y podemos limpiar) copias generadas en una corrida anterior


def detect_background(img: Image.Image) -> tuple[int, int, int]:
    """Promedia las 4 esquinas para adivinar el color de fondo real de la foto.

    Las fotos vienen de un recorte con remove.bg exportado a JPG (sin canal
    alpha), asi que el fondo "transparente" termina siendo un color solido
    -- casi siempre negro, no blanco. Si el relleno de la rotacion no
    coincide, cada copia aumentada queda con parches negros y blancos mezclados.
    """
    w, h = img.size
    corners = [
        img.getpixel((0, 0)),
        img.getpixel((w - 1, 0)),
        img.getpixel((0, h - 1)),
        img.getpixel((w - 1, h - 1)),
    ]
    return tuple(sum(c[ch] for c in corners) // 4 for ch in range(3))


def build_transform(size: int, background: tuple[int, int, int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),          # el pellet no tiene una orientacion "correcta"
        transforms.RandomRotation(degrees=180, fill=background),
        transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        # Deliberadamente leve y sin hue: ver el comentario del encabezado.
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.08, hue=0.0),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="data/train_instance")
    ap.add_argument("--copies", type=int, default=8, help="copias aumentadas por cada foto real")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--background", type=int, default=None,
                     help="gris de relleno fijo (0-255) para las esquinas que deja la rotacion; "
                          "por default se detecta automaticamente el color de fondo de cada foto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    train_dir = Path(args.train_dir)
    if not train_dir.exists():
        raise SystemExit(f"No existe {train_dir}. Corre prepare_data.py primero.")

    fixed_background = None if args.background is None else (args.background,) * 3
    classes = sorted(p for p in train_dir.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"{train_dir} no tiene subcarpetas. Corre prepare_data.py primero.")

    print(f"{'clase':30s} {'reales':>7s} {'+ aumentadas':>13s}")
    for cls_dir in classes:
        # Limpia aumentaciones de una corrida anterior para que sea seguro re-ejecutar el script
        for old in cls_dir.glob(f"*{AUG_MARKER}*"):
            old.unlink()

        originals = sorted(
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS and AUG_MARKER not in p.stem
        )
        if not originals:
            print(f"  [!] {cls_dir.name}: no encontre fotos reales, la salteo.")
            continue

        made = 0
        for img_path in originals:
            img = Image.open(img_path).convert("RGB")
            background = fixed_background or detect_background(img)
            transform = build_transform(args.size, background)
            for i in range(args.copies):
                aug_img = transform(img)
                out_name = f"{img_path.stem}{AUG_MARKER}{i:02d}{img_path.suffix}"
                aug_img.save(cls_dir / out_name)
                made += 1

        print(f"{cls_dir.name:30s} {len(originals):7d} {made:13d}")

    print("\nListo. Las copias quedaron en data/train_instance/<clase>/, junto a las reales.")
    print("run_training.py las va a levantar automaticamente, no hace falta tocar nada mas.")


if __name__ == "__main__":
    main()

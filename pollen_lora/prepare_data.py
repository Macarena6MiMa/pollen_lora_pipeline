"""Prepara un dataset que ya viene separado en train, valid y test.

La clase se toma del nombre de la carpeta que contiene cada imagen. Los nombres
de los archivos no se usan para determinar la clase, pero SI se usan para
detectar fugas entre splits (ver abajo).

Descarta automaticamente fotos casi enteramente negras (recortes de remove.bg
donde no quedo pellet visible, o exportaciones corruptas) para que no contaminen
el entrenamiento. Se imprime cada foto descartada para que la revises a ojo.

Deteccion de fuga entre splits: algunas fotos (sobre todo las de WhatsApp con
sufijo "__1_", "__2_", etc.) tienen VARIOS pellets recortados de la MISMA foto
original. Si Roboflow reparte esos recortes hermanos entre train/valid/test,
comparten fondo, luz y ruido de camara -- el clasificador puede aprender a
reconocer la foto en vez del pellet, e inflar artificialmente las metricas de
validacion/test. Para evitarlo, se agrupan los recortes por foto de origen y,
si un grupo quedo repartido entre splits, TODOS sus recortes se mueven a train
(nunca se deja un recorte "hermano" de uno de train en valid o test).

Uso:
    python prepare_data.py
"""
import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Un pixel se cuenta como "negro" si sus 3 canales estan por debajo de esto.
BLACK_PIXEL_THRESHOLD = 15
# Si una foto tiene mas de esta fraccion de pixels negros, se descarta por no
# tener pellet visible (o casi).
MAX_BLACK_FRACTION = 0.97

SPLITS = {
    "train": "train_instance",
    "valid": "valid",
    "test": "holdout_test",
}
# Si un grupo de recortes-hermanos queda repartido entre splits, se reasignan
# TODOS al split donde el grupo ya tenia mas miembros (mayoria) -- minimiza
# cuanto se mueve. En caso de empate, prioriza test y despues valid: mover un
# grupo entero a train por un empate podria vaciar por completo el holdout de
# una clase, que es peor que dejar el empate resuelto a favor de la evaluacion.
SPLIT_TIEBREAK_PRIORITY = ["test", "valid", "train"]


def source_key(filename: str) -> str:
    """Foto de origen de un recorte, para poder agrupar 'hermanos'.

    Roboflow exporta como "<nombre_original>_png.rf.<hash>.jpg". Si el nombre
    original viene de remove.bg con varios pellets recortados de la misma
    foto, tiene forma "<...>__<indice>_-removebg-preview". Se le saca el hash
    de Roboflow, el sufijo de remove.bg y el indice de recorte para quedarse
    con un identificador comun a todos los recortes de la misma foto.
    """
    stem = filename.split("_png.rf.")[0]
    stem = re.sub(r"-removebg-preview$", "", stem)
    stem = re.sub(r"__\d+_$", "", stem)
    return stem


def black_fraction(path: Path) -> float:
    img = Image.open(path).convert("RGB")
    # Reducido para que el conteo de pixels sea rapido; suficiente para estimar la fraccion.
    img = img.resize((64, 64))
    pixels = img.getdata()
    black = sum(1 for r, g, b in pixels if r < BLACK_PIXEL_THRESHOLD and g < BLACK_PIXEL_THRESHOLD and b < BLACK_PIXEL_THRESHOLD)
    return black / len(pixels)


def collect_images(dataset_dir: Path, class_name: str) -> dict[str, list[tuple[str, Path]]]:
    """Todas las imagenes de una clase, en los 3 splits, como (split, path)."""
    by_split = defaultdict(list)
    for split in SPLITS:
        class_dir = dataset_dir / split / class_name
        if not class_dir.exists():
            continue
        for p in sorted(class_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                by_split[split].append(p)
    return by_split


def resolve_leaks(by_split: dict[str, list[Path]]) -> tuple[dict[str, list[Path]], int]:
    """Reasigna los grupos de recortes-hermanos repartidos entre splits, todos
    al split donde el grupo ya tenia mas miembros (ver comentario de
    SPLIT_TIEBREAK_PRIORITY mas arriba)."""
    members_by_group = defaultdict(lambda: defaultdict(list))
    for split, paths in by_split.items():
        for p in paths:
            members_by_group[source_key(p.name)][split].append(p)

    leaked_keys = {key for key, splits in members_by_group.items() if len(splits) > 1}

    resolved = defaultdict(list)
    for split, paths in by_split.items():
        for p in paths:
            if source_key(p.name) not in leaked_keys:
                resolved[split].append(p)

    for key in leaked_keys:
        splits_for_group = members_by_group[key]
        best_count = max(len(v) for v in splits_for_group.values())
        candidates = [s for s, v in splits_for_group.items() if len(v) == best_count]
        target = next(s for s in SPLIT_TIEBREAK_PRIORITY if s in candidates)
        for split_members in splits_for_group.values():
            resolved[target].extend(split_members)

    return resolved, len(leaked_keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_dir",
        default=r"C:\Users\maca\OneDrive - Universidad de Talca\Escritorio\PollenColor Cropped.v2-pollencolor-cropped-corrected-v2.folder",
        help="carpeta con los subdirectorios train, valid y test",
    )
    ap.add_argument("--output_dir", default="data")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    if not dataset_dir.exists():
        raise SystemExit(f"No existe el dataset: {dataset_dir}")

    class_names = sorted({
        p.name for split in SPLITS for p in (dataset_dir / split).iterdir() if (dataset_dir / split).exists() and p.is_dir()
    })
    if not class_names:
        raise SystemExit(f"No encontre carpetas de clase dentro de {dataset_dir}/train (ni valid/test).")

    print(f"{'split':12s} {'clase':30s} {'imagenes':>8s}")
    total_leaked_groups = 0
    total_discarded = 0

    for class_name in class_names:
        by_split = collect_images(dataset_dir, class_name)
        resolved, n_leaked = resolve_leaks(by_split)
        total_leaked_groups += n_leaked
        if n_leaked:
            print(f"  [fuga] {class_name}: {n_leaked} foto(s) de origen con recortes repartidos "
                  f"entre splits -> cada grupo se consolido en el split donde tenia mas recortes.")

        for split, output_split in SPLITS.items():
            images = sorted(resolved.get(split, []))
            kept = []
            for image in images:
                frac = black_fraction(image)
                if frac > MAX_BLACK_FRACTION:
                    total_discarded += 1
                    print(f"  [descartada] {split}/{class_name}/{image.name} "
                          f"({frac:.0%} de la imagen es negro, no parece tener pellet visible)")
                    continue
                kept.append(image)

            destination_dir = output_dir / output_split / class_name
            if destination_dir.exists():
                shutil.rmtree(destination_dir)
            if not kept:
                continue
            destination_dir.mkdir(parents=True)
            for index, image in enumerate(kept):
                destination_name = f"{index:06d}_{image.name}"
                shutil.copy2(image, destination_dir / destination_name)
            print(f"{split:12s} {class_name:30s} {len(kept):8d}")

    print(f"\nListo. {total_leaked_groups} grupo(s) de recortes-hermanos consolidados en un solo "
          f"split, {total_discarded} foto(s) casi negras descartadas.")
    print("Se conservaron los splits (corregidos):")
    print(f"  Entrenamiento : {output_dir / 'train_instance'}/<clase>/")
    print(f"  Validacion    : {output_dir / 'valid'}/<clase>/")
    print(f"  Test reservado: {output_dir / 'holdout_test'}/<clase>/")


if __name__ == "__main__":
    main()

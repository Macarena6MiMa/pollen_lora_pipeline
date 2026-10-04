"""
Genera imagenes sinteticas a partir de los LoRA ya entrenados (uno por clase).

Antes de correr esto: python run_training.py  (necesita token_map.json)

Uso:
    python generate_synthetic.py --n 50
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DiffusionPipeline

MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# El safety checker de SD1.5 reemplaza por un frame negro cualquier imagen que
# su detector de NSFW marque como sospechosa. Con fotos macro de polen (texturas
# y colores fuera de lo que el detector espera) da falsos positivos seguido.
# Como esto es para generar dataset interno (no para publicar), lo desactivamos
# y en su lugar detectamos y reintentamos nosotros mismos las imagenes negras
# -- asi cubrimos tambien cualquier otra causa (ej. overflow numerico en fp16).
MAX_BLACK_FRACTION = 0.97
BLACK_PIXEL_THRESHOLD = 15
MAX_RETRIES = 4


def is_black(image) -> bool:
    arr = np.asarray(image.resize((64, 64)).convert("RGB"))
    black_pixels = np.all(arr < BLACK_PIXEL_THRESHOLD, axis=-1).sum()
    return black_pixels / (arr.shape[0] * arr.shape[1]) > MAX_BLACK_FRACTION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="imagenes sinteticas a generar por clase")
    ap.add_argument("--steps", type=int, default=30, help="pasos de inferencia por imagen")
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--out_dir", default="data/synthetic")
    ap.add_argument("--seed_offset", type=int, default=1000)
    args = ap.parse_args()

    token_map_path = Path("token_map.json")
    if not token_map_path.exists():
        raise SystemExit("No encuentro token_map.json. Corre run_training.py primero.")

    with open(token_map_path, encoding="utf-8") as f:
        token_map = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cpu":
        print("[!] No se detecto GPU, la generacion en CPU va a ser muy lenta.")

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    out_root = Path(args.out_dir)
    loaded_any = False

    for cls_name, info in token_map.items():
        lora_dir = Path(info["output_dir"])
        if not (lora_dir / "pytorch_lora_weights.safetensors").exists():
            print(f"[!] No encuentro pesos LoRA en {lora_dir}, salteo '{cls_name}'.")
            continue

        prompt = info["prompt"]
        cls_out = out_root / cls_name
        cls_out.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Generando {args.n} imagenes para '{cls_name}' ({prompt}) ===")

        if loaded_any:
            pipe.unload_lora_weights()
        pipe.load_lora_weights(str(lora_dir))
        loaded_any = True

        discarded = 0
        for i in range(args.n):
            for attempt in range(MAX_RETRIES + 1):
                seed = args.seed_offset + i + attempt * 100_000
                generator = torch.Generator(device=device).manual_seed(seed)
                image = pipe(
                    prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images[0]
                if not is_black(image):
                    break
                discarded += 1
                print(f"  [!] imagen {i:03d} salio negra (intento {attempt + 1}/{MAX_RETRIES + 1}), reintentando con otra semilla...")
            else:
                print(f"  [!] imagen {i:03d} de '{cls_name}' siguio negra tras {MAX_RETRIES + 1} intentos, la salteo.")
                continue
            image.save(cls_out / f"{cls_name}_synth_{i:03d}.png")

        if discarded:
            print(f"  ({discarded} imagenes negras descartadas y reintentadas en el camino)")
        print(f"  guardadas en {cls_out}/")

    print("\nListo. Revisa data/synthetic/<especie>/ a ojo antes de sumarlas al entrenamiento del clasificador.")


if __name__ == "__main__":
    main()

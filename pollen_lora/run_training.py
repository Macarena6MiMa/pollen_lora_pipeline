"""
Entrena un LoRA (DreamBooth) por cada especie encontrada en data/train_instance/<especie>/.

Antes de correr esto:
    1. python prepare_data.py --holdout 2
    2. accelerate config default   (una sola vez por maquina)

Uso:
    python run_training.py

Cada especie termina con su propio adaptador LoRA en models/<especie>/, y un
token unico (sks00, sks01, ...) que se guarda en token_map.json para que
generate_synthetic.py sepa que LoRA y que prompt corresponden a cada clase.
"""
import json
import subprocess
import sys
from pathlib import Path

MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
TRAIN_DIR = Path("data/train_instance")
MODELS_DIR = Path("models")

RESOLUTION = 512
RANK = 8                        # subido de 4: con mas imagenes por clase, rank 4 se quedaba corto para fijar color/textura
LEARNING_RATE = "1e-4"         # el valor recomendado en la guia oficial para LoRA (mas alto que en fine-tuning completo)
SEED = 42

# Los pasos escalan con la cantidad de imagenes de cada clase (reales + aumentadas)
# en vez de un numero fijo: un valor fijo pensado para ~8 fotos deja el LoRA
# muy sub-entrenado cuando una clase tiene 100+ imagenes (menos de 2 epocas en 300 pasos).
TARGET_EPOCHS = 15              # cuantas veces en promedio el modelo recorre el dataset de cada clase
MIN_TRAIN_STEPS = 300           # piso para clases muy chicas
CHECKPOINTING_STEPS = 50        # guarda un checkpoint cada 50 pasos para poder elegir uno mas temprano si sobreajusta
VALIDATION_EVERY_N_EPOCHS = 5   # genera una imagen de preview cada 5 epocas para poder inspeccionar visualmente

# Si alguna especie en particular necesita mas o menos pasos (fotos muy variadas
# entre si, o al reves muy uniformes), agregala aca y se usa en vez del calculo
# automatico. Ej: {"especie_03": 500}
MAX_TRAIN_STEPS_OVERRIDES = {}


def main():
    classes = sorted(p for p in TRAIN_DIR.iterdir() if p.is_dir())
    if not classes:
        sys.exit(f"No encontre carpetas de clase en {TRAIN_DIR}. Corre prepare_data.py primero.")

    MODELS_DIR.mkdir(exist_ok=True)
    token_map = {}
    failed = []

    for i, cls_dir in enumerate(classes):
        token = f"sks{i:02d}"
        prompt = f"a photo of {token} corbicular pollen pellet"
        output_dir = MODELS_DIR / cls_dir.name

        n_images = len(list(cls_dir.glob("*")))
        steps = MAX_TRAIN_STEPS_OVERRIDES.get(cls_dir.name, max(MIN_TRAIN_STEPS, n_images * TARGET_EPOCHS))

        token_map[cls_dir.name] = {
            "token": token,
            "prompt": prompt,
            "output_dir": str(output_dir),
        }

        print(f"\n=== [{i + 1}/{len(classes)}] Entrenando '{cls_dir.name}' ({n_images} imagenes, {steps} pasos) -> token '{token}' ===")

        cmd = [
            "launch", "train_dreambooth_lora.py",
            f"--pretrained_model_name_or_path={MODEL_NAME}",
            f"--instance_data_dir={cls_dir}",
            f"--output_dir={output_dir}",
            f"--instance_prompt={prompt}",
            f"--resolution={RESOLUTION}",
            "--train_batch_size=1",
            "--gradient_accumulation_steps=1",
            f"--checkpointing_steps={CHECKPOINTING_STEPS}",
            f"--learning_rate={LEARNING_RATE}",
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            f"--max_train_steps={steps}",
            f"--rank={RANK}",
            f"--validation_prompt={prompt}",
            f"--validation_epochs={VALIDATION_EVERY_N_EPOCHS}",
            f"--seed={SEED}",
            "--mixed_precision=fp16",
            "--gradient_checkpointing",
            "--use_8bit_adam",
            # Si tenes xformers instalado, sumar esta linea ahorra bastante VRAM:
            # "--enable_xformers_memory_efficient_attention",
        ]
        cmd = [
            sys.executable,
            "-m",
            "accelerate.commands.accelerate_cli",
            *cmd,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[!] El entrenamiento de '{cls_dir.name}' termino con error, revisa el log de arriba.")
            failed.append(cls_dir.name)

    with open("token_map.json", "w", encoding="utf-8") as f:
        json.dump(token_map, f, indent=2, ensure_ascii=False)

    print(f"\nListo. Mapeo de clases guardado en token_map.json")
    if failed:
        print(f"[!] Estas clases fallaron y no tienen LoRA valido: {', '.join(failed)}")


if __name__ == "__main__":
    main()

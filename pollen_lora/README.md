# Pipeline de LoRA (DreamBooth) para polen corbicular

Fine-tuning de Stable Diffusion 1.5 con LoRA, un adaptador por especie, para generar
imagenes sinteticas de cargas corbiculares a partir de ~10 fotos reales por clase.

Verificado contra el script y la documentacion oficial de `huggingface/diffusers`
(carpeta `examples/dreambooth`, licencia Apache-2.0) al momento de armar esto.

## Que hace cada archivo

| Archivo | Que hace |
|---|---|
| `prepare_data.py` | Importa los splits existentes (`train`, `valid`, `test`) usando la carpeta de clase |
| `augment_data.py` | Agrega copias aumentadas (`_aug`) de las fotos de `data/train_instance/` |
| `train_dreambooth_lora.py` | Script oficial de `diffusers`, sin modificar. No hace falta tocarlo |
| `run_training.py` | Entrena un LoRA por especie llamando al script anterior |
| `generate_synthetic.py` | Genera imagenes sinteticas con los LoRA ya entrenados |
| `train_classifier.py` | Entrena un clasificador (transfer learning) y compara real vs real+sintetico contra el holdout |
| `export_classifier_ncnn.py` | Exporta el clasificador de `train_classifier.py` a NCNN para la Raspberry |
| `pipeline_color_svm.py` | Pipeline alternativo B: descriptores de color + SVM (seccion 6) |
| `pipeline_dinov2_linear.py` | Pipeline alternativo C: DINOv2 congelado + regresion logistica (seccion 6) |
| `compare_pipelines.py` | Compara todos los clasificadores con el mismo protocolo (seccion 6) |
| `cv_common.py` | Funciones compartidas de validacion cruzada agrupada (no se corre directamente) |

## Orden de ejecucion

Todos los comandos se corren desde la carpeta `pollen_lora/`. Los pipelines no dependen
entre si; lo que importa es el orden de las etapas:

1. **Preparar los datos** (una vez): `prepare_data.py` y despues `augment_data.py`
   (seccion 2). Si volves a correr `prepare_data.py`, borra las carpetas de destino
   (incluidas las copias aumentadas): siempre corre `augment_data.py` despues.
2. **Comparar** los clasificadores con `compare_pipelines.py` (seccion 6).
3. **Entrenar el modelo final** del ganador con el hiperparametro elegido (seccion 6).
4. **Exportar y copiar** a la Raspberry (seccion 6).

El LoRA (secciones 3 a 5) es un camino aparte: sirve para probar si las imagenes
sinteticas ayudan, y eso se mide con `train_classifier.py` (real_only vs
real_plus_synthetic). `compare_pipelines.py` usa solo fotos reales.

## 1. Instalacion

Vas a necesitar una GPU NVIDIA (con las optimizaciones ya activadas en `run_training.py`,
~8 GB de VRAM alcanzan). Si no tenes una a mano, un Colab gratuito con GPU T4 sobra.

```bash
git clone https://github.com/huggingface/diffusers
cp diffusers/examples/dreambooth/train_dreambooth_lora.py .   # ya incluido en este paquete, por las dudas

pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121   # ajusta cu121 a tu version de CUDA (ver pytorch.org)

accelerate config default
```

## 2. Prepara el dataset

```
PollenColor.v1-pollencolor-clasificado-v1.folder/
  train/
    aromo/       (imagenes de entrenamiento)
  valid/
    aromo/       (imagenes de validacion)
  test/
    aromo/       (imagenes de test)
```

Cada subcarpeta dentro de `train`, `valid` y `test` se interpreta como una clase. El nombre
del archivo no se usa para clasificar la imagen.

```powershell
python prepare_data.py
```

La ruta por defecto es la carpeta del dataset indicada arriba. Para usar otra ruta:

```powershell
python prepare_data.py --dataset_dir "C:\ruta\al\dataset"
```

Esto copia `train` a `data/train_instance/`, `valid` a `data/valid/` y `test` a
`data/holdout_test/`. El test queda afuera del entrenamiento y es la particion reservada
para evaluar el clasificador final.

Despues, agrega las copias aumentadas (solo toca `data/train_instance/`):

```powershell
python augment_data.py --copies 8
```

`prepare_data.py` borra y vuelve a crear las carpetas de destino, asi que cada vez que
lo corras tenes que volver a correr `augment_data.py`.

## 3. Entrena

```bash
python run_training.py
```

Entrena un LoRA por especie, una a la vez. Con 10 clases y ~8 imagenes de entrenamiento
cada una, contá con algo entre 15 y 40 minutos en una GPU de gama media (bastante menos
en una mas potente). Cada LoRA queda en `models/<especie>/`, guardando un checkpoint
cada 50 pasos — si el ultimo checkpoint sobreajusta (las imagenes de validacion empiezan
a verse practicamente identicas a las fotos de entrenamiento), pasale a
`generate_synthetic.py` un checkpoint intermedio en vez del final.

Para revisar las imagenes de validacion generadas durante el entrenamiento:

```bash
tensorboard --logdir models
```

Es la mejor senal que tenes de sobre/sub-ajuste sin necesitar metricas formales.

## 4. Genera las imagenes sinteticas

```bash
python generate_synthetic.py --n 50
```

Genera 50 imagenes por especie en `data/synthetic/<especie>/`. Revisalas a ojo antes de
usarlas: si alguna especie salio con imagenes rotas o irreconocibles, bajá
`MAX_TRAIN_STEPS` en `run_training.py` para esa clase (o el override por clase que ya
esta armado en el script) y reentrena; si se ven demasiado genericas o poco parecidas al
pellet real, probá subirlo.

## 5. Entrena el clasificador y compara real vs real+sintetico

```bash
python train_classifier.py
```

Hace transfer learning con una CNN preentrenada (ResNet18 por default; `--model
efficientnet_b0` como alternativa) y entrena dos versiones: una solo con las fotos
reales (`data/train_instance/`, incluye las aumentadas de `augment_data.py`) y otra
sumando ademas `data/synthetic/`. Las dos se evaluan sobre `data/holdout_test/`, que
nunca vio ni el LoRA ni el clasificador, y el resultado (accuracy, precision/recall
por clase, matriz de confusion) queda en `models/classifier/results.json`.

Si el clasificador con datos sinteticos no le gana al que solo usa reales, la
conclusion valida es que el aumento no esta ayudando todavia — no asumas que ayuda
solo porque las imagenes generadas se ven bien. En ese caso revisa `data/synthetic/`
a ojo y considera reentrenar los LoRA (mas pasos, otro rank, o descartar la clase
problematica) antes de sumarlas.

## 6. Pipelines alternativos de clasificacion y comparacion

Ademas del fine-tuning de ResNet18 (`train_classifier.py`) hay dos pipelines mas,
pensados para cuando hay muy pocas fotos reales:

| Archivo | Pipeline | Idea |
|---|---|---|
| `pipeline_color_svm.py` | **B: color + SVM** | ~55 descriptores de color del pellet (histogramas HSV, estadisticas Lab/RGB, ignorando el fondo negro) + SVM. Pone a prueba que el color alcance por si solo. Entrena en segundos. |
| `pipeline_dinov2_linear.py` | **C: DINOv2 + regresion logistica** | DINOv2 ViT-S/14 congelado como extractor de descriptores + regresion logistica. No re-entrena la red, solo 3x385 parametros. |
| `compare_pipelines.py` | comparacion | Corre ResNet18, EfficientNet-B0, B y C con el mismo protocolo |

Para decidir cual es mejor **no mires el holdout** (7 fotos: una sola foto cambia la
accuracy 14 puntos). `compare_pipelines.py` aplica el mismo protocolo a todos, de modo
que lo unico distinto sea el pipeline:

- **Mismos datos y semilla (42)**: los splits ya corregidos por fuga, `train_instance/`
  con sus copias `_aug`, las mismas transformaciones (sin hue).
- **Las CNN usan `train_one()` de `train_classifier.py` tal cual**: AdamW, 100 epocas,
  checkpoint en la mejor accuracy de validacion. Las dos redes ven los mismos batches
  en el mismo orden.
- **Mismo presupuesto de ajuste**: 3 valores elegidos con validacion (nunca con test).
  CNN: lr {1e-4, 3e-4, 1e-3}; color_svm: C {1, 10, 100}; dinov2_linear: C {0.1, 0.5, 2}.
- **Evaluacion** (`--protocol`):
  - `kfold` (default, recomendado): 5 folds agrupados por foto de origen (los mismos
    grupos de la correccion de fuga de `prepare_data.py`) sobre las 65 fotos reales.
    Dentro de cada fold se separa una validacion, tambien agrupada. Cada foto real se
    usa exactamente una vez como test, y nunca se evalua sobre una copia `_aug`.
  - `seeds` (minimo): los splits fijos, repetidos con 5 semillas; media +- desvio.

```bash
python compare_pipelines.py                                     # todos (las CNN necesitan GPU, ver abajo)
python compare_pipelines.py --pipelines color_svm dinov2_linear # solo B y C, ~1 minuto en CPU
python compare_pipelines.py --pipelines resnet18_finetune efficientnet_b0_finetune --output models/comparison/cnn.json
python compare_pipelines.py --protocol seeds
```

Si corres los pipelines por separado, usa `--output` con otro nombre para que la segunda
corrida no sobrescriba el `results.json` de la primera.

**Tiempo**: el protocolo completo de las CNN son 5 folds x 3 lr x 100 epocas por
arquitectura. En CPU son del orden de 40 horas para las dos; correlo en una GPU (por
ejemplo Colab, como el LoRA).

El resultado queda en `models/comparison/results.json`, con el hiperparametro elegido y
la validacion de cada corrida. Si la diferencia entre los dos mejores es menor que el
desvio entre corridas, el script lo avisa: con estos datos no se puede decir que uno
sea mejor.

### Entrenar el modelo final del ganador

En el `results.json` de la comparacion, cada corrida guarda el hiperparametro elegido
(`chosen_lr` o `chosen_C`). Usa el que mas se repite entre los folds:

| Ganador | Entrenar | Exportar |
|---|---|---|
| ResNet18 | `python train_classifier.py --model resnet18 --epochs 100 --lr <lr>` | `python export_classifier_ncnn.py --model resnet18 --weights models/classifier/real_only.pt` |
| EfficientNet-B0 | `python train_classifier.py --model efficientnet_b0 --epochs 100 --lr <lr>` | `python export_classifier_ncnn.py --model efficientnet_b0 --weights models/classifier/real_only.pt` |
| DINOv2 | `python pipeline_dinov2_linear.py --C <C> --export` | lo hace el mismo comando y lo verifica contra PyTorch |
| Color + SVM | `python pipeline_color_svm.py --C <C>` | no usa NCNN: queda en `models/color_svm/color_svm.joblib` |

`train_classifier.py` usa 20 epocas por default: pasale `--epochs 100` para que coincida
con lo que se comparo. `pipeline_dinov2_linear.py` y `pipeline_color_svm.py` entrenan con
train + valid (no necesitan validacion para elegir epoca).

### Despliegue en la Raspberry

- **ResNet18 / EfficientNet-B0 / DINOv2**: copia el `.ncnn.param` y el `.ncnn.bin` a la
  carpeta `MODELO_CLASIF` de `detector.py`, con los nombres `model.ncnn.param` y
  `model.ncnn.bin` (DINOv2 ya los deja con ese nombre en `models/dinov2_linear/`). Los tres
  tienen la misma entrada (`in0`, 1x3x224x224, normalizacion ImageNet) y salida (`out0`,
  3 logits), asi que `detector.py` los usa sin cambios.
- **DINOv2** en particular: el script verifica contra PyTorch que el modelo NCNN de la
  misma clase en todas las fotos antes de darlo por bueno. Es unas 3 veces mas pesado
  que el ResNet18 (~12 GFLOPs).
- **Color + SVM**: no usa NCNN. En la Raspberry hace falta `scikit-learn` y `joblib`, y
  reemplazar `clasificar_tipo_polen()` por `predict_bgr()` de `pipeline_color_svm.py`.

## Notas y troubleshooting

- El modelo base es `stable-diffusion-v1-5/stable-diffusion-v1-5`. La primera vez que
  corras `run_training.py` o `generate_synthetic.py` va a descargar ~4-5 GB de pesos.
- Los LoRA entrenados son chicos (unos pocos MB cada uno) — se pueden compartir o
  versionar sin problema.
- **CUDA out of memory**: en `run_training.py` bajá `RESOLUTION` a 384, o agregá
  `"--enable_xformers_memory_efficient_attention"` a la lista `cmd` si tenes `xformers`
  instalado (`pip install xformers`).
- **Las imagenes generadas no se parecen a nada**: el LoRA probablemente necesita mas
  pasos, o revisa que `instance_prompt` en `token_map.json` use el token correcto.
- **Las imagenes generadas son casi identicas a las fotos de entrenamiento**:
  sobreajuste — usa un checkpoint mas temprano o bajá `MAX_TRAIN_STEPS`.

"""Pipeline C: DINOv2 congelado + regresion logistica ("linear probe").

Hipotesis que pone a prueba: con ~65 fotos reales, re-entrenar una CNN entera
(lo que hace train_classifier.py) sobreajusta facil. DINOv2 es un modelo
autosupervisado entrenado sobre 142M de imagenes cuyos descriptores ya separan
muy bien categorias visuales sin fine-tuning. Aca NO se toca ningun peso de la
red: solo se extrae un vector de 384 numeros por imagen y se entrena una
regresion logistica encima -- 3 x 385 parametros en vez de 11 millones.

Que hace:
  1. Pasa cada imagen por DINOv2 ViT-S/14 (timm) a 224x224, con el MISMO
     preprocesamiento que train_classifier.py y detector.py (resize, RGB,
     normalizacion ImageNet).
  2. Entrena StandardScaler + LogisticRegression sobre esos vectores.
  3. Con --export, junta escalado + regresion en una sola capa lineal pegada
     a la red y la exporta a NCNN. El modelo resultante tiene la misma entrada
     ("in0", 1x3x224x224) y salida ("out0", 3 logits) que el ResNet18 actual,
     asi que detector.py lo usa sin cambiar una linea.

La primera vez descarga ~90 MB de pesos de Hugging Face. Corre bien en CPU.

Uso:
    python pipeline_dinov2_linear.py --export              # entrena con train+valid, evalua en holdout, exporta a NCNN
    python compare_pipelines.py --pipelines dinov2_linear   # validacion cruzada (lo que conviene mirar para comparar)
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from cv_common import load_samples, print_summary, summarize
from train_classifier import IMAGENET_MEAN, IMAGENET_STD, build_transforms

BACKBONE = "vit_small_patch14_dinov2.lvd142m"
SIZE = 224          # mismo tamano que espera detector.py (CLASIF_SIZE); timm re-interpola los embeddings de posicion
BATCH_SIZE = 16
MAX_NCNN_PROB_DIFF = 0.02   # tolerancia al comparar las probabilidades de NCNN contra PyTorch


def build_backbone() -> nn.Module:
    # Sin num_classes: la salida es el token CLS (384 numeros), no logits.
    return timm.create_model(BACKBONE, pretrained=True, num_classes=0, img_size=SIZE).eval()


class DinoV2LinearPipeline:
    name = "dinov2_linear"

    def __init__(self, seed: int = 42, C: float = 0.5):
        self.seed, self.C = seed, C
        self.backbone = build_backbone()
        self.transform = build_transforms(train=False, size=SIZE)
        self._cache: dict[Path, np.ndarray] = {}

    @torch.no_grad()
    def _features(self, paths) -> np.ndarray:
        todo = [p for p in dict.fromkeys(paths) if p not in self._cache]
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            x = torch.stack([self.transform(Image.open(p).convert("RGB")) for p in batch])
            for p, f in zip(batch, self.backbone(x).numpy()):
                self._cache[p] = f
        return np.stack([self._cache[p] for p in paths])

    def fit(self, paths, labels):
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.C, max_iter=5000, class_weight="balanced", random_state=self.seed),
        )
        self.model.fit(self._features(paths), labels)
        return self

    def predict(self, paths) -> np.ndarray:
        return self.model.predict(self._features(paths))

    def predict_proba(self, paths) -> np.ndarray:
        return self.model.predict_proba(self._features(paths))

    def to_torch(self) -> nn.Module:
        """Red + (StandardScaler -> LogisticRegression) plegados en una nn.Linear:
        W @ ((x - mu) / sigma) + b  ==  (W / sigma) @ x + (b - W @ (mu / sigma))."""
        scaler, logreg = self.model[0], self.model[1]
        W = logreg.coef_ / scaler.scale_
        b = logreg.intercept_ - W @ scaler.mean_
        if W.shape[0] == 1:  # con 2 clases sklearn guarda una sola fila
            W, b = np.vstack([np.zeros_like(W), W]), np.concatenate([[0.0], b])
        head = nn.Linear(W.shape[1], W.shape[0])
        head.weight.data = torch.tensor(W, dtype=torch.float32)
        head.bias.data = torch.tensor(b, dtype=torch.float32)
        return nn.Sequential(self.backbone, head).eval()


class NcnnFriendlyViT(nn.Module):
    """Mismo calculo que timm VisionTransformer.forward + cabeza lineal, para
    UNA imagen, sin que ninguna operacion mueva el eje de batch.

    Por que: la atencion de timm hace reshape(B, N, 3, heads, dim).permute(2, 0, ...)
    -- manda el batch al medio -- y NCNN no lo soporta: pnnx exporta igual (solo
    avisa "permute across batch dim is not supported") pero el modelo resultante
    da cualquier cosa. pnnx siempre toma el primer eje como batch, asi que aca
    se deja ese eje (=1) quieto en todas las operaciones."""

    def __init__(self, vit: nn.Module, head: nn.Linear):
        super().__init__()
        self.vit, self.head = vit, head
        for blk in vit.blocks:
            if getattr(blk.attn, "gate", None) is not None:
                raise ValueError("bloque de atencion con gate: no soportado por esta exportacion")

    @staticmethod
    def _heads(t: torch.Tensor, h: int, d: int) -> torch.Tensor:
        return t.reshape(1, -1, h, d).permute(0, 2, 1, 3)       # (1, N, C) -> (1, heads, N, dim)

    def _attention(self, attn: nn.Module, x: torch.Tensor) -> torch.Tensor:
        c, h, d = x.shape[-1], attn.num_heads, attn.head_dim
        qkv = attn.qkv(x)
        q = attn.q_norm(self._heads(qkv[:, :, :c], h, d)) * attn.scale
        k = attn.k_norm(self._heads(qkv[:, :, c:2 * c], h, d))
        v = self._heads(qkv[:, :, 2 * c:], h, d)
        out = (q @ k.transpose(-2, -1)).softmax(dim=-1) @ v     # (1, heads, N, dim)
        return attn.proj(attn.norm(out.permute(0, 2, 1, 3).reshape(1, -1, c)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vit = self.vit
        x = vit.patch_embed.proj(x)                             # (1, C, 16, 16)
        x = vit.patch_embed.norm(x.flatten(2).transpose(1, 2))  # (1, 256, C)
        x = torch.cat([vit.cls_token, x], dim=1) + vit.pos_embed
        for blk in vit.blocks:
            x = x + self._layer_scale(blk.ls1, self._attention(blk.attn, blk.norm1(x)))
            x = x + self._layer_scale(blk.ls2, blk.mlp(blk.norm2(x)))
        return self.head(vit.norm(x)[:, 0])                     # token CLS -> (1, clases)

    @staticmethod
    def _layer_scale(ls: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # gamma es un vector (C,); con forma (1, 1, C) pnnx no lo confunde con un eje de batch
        return x * ls.gamma.reshape(1, 1, -1) if hasattr(ls, "gamma") else ls(x)


def verify_ncnn(ncnn_param: Path, ncnn_bin: Path, model: nn.Module, paths) -> None:
    """Corre el .param/.bin con el preprocesamiento EXACTO de detector.py y lo
    compara contra PyTorch. Si no coincide, el modelo no sirve para la Raspberry."""
    try:
        import cv2
        import ncnn
    except ImportError:
        print("[!] No pude verificar el modelo NCNN (falta 'pip install ncnn opencv-python'). "
              "Verificalo antes de llevarlo a la Raspberry.")
        return

    mean, std = np.array(IMAGENET_MEAN, np.float32), np.array(IMAGENET_STD, np.float32)
    net = ncnn.Net()
    net.load_param(str(ncnn_param))
    net.load_model(str(ncnn_bin))
    max_diff, max_prob_diff, disagree = 0.0, 0.0, 0
    for p in paths:
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img = np.transpose((img - mean) / std, (2, 0, 1)).copy()
        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(img).clone())
            _, out0 = ex.extract("out0")
            ncnn_logits = np.array(out0).reshape(-1)
        with torch.no_grad():
            torch_logits = model(torch.from_numpy(img)[None]).numpy().reshape(-1)
        max_diff = max(max_diff, float(np.abs(ncnn_logits - torch_logits).max()))
        max_prob_diff = max(max_prob_diff, float(np.abs(softmax(ncnn_logits) - softmax(torch_logits)).max()))
        disagree += int(ncnn_logits.argmax() != torch_logits.argmax())
    # NCNN no reproduce PyTorch bit a bit (otra implementacion de GELU/LayerNorm),
    # asi que se exige la misma clase en todas las imagenes y probabilidades casi iguales.
    if disagree or max_prob_diff > MAX_NCNN_PROB_DIFF:
        raise SystemExit(f"[!] El modelo NCNN NO coincide con PyTorch ({disagree} predicciones distintas, "
                         f"diferencia maxima de probabilidad {max_prob_diff:.3f}). No lo lleves a la Raspberry.")
    print(f"Verificado: NCNN da la misma clase que PyTorch en {len(paths)} imagenes "
          f"(diferencia maxima: {max_prob_diff:.4f} en probabilidad, {max_diff:.4f} en logits).")


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dirs", nargs="+", default=["data/train_instance", "data/valid"])
    ap.add_argument("--test_dir", default="data/holdout_test")
    ap.add_argument("--output_dir", default="models/dinov2_linear")
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export", action="store_true", help="exportar a NCNN para la Raspberry (necesita pnnx)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    train, class_names = load_samples(args.train_dirs)
    test, test_classes = load_samples([args.test_dir])
    if test_classes != class_names:
        raise SystemExit("Las clases de train y test no coinciden entre si, revisa las carpetas.")

    print(f"=== Entrenando dinov2_linear ({len(train)} imagenes, {sum(not s.is_aug for s in train)} reales) ===")
    pipe = DinoV2LinearPipeline(seed=args.seed, C=args.C).fit([s.path for s in train], [s.label for s in train])

    test = [s for s in test if not s.is_aug]
    summary = summarize([s.label for s in test], pipe.predict([s.path for s in test]), class_names)
    print_summary(f"dinov2_linear (holdout, {len(test)} fotos)", summary, class_names)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = pipe.to_torch()

    # Control: el modelo plegado tiene que dar las mismas predicciones que sklearn.
    with torch.no_grad():
        x = torch.stack([pipe.transform(Image.open(s.path).convert("RGB")) for s in test])
        torch_logits = model(x).numpy()
    sk_logits = pipe.model.decision_function(pipe._features([s.path for s in test]))
    if not np.allclose(torch_logits, sk_logits.reshape(torch_logits.shape), atol=1e-3):
        raise SystemExit("[!] El modelo plegado no coincide con sklearn -- no lo exporto.")

    torch.save(model.state_dict(), output_dir / "dinov2_linear.pt")
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "holdout": summary}, f, indent=2, ensure_ascii=False)
    print(f"\nModelo guardado en {output_dir / 'dinov2_linear.pt'}")

    if args.export:
        from export_classifier_ncnn import export_to_ncnn

        export_model = NcnnFriendlyViT(model[0], model[1]).eval()
        with torch.no_grad():
            export_logits = torch.cat([export_model(xi[None]) for xi in x]).numpy()
        if not np.allclose(export_logits, torch_logits, atol=1e-3):
            raise SystemExit("[!] La version sin batch del ViT no coincide con la original -- no lo exporto.")

        ncnn_param, ncnn_bin = export_to_ncnn(export_model, output_dir / "dinov2_linear.torchscript.pt", SIZE)
        verify_ncnn(ncnn_param, ncnn_bin, export_model, [s.path for s in [*test, *train] if not s.is_aug])
        # detector.py busca model.ncnn.param / model.ncnn.bin dentro de la carpeta del clasificador
        shutil.copy2(ncnn_param, output_dir / "model.ncnn.param")
        shutil.copy2(ncnn_bin, output_dir / "model.ncnn.bin")
        print(f"\nListo. Para la Raspberry copia {output_dir / 'model.ncnn.param'} y "
              f"{output_dir / 'model.ncnn.bin'} a la carpeta MODELO_CLASIF de detector.py "
              f"(mismas clases en el mismo orden: {class_names}).")

    print("Ojo: el holdout tiene muy pocas fotos; para comparar pipelines usa compare_pipelines.py.")


if __name__ == "__main__":
    main()

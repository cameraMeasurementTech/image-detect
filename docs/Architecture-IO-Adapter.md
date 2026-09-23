# SN34 image I/O: base model → submission architecture

This note ties **gasbench harness I/O** ([Image-Discriminator-Gates.md](../../bitmind-subnet/docs/Image-Discriminator-Gates.md) Gates 1–3) to the adapters in `ai-image-detection`.

## What the subnet feeds and expects

| | Contract |
|--|----------|
| **Input** | `torch.uint8` tensor `[B, 3, H, W]`, RGB, values `0..255`. `H,W` = `preprocessing.resize` in `model_config.yaml` (often `224` or `384`). |
| **Output** | Raw **logits** `[B, 3]` — class order **`[real, synthetic, semisynthetic]`**. Softmax is applied by the scorer, not by you. |
| **Load** | Offline `load_model(weights_path, num_classes=3, **cfg.model)` → `nn.Module` in eval mode. No network, no blocked imports. |

Harness call site is effectively:

```python
logits = model(uint8_batch)   # positional; NOT pixel_values=...
```

## Does the downloaded base model meet this?

**Almost never by itself.** Probe it:

```bash
python scripts/probe_base_inference.py --config configs/train_vit_phase1.yaml
python scripts/check_gates.py --config configs/train_vit_phase1.yaml
```

Typical findings for `google/vit-base-patch16-224-in21k`:

| Pattern | Result |
|---------|--------|
| HF `model(pixel_values=x.float()/255)` | Runs (training style) — **not** the harness API |
| HF `model(uint8_batch)` positional | Often **runs** but **mis-scales** (cast to float without `/255` + ImageNet norm) → wrong logits |
| ImageNet-1k head (`num_labels=1000`) | **Fail** — last dim must be 3 |
| Binary deepfake head (`num_labels=1` or `2`) | **Fail** for image SN34 multiclass |
| `Uint8ImageClassifier` wrap | **Pass** I/O (still need training for ≥80% exam) |

So: keep the backbone weights; **change the head + wrap the forward**.

## Minimal fix (recommended default)

Already implemented in `src/models/vit_detector.py` + `src/models/export_bundle.py`:

```text
HF / timm backbone (features or logits)
        │
        ▼
Uint8ImageClassifier
  • uint8 → float / 255
  • ImageNet (or CLIP) mean/std
  • backbone(...)
  • return bare logits Tensor [B,3]
        │
        ▼
export → model.py + model.safetensors + model_config.yaml
```

Train-time:

```python
from src.models.vit_detector import load_hf_classifier, build_export_wrapper

backbone = load_hf_classifier("google/vit-base-patch16-224-in21k", num_classes=3)
model = build_export_wrapper(backbone)   # use this for train + export
```

`num_labels=3` + `ignore_mismatched_sizes=True` replaces a missing/wrong head; pretrained trunk stays.

## If your architecture does not fit

### 1) Wrong number of classes

```python
# HF
backbone = AutoModelForImageClassification.from_pretrained(
    backbone_id, num_labels=3, ignore_mismatched_sizes=True
)

# timm feature + new head
import timm, torch.nn as nn
trunk = timm.create_model("convnext_base", pretrained=True, num_classes=0)
head = nn.Linear(trunk.num_features, 3)
```

Map old binary labels when fine-tuning: `fake → synthetic (1)`, `real → 0`; add semisynthetic data for class `2`.

### 2) Backbone wants `pixel_values=` or NHWC / different size

Put **all** conversion inside one `forward(x)`:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B,3,H,W] uint8 from gasbench
    x = x.float() / 255.0
    x = (x - self.mean) / self.std
    if x.shape[-1] != self.native_size:
        x = torch.nn.functional.interpolate(
            x, size=(self.native_size, self.native_size), mode="bilinear", align_corners=False
        )
    out = self.backbone(pixel_values=x)   # or self.backbone(x) for timm
    logits = out.logits if hasattr(out, "logits") else out
    return logits  # [B,3] bare tensor preferred
```

Declare the harness size in config (what you are *fed*), then resize internally if the trunk needs another resolution:

```yaml
preprocessing:
  resize: [384, 384]   # what gasbench sends
```

### 3) Binary logits / single logit

Do **not** return `[B,1]` (sigmoid path is legacy and not competitive). Expand to 3-way:

```python
class BinaryToThree(nn.Module):
    """Optional bridge while migrating a binary detector."""
    def __init__(self, binary_backbone, hidden=768):
        super().__init__()
        self.backbone = binary_backbone  # returns [B,1] or [B,2]
        self.lift = nn.Linear(1, 3)       # or Linear(2,3); train this

    def forward(self, x):
        x = x.float() / 255.0
        # ... normalize ...
        y = self.backbone(x)
        y = y.logits if hasattr(y, "logits") else y
        if y.shape[-1] == 2:
            y = y[:, 1:2] - y[:, 0:1]  # fake-real margin
        return self.lift(y)
```

Prefer training a true 3-class head on the trunk features instead of this bridge.

### 4) Ensemble (BMF-style)

Average **3-class logits** from each branch after a shared uint8 preprocess (or per-branch mean/std):

```python
class Ensemble(nn.Module):
    def forward(self, x):
        x = x.float() / 255.0
        logits = 0
        for branch, w in zip(self.branches, self.weights):
            logits = logits + w * branch(self.norm(x))  # each returns [B,3]
        return logits
```

Export **one** `nn.Module` and **one** `model.safetensors` zip. All imports in `model.py` must stay on the sandbox allowlist (`torch`, `transformers`, `timm`, `safetensors`, … — no `os`/`requests`).

### 5) Custom `model.py` without the training wrapper

Mirror the gates doc starter: bake preprocess + head into the submitted module, load only local weights:

```python
def load_model(weights_path: str, num_classes: int = 3, **kwargs):
    model = ImageDetector(num_classes=num_classes)
    model.load_state_dict(load_file(weights_path))
    model.train(False)
    return model
```

Use `python scripts/check_gates.py --model-dir ...` after packaging.

## Decision flowchart

```text
probe_base_inference.py
        │
        ├─ head ≠ 3 ──────────────► replace head (num_classes=3)
        ├─ raw uint8 wrong/fail ──► wrap Uint8ImageClassifier / custom forward
        ├─ output not [B,3] tensor ► strip .logits / map dict → tensor
        └─ I/O PASS ──────────────► train on gasbench labels → check_gates → exam
```

## Commands cheat sheet

```bash
# 1) Inference probe (synthetic uint8 only)
python scripts/probe_base_inference.py --config configs/train_vit_phase1.yaml

# 2) Full structural gates on exported probe zip
python scripts/check_gates.py --config configs/train_vit_phase1.yaml \
  --keep-probe-dir "$DATA/submission_probe"

# 3) After training: score real images (small download optional)
python scripts/check_gates.py --model-dir "$DATA/submission" --mini-exam \
  --data-dir "$DATA/mini" --dataset-limit 2 --max-per-source 32
```

# Neural Network Implementations: NumPy · PyTorch · TensorFlow

> 3-layer deep neural network for nonlinear regression — built 8 ways across NumPy, PyTorch, and TensorFlow.

---

## Problem

**Equation:** &nbsp; $y = \sin(x_1)\cos(x_2) + x_3^2 + 0.5\sin(x_1 x_3) + \varepsilon$

**Architecture (all colabs):**
```
Input (3)  →  Dense (64, ReLU)  →  Dense (32, ReLU)  →  Output (1, Linear)
```
2000 synthetic samples · 80/20 train-test split · 4D visualization via PCA

---

## Files

| File | Description |
|------|-------------|
| [`colab_a_numpy_scratch.ipynb`](./colab_a_numpy_scratch.ipynb) | NumPy + `tf.einsum` — fully manual backprop |
| [`colab_b_pytorch_scratch.ipynb`](./colab_b_pytorch_scratch.ipynb) | PyTorch raw tensors — autograd, manual update |
| [`colab_c_pytorch_classes.ipynb`](./colab_c_pytorch_classes.ipynb) | PyTorch `nn.Module` — Adam, DataLoader |
| [`colab_d_pytorch_lightning.ipynb`](./colab_d_pytorch_lightning.ipynb) | PyTorch Lightning — Trainer, callbacks |
| [`colab_e_tensorflow_variants.ipynb`](./colab_e_tensorflow_variants.ipynb) | TensorFlow × 4 variants |

---

## Video Walkthroughs

### Colab A — NumPy from Scratch
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat&logo=youtube)](https://youtu.be/K-5ZSKlDciM)
[![Colab File](https://img.shields.io/badge/Google%20Drive-Colab%20File-blue?style=flat&logo=googledrive)](https://drive.google.com/file/d/1GjTkmQAxEdttYr1JOV-8mgMYxJb7ahG-/view?usp=drive_link)
[![Video File](https://img.shields.io/badge/Google%20Drive-Video%20File-orange?style=flat&logo=googledrive)](https://drive.google.com/file/d/1hy7FIqOGgm5qvUa4pMlYhyOYYkI05C40/view?usp=sharing)

Synthetic data · 4D PCA plot · manual `relu_derivative()` · He init · `tf.einsum('bi,ij->bj')` · chain rule backprop layer-by-layer · SGD with momentum · loss curve + residuals

---

### Colab B — PyTorch from Scratch
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat&logo=youtube)](https://youtu.be/R2Rgo6z6EKY)
[![Colab File](https://img.shields.io/badge/Google%20Drive-Colab%20File-blue?style=flat&logo=googledrive)](https://drive.google.com/file/d/19uLbX8tpyPr1F_AOdW1VaWFRAhvELj1U/view?usp=drive_link)
[![Video File](https://img.shields.io/badge/Google%20Drive-Video%20File-orange?style=flat&logo=googledrive)](https://drive.google.com/file/d/1bM2-EY5QVoRcNYNiY6VRv-NFTtIt7ScM/view?usp=sharing)

`requires_grad=True` · `loss.backward()` fills `.grad` · `torch.no_grad()` during update · manual `w -= lr * w.grad` · `w.grad.zero_()` — why this is mandatory

---

### Colab C — PyTorch Class-Based
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat&logo=youtube)](https://youtu.be/c4U2zFxWos4)
[![Colab File](https://img.shields.io/badge/Google%20Drive-Colab%20File-blue?style=flat&logo=googledrive)](https://drive.google.com/file/d/1-wN9wLMMNScNOg1EKnHXWfTpH4gWHIwR/view?usp=drive_link)
[![Video File](https://img.shields.io/badge/Google%20Drive-Video%20File-orange?style=flat&logo=googledrive)](https://drive.google.com/file/d/14d_VU7NT_vn-WLDfGdnD_y3CW-cjP8nN/view?usp=sharing)

`nn.Module` + `forward()` · `nn.Linear` · `DataLoader` · the 4-line pattern: `zero_grad → forward → backward → step` · Adam · `StepLR` scheduler

---

### Colab D — PyTorch Lightning
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat&logo=youtube)](https://youtu.be/MERHWZRTpFE)
[![Colab File](https://img.shields.io/badge/Google%20Drive-Colab%20File-blue?style=flat&logo=googledrive)](https://drive.google.com/file/d/1wWH1BeGSBgGskWMu_4x89NhpJnV8OkjI/view?usp=drive_link)
[![Video File](https://img.shields.io/badge/Google%20Drive-Video%20File-orange?style=flat&logo=googledrive)](https://drive.google.com/file/d/1xJIqN4mYRP-jxY_yx4ALPKCLMT8mQXjH/view?usp=sharing)

`LightningDataModule` · `training_step()` / `validation_step()` hooks · `configure_optimizers()` · `EarlyStopping` · `ModelCheckpoint` · `trainer.fit()` replaces the entire training loop

---

### Colab E — TensorFlow (4 Variants)
[![YouTube](https://img.shields.io/badge/YouTube-Watch-red?style=flat&logo=youtube)](https://youtu.be/nrrK8BgdiW4)
[![Colab File](https://img.shields.io/badge/Google%20Drive-Colab%20File-blue?style=flat&logo=googledrive)](https://drive.google.com/file/d/1zdOBj_vXAnzOlvpKJGC4AGUHZ8egn2Yg/view?usp=drive_link)
[![Video File](https://img.shields.io/badge/Google%20Drive-Video%20File-orange?style=flat&logo=googledrive)](https://drive.google.com/file/d/1DVYIQHq2I-AHO9hVrD421YrZ3mCDe1HP/view?usp=sharing)

| Variant | Approach |
|---------|----------|
| **E-i** | `tf.Variable` + `GradientTape` + `tape.gradient()` + `apply_gradients()` |
| **E-ii** | `Dense` layers + manual `GradientTape` loop |
| **E-iii** | Functional API — `keras.Input` → `keras.Model` → `model.fit` |
| **E-iv** | `Sequential` → `compile` → `fit` |

---

## Open in Colab

| | |
|-|-|
| A — NumPy Scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_a_numpy_scratch.ipynb) |
| B — PyTorch Scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_b_pytorch_scratch.ipynb) |
| C — PyTorch Classes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_c_pytorch_classes.ipynb) |
| D — PyTorch Lightning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_d_pytorch_lightning.ipynb) |
| E — TensorFlow Variants | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_e_tensorflow_variants.ipynb) |

> Replace `YOUR_USERNAME/YOUR_REPO` with your actual GitHub path to activate these badges.

---

## Concept Index

| Concept | Where |
|---------|-------|
| Manual chain rule — `dL/dW = dL/dA · dA/dZ · dZ/dW` | Colab A · Section 7 |
| `tf.einsum('bi,ij->bj')` | Colab A · Section 6 |
| He initialization `√(2/fan_in)` | Colabs A, B, C |
| 4D scatter plot via PCA + color | Colab A · Section 3 |
| `requires_grad=True` + graph construction | Colab B · Section 3 |
| `w.grad.zero_()` — gradient accumulation problem | Colab B · Section 5 |
| `zero_grad → forward → backward → step` | Colab C · Section 4 |
| `LightningModule` hooks + `Trainer` | Colab D · Sections 4–5 |
| `tf.GradientTape` context manager | Colab E-i, E-ii |
| Keras Functional API | Colab E-iii |

---

## Expected Results

| Metric | Range |
|--------|-------|
| R² Score | 0.90 – 0.99 |
| RMSE | 0.07 – 0.25 |
| Final Val Loss | < 0.05 |

**Runtime:** `Runtime → Change runtime type → T4 GPU` recommended.  
**Dependencies:** `numpy` `tensorflow` `torch` `matplotlib` `scikit-learn` · lightning auto-installed in Colab D.

---

*Neural Networks Assignment · Spring 2025*

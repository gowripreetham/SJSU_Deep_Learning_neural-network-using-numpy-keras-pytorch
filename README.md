# Neural Network Implementations: NumPy · PyTorch · TensorFlow

> **Assignment:** Build a 3-layer deep neural network for nonlinear regression across 8 implementations spanning NumPy, PyTorch, and TensorFlow — with full video walkthroughs for each.

---

## 📌 Problem Statement

**Task:** Nonlinear regression on a 3-variable equation:

$$y = \sin(x_1) \cdot \cos(x_2) + x_3^2 + 0.5 \cdot \sin(x_1 \cdot x_3) + \varepsilon$$

- 3 input variables (x₁, x₂, x₃) with nonlinear interactions
- 2000 synthetic samples with Gaussian noise (σ = 0.05)
- 4D visualization using PCA dimensionality reduction + color as 4th dimension

**Network Architecture (consistent across all colabs):**
```
Input (3)  →  Hidden Layer 1 (64 neurons, ReLU)  →  Hidden Layer 2 (32 neurons, ReLU)  →  Output (1, Linear)
```

---

## 📂 File Index

| File | Description | Key Concepts |
|------|-------------|--------------|
| [`colab_a_numpy_scratch.ipynb`](./colab_a_numpy_scratch.ipynb) | NumPy + `tf.einsum` from scratch | Manual backprop, chain rule, He init, 4D plot |
| [`colab_b_pytorch_scratch.ipynb`](./colab_b_pytorch_scratch.ipynb) | PyTorch raw tensors, no builtin layers | `requires_grad`, manual SGD, `.grad.zero_()` |
| [`colab_c_pytorch_classes.ipynb`](./colab_c_pytorch_classes.ipynb) | PyTorch `nn.Module` class-based | `nn.Linear`, Adam, `DataLoader`, LR scheduler |
| [`colab_d_pytorch_lightning.ipynb`](./colab_d_pytorch_lightning.ipynb) | PyTorch Lightning | `LightningModule`, `Trainer`, `EarlyStopping` |
| [`colab_e_tensorflow_variants.ipynb`](./colab_e_tensorflow_variants.ipynb) | TensorFlow × 4 variants | `GradientTape`, Functional API, Sequential |

---

## 🎬 Video Walkthroughs

> Each video is a full section-by-section code walkthrough with live output explanation.

---

### Colab A — NumPy from Scratch

[![Watch on YouTube](https://img.shields.io/badge/YouTube-Watch%20Walkthrough-red?style=for-the-badge&logo=youtube)](https://youtu.be/K-5ZSKlDciM)
&nbsp;
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Video%20File-blue?style=for-the-badge&logo=googledrive)](https://drive.google.com/file/d/1hy7FIqOGgm5qvUa4pMlYhyOYYkI05C40/view?usp=sharing)

**What this video covers:**
- Generating 3-variable synthetic data from the nonlinear equation
- 4D scatter plot using PCA to reduce input dimensions + color as 4th axis
- Implementing `relu()` and `relu_derivative()` manually
- He initialization: why `sqrt(2 / fan_in)` for ReLU networks
- Forward pass using `tf.einsum('bi,ij->bj', X, W)` — full notation breakdown
- **Manual backpropagation:** computing `dL/dA → dL/dZ → dL/dW` layer by layer via chain rule
- SGD with momentum parameter update
- Training loop with mini-batches, loss logging, and final metrics

---

### Colab B — PyTorch from Scratch

[![Watch on YouTube](https://img.shields.io/badge/YouTube-Watch%20Walkthrough-red?style=for-the-badge&logo=youtube)](https://youtu.be/R2Rgo6z6EKY)
&nbsp;
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Video%20File-blue?style=for-the-badge&logo=googledrive)](https://drive.google.com/file/d/1bM2-EY5QVoRcNYNiY6VRv-NFTtIt7ScM/view?usp=sharing)

**What this video covers:**
- Creating raw `torch.Tensor` weights with `requires_grad=True` — how autograd graph is built
- Forward pass with `torch.mm()` — graph construction happening silently
- `loss.backward()` — PyTorch walking the graph and filling `.grad` on every tracked tensor
- Why the weight update must be wrapped in `torch.no_grad()`
- Manual `w -= lr * w.grad` update (vanilla SGD)
- **Why `w.grad.zero_()` is critical** — PyTorch accumulates gradients by default
- Comparison of results vs Colab A

---

### Colab C — PyTorch Class-Based

[![Watch on YouTube](https://img.shields.io/badge/YouTube-Watch%20Walkthrough-red?style=for-the-badge&logo=youtube)](https://youtu.be/c4U2zFxWos4)
&nbsp;
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Video%20File-blue?style=for-the-badge&logo=googledrive)](https://drive.google.com/file/d/14d_VU7NT_vn-WLDfGdnD_y3CW-cjP8nN/view?usp=sharing)

**What this video covers:**
- `nn.Module` class pattern — `__init__` defines layers, `forward()` defines data flow
- `nn.Linear` auto-creates weights as `nn.Parameter` objects registered with the module
- `torch.utils.data.DataLoader` replacing manual batch slicing
- **The four-line training pattern:** `optimizer.zero_grad()` → `model(X)` → `loss.backward()` → `optimizer.step()`
- Adam optimizer vs plain SGD — adaptive per-parameter learning rates
- `model.train()` vs `model.eval()` mode switching
- `StepLR` learning rate scheduler — halving LR every 150 epochs

---

### Colab D — PyTorch Lightning

[![Watch on YouTube](https://img.shields.io/badge/YouTube-Watch%20Walkthrough-red?style=for-the-badge&logo=youtube)](https://youtu.be/MERHWZRTpFE)
&nbsp;
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Video%20File-blue?style=for-the-badge&logo=googledrive)](https://drive.google.com/file/d/1xJIqN4mYRP-jxY_yx4ALPKCLMT8mQXjH/view?usp=sharing)

**What this video covers:**
- `LightningDataModule` — separating data loading from model logic
- `LightningModule` combining architecture + training logic + optimizer config in one class
- `training_step()` hook — Lightning calls this per batch, no manual `backward()` needed
- `validation_step()` hook — Lightning handles `eval()` mode and `no_grad()` automatically
- `configure_optimizers()` returning optimizer + scheduler as lists
- `EarlyStopping` callback — stops training when val loss stops improving
- `ModelCheckpoint` callback — auto-saves best weights
- **`trainer.fit(model, datamodule=dm)`** — one line replacing the entire training loop

---

### Colab E — TensorFlow Variants (All 4)

[![Watch on YouTube](https://img.shields.io/badge/YouTube-Watch%20Walkthrough-red?style=for-the-badge&logo=youtube)](https://youtu.be/nrrK8BgdiW4)
&nbsp;
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Video%20File-blue?style=for-the-badge&logo=googledrive)](https://drive.google.com/file/d/1DVYIQHq2I-AHO9hVrD421YrZ3mCDe1HP/view?usp=sharing)

**What this video covers (all 4 variants):**

| Variant | API Level | Key Concept in Video |
|---------|-----------|----------------------|
| **E-i** — TF from scratch | Lowest | `tf.Variable` weights + `tf.GradientTape` context manager + `tape.gradient()` + `apply_gradients()` |
| **E-ii** — TF builtin layers | Low | `Dense` layers + manual `GradientTape` loop + `model.trainable_variables` |
| **E-iii** — Functional API | Medium | `keras.Input` → layer chaining → `keras.Model(inputs, outputs)` → `model.compile` + `model.fit` |
| **E-iv** — Sequential API | Highest | `tf.keras.Sequential([...])` → `compile` → `fit` — full abstraction in 3 lines |

Video ends with a **comparison plot** of all 4 validation loss curves on the same log-scale axis.

---

## 🔗 Open in Google Colab

| Colab | Open in Colab |
|-------|--------------|
| A — NumPy Scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_a_numpy_scratch.ipynb) |
| B — PyTorch Scratch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_b_pytorch_scratch.ipynb) |
| C — PyTorch Classes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_c_pytorch_classes.ipynb) |
| D — PyTorch Lightning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_d_pytorch_lightning.ipynb) |
| E — TensorFlow Variants | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_e_tensorflow_variants.ipynb) |

> **To activate Colab badges:** Replace `YOUR_USERNAME` and `YOUR_REPO` with your GitHub username and repository name. The badges will then open each notebook directly in Colab.

---

## 📊 Colab Summaries

### Colab A — NumPy from Scratch
**Framework:** NumPy + `tf.einsum`  
**Gradient method:** 100% manual chain rule — no autograd of any kind

Every weight update in this colab is the result of manually computing the chain rule backwards through three layers. The core equation is `dL/dW = dL/dA × dA/dZ × dZ/dW`, applied layer by layer from output to input. `tf.einsum('bi,ij->bj')` replaces all matrix multiplications — `b` is batch, `i` is input dimension, `j` is output dimension, and the shared `i` is the summed-over axis.

Key functions: `generate_data()` · `forward_pass()` · `backprop()` · `update_params()` · `train()`

---

### Colab B — PyTorch from Scratch
**Framework:** PyTorch raw tensors  
**Gradient method:** PyTorch autograd via `loss.backward()`, manual weight update

Weights are plain `torch.Tensor` objects with `requires_grad=True`. PyTorch silently builds a computation graph during the forward pass. `loss.backward()` traverses that graph in reverse and deposits gradients into `.grad` on every tracked tensor. Weight update is manual: `w -= lr * w.grad`, wrapped in `torch.no_grad()`. Gradients must be explicitly zeroed with `w.grad.zero_()` before every backward call — PyTorch accumulates by default.

Key pattern: `requires_grad=True` → `forward` → `backward` → `w -= lr * w.grad` → `w.grad.zero_()`

---

### Colab C — PyTorch Class-Based
**Framework:** PyTorch `nn.Module`  
**Gradient method:** Autograd + `optimizer.step()`

Standard idiomatic PyTorch. The network is a class inheriting `nn.Module` with layers defined in `__init__` and data flow defined in `forward()`. `nn.Linear` auto-creates and registers weights. `DataLoader` handles batching. The canonical four-line training pattern — `zero_grad` → forward → `backward` → `step` — replaces all manual update code from Colab B. Adam optimizer with `StepLR` scheduling.

Key pattern: `optimizer.zero_grad()` → `model(X)` → `loss.backward()` → `optimizer.step()`

---

### Colab D — PyTorch Lightning
**Framework:** PyTorch Lightning  
**Gradient method:** Fully automated by `Trainer`

Lightning removes the training loop entirely. The `LightningModule` defines three hooks that Lightning reads: `training_step()` (what to do per batch), `validation_step()` (how to evaluate), and `configure_optimizers()` (what optimizer and scheduler to use). `EarlyStopping` halts training when validation loss plateaus. `ModelCheckpoint` saves best weights automatically. `trainer.fit(model, datamodule=dm)` is the only line that runs training.

Key pattern: implement hooks → `trainer.fit()` → done

---

### Colab E — TensorFlow (4 Variants)

**E-i — TF Scratch:** `tf.Variable` weights + `tf.GradientTape` context manager. `tape.gradient(loss, variables)` returns gradient tensors. `optimizer.apply_gradients(zip(grads, vars))` applies them. Direct TF equivalent of Colab B.

**E-ii — TF Layers + Manual Loop:** `tf.keras.layers.Dense` for layer creation (no manual `tf.Variable`), but still uses a `GradientTape` training loop. `model.trainable_variables` collects all parameters automatically. Hybrid between E-i and E-iii.

**E-iii — Functional API:** Builds a computational graph using symbolic tensors: `keras.Input(shape=(3,))` → Dense layers → `keras.Model(inputs, outputs)`. Supports skip connections and multi-input/output topologies. Uses `model.compile()` and `model.fit()` for training.

**E-iv — Sequential API:** `tf.keras.Sequential([...])` stacks layers linearly. `model.compile(loss='mse', optimizer='adam')` attaches the training configuration. `model.fit(X, y, validation_data=..., epochs=500)` runs everything. Highest abstraction, fewest lines.

---

## 🏗️ Framework Comparison

```
                    Colab A       Colab B       Colab C       Colab D           Colab E
─────────────────────────────────────────────────────────────────────────────────────────
Framework           NumPy+TF      PyTorch       PyTorch       PyTorch+Lightning  TensorFlow
Weight storage      np.ndarray    raw Tensor    nn.Parameter  nn.Parameter       tf.Variable/Dense
Matrix ops          tf.einsum     torch.mm      nn.Linear     nn.Linear          tf.einsum/Dense
Backprop            Manual        autograd      autograd      autograd           GradientTape/auto
Weight update       Manual SGD    Manual SGD    Adam          Adam               Adam
Training loop       Manual        Manual        Manual        trainer.fit()      Manual / fit()
Gradient zeroing    N/A           w.grad.zero_  zero_grad()   Automatic          Automatic
─────────────────────────────────────────────────────────────────────────────────────────
```

---

## 🧩 Key Concepts Index

| Concept | Colab | Exact location |
|---------|-------|---------------|
| Manual chain rule backprop | A | `backprop()` — Sections 7 |
| `tf.einsum('bi,ij->bj')` notation | A, E-i | Forward pass sections |
| He initialization `sqrt(2/fan_in)` | A, B, C | Weight init sections |
| 4D visualization via PCA + color | A | Section 3 |
| `requires_grad=True` | B | Section 3 |
| `loss.backward()` fills `.grad` | B | Section 5 |
| `w.grad.zero_()` — gradient accumulation | B | Section 5 (inner loop) |
| `torch.no_grad()` during update | B | Section 5 |
| `nn.Module` / `forward()` pattern | C | Section 3 |
| `DataLoader` + `TensorDataset` | C | Section 2 |
| Adam optimizer | C, D, E | Optimizer sections |
| `LightningModule` hooks | D | Section 4 |
| `EarlyStopping` + `ModelCheckpoint` | D | Section 5 |
| `trainer.fit()` one-liner | D | Section 5 |
| `tf.GradientTape` context manager | E-i, E-ii | Variant E-i/E-ii cells |
| `tape.gradient()` + `apply_gradients()` | E-i | Variant E-i cell |
| Keras Functional API | E-iii | Variant E-iii cell |
| `keras.Input` + `keras.Model` | E-iii | Variant E-iii cell |
| `model.compile()` + `model.fit()` | E-iii, E-iv | Variants E-iii/E-iv |
| `ReduceLROnPlateau` callback | E-iii, E-iv | Variant callbacks |

---

## 🚀 How to Run

1. Click any **Open in Colab** badge above
2. Select `Runtime → Change runtime type → T4 GPU` (faster, optional)
3. Run all cells: `Runtime → Run all` or `Ctrl+F9`
4. All plots and metrics print inline — no downloads needed

**Dependencies** (pre-installed in Colab except one):
```
numpy · tensorflow · torch · matplotlib · scikit-learn · sklearn
lightning  ← installed automatically in Colab D via !pip install lightning
```

---

## 📈 Expected Results

All variants target the same problem and should achieve similar performance after full training:

| Metric | Expected Range |
|--------|----------------|
| R² Score | 0.90 – 0.99 |
| MAE | 0.05 – 0.20 |
| RMSE | 0.07 – 0.25 |
| Final Val Loss (MSE) | < 0.05 |

---

*Neural Networks Assignment · Spring 2025*

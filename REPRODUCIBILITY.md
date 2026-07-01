# Reproducibility Documentation

This document outlines the details required to replicate the execution environment and achieve deterministic outcomes for this assessment.

---

## 1. Environment Details

* **Python Version**: `3.11.9`
* **Virtual Environment**: `.venv` (created via `python -m venv .venv`)
* **Operating System**: Windows 10/11
* **Key Dependencies (pinned in `requirements.txt`)**:
  - `torch>=2.1.0`
  - `torchvision>=0.16.0`
  - `adversarial-robustness-toolbox>=1.17.0`
  - `mlflow>=2.10.0`

---

## 2. Seed Configuration and Determinism

Reproducibility is enforced at the core of all pipelines using the centralized utility `utils/seed.py` with a fixed seed of `42`.

This utility guarantees consistency across:
* **Standard Random**: `random.seed(seed)`
* **NumPy Operations**: `np.random.seed(seed)` and local generators `np.random.default_rng(seed)` for indices splitting.
* **PyTorch Operations**: `torch.manual_seed(seed)`
* **CUDA Operations**: `torch.cuda.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)` (when GPU is present).
* **CuDNN Deterministic Flags**:
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
* **DataLoader Shuffling**:
  - Shuffling in the training DataLoader is governed by a seeded `torch.Generator` instance passed as `generator=g` to ensure identical batch sequences across runs.

---

## 3. Setup and Verification Instructions

To set up the environment and verify the Data Pipeline correctness:

### Step 1: Create and Activate Virtual Environment
```powershell
# Create environment
python -m venv .venv

# Activate environment
.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run Unit Tests
```powershell
pytest tests/test_mapping.py -v
```

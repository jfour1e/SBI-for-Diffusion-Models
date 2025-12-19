# SBI-for-Diffusion-Models

This repository builds upon existing Drift-Diffusion models, specifically pulse variants. It contains experiments and utilities for **simulation-based inference (SBI)**.  
The project uses **[`uv`](https://github.com/astral-sh/uv)** for fast, reproducible Python environments (shout out Ryan for this one). 
We use SBI version 0.22.0 
---

## Requirements

- Python **>= 3.10**
- `uv` (package & environment manager)

---

## Installing `uv`

### macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create Virtual Environment 
```bash 
uv venv
```

### Install Dependencies 
```bash
uv sync 
```

### Add dependencies 
```bash 
uv add numpy pandas matplotlib torch sbi
```

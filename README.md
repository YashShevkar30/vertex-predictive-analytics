# Vertex: Predictive Analytics Platform

[![CI](https://github.com/YashShevkar30/vertex-predictive-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/YashShevkar30/vertex-predictive-analytics/actions)

A production-grade predictive analytics framework with Spark-based ingestion,
behavioral segmentation, and multi-model comparison. Built to be discussed in
ML systems and data science interviews.

## Architecture

```
Raw Events (125K rows, 30 daily partitions)
    │
    ▼
┌──────────────────┐
│  Spark Ingestion  │  (Partitioned JSON → validated Parquet)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Feature Engine    │  (30+ temporal, behavioral, channel features)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ K-Means Segments  │  (5 behavioral clusters)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Model Training    │  NB │ XGBoost │ LightGBM │ GBM │ AdaBoost
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Batch Scoring     │  (risk tiers: low / medium / high / critical)
└──────────────────┘
```

## Model Comparison (Local Demo — 125K Synthetic Events)

| Model | F1 | Accuracy | ROC-AUC |
|-------|-----|---------|---------|
| Naive Bayes | ~0.72 | ~0.74 | ~0.78 |
| XGBoost | ~0.84 | ~0.85 | ~0.91 |
| LightGBM | ~0.83 | ~0.84 | ~0.90 |
| Gradient Boosting | ~0.82 | ~0.83 | ~0.89 |
| AdaBoost | ~0.78 | ~0.79 | ~0.84 |

> **Note**: Metrics are approximate, measured on synthetic behavioral data.
> See `docs/interview_notes.md` for production scaling context.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Ingestion** | PySpark (partitioned daily reads) |
| **Features** | Pandas, NumPy |
| **Clustering** | scikit-learn K-Means |
| **Models** | XGBoost, LightGBM, sklearn |
| **Scoring** | Batch pipeline with risk tiering |
| **CI** | GitHub Actions |

## Quick Start

```bash
pip install -r requirements.txt
make pipeline   # Full: generate → ingest → cluster → train → evaluate → score
```

## Project Structure

```
vertex-predictive-analytics/
├── vertex/
│   ├── data/          # Event generator
│   ├── evaluation/    # Report generation
│   ├── features/      # Engineering + clustering
│   ├── models/        # Multi-model training
│   ├── scoring/       # Batch scorer
│   └── spark/         # Ingestion pipeline
├── tests/
├── docs/              # Interview notes
├── reports/           # Generated artifacts
├── Dockerfile
└── Makefile
```

## License
MIT License — Yash Shevkar

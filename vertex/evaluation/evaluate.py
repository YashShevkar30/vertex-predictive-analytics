"""
Evaluation Report Generator
=============================
Reads model comparison results and generates a structured
evaluation report with analysis insights.
"""
import json
import pandas as pd
from loguru import logger
from vertex.config import config


def generate_report():
    report_path = config.REPORTS_DIR / "model_comparison.json"
    if not report_path.exists():
        logger.error("Model comparison not found. Run `make train` first.")
        return

    with open(report_path) as f:
        results = json.load(f)

    df = pd.DataFrame(results).T
    df = df.sort_values("f1", ascending=False)

    report = {
        "best_model": df.index[0],
        "best_f1": float(df.iloc[0]["f1"]),
        "comparison_table": df.to_dict(),
        "model_rankings": {
            "by_f1": list(df.sort_values("f1", ascending=False).index),
            "by_accuracy": list(df.sort_values("accuracy", ascending=False).index),
        },
    }

    if "roc_auc" in df.columns:
        report["model_rankings"]["by_auc"] = list(
            df.sort_values("roc_auc", ascending=False).index
        )

    with open(config.REPORTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION REPORT")
    logger.info(f"{'='*60}")
    logger.info(f"\n{df.to_string()}")
    logger.info(f"\nBest model: {report['best_model']} (F1={report['best_f1']})")

    return report


if __name__ == "__main__":
    generate_report()

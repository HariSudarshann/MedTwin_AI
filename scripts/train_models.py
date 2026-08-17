"""
MediTwin AI — Model Training Script
Trains both imaging (ResNet-18) and tabular (RandomForest) models.
"""
import sys, logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_tabular():
    logger.info("=" * 60)
    logger.info("TRAINING TABULAR CARDIAC RISK MODEL")
    logger.info("Dataset: UCI Heart Disease (Cleveland) — 303 samples")
    logger.info("=" * 60)
    from models.tabular_model import train_tabular_model
    metrics = train_tabular_model(
        data_dir=ROOT / "data",
        save_path=ROOT / "models" / "heart_risk_rf.joblib"
    )
    logger.info(f"Tabular model metrics: {metrics}")
    return metrics


def train_imaging(data_dir: Path):
    logger.info("=" * 60)
    logger.info("TRAINING IMAGING MODEL (ResNet-18)")
    logger.info(f"Data: {data_dir}")
    logger.info("=" * 60)
    from models.imaging_model import train_model
    metrics = train_model(
        data_dir=data_dir,
        save_path=ROOT / "models" / "resnet18_xray.pt",
        epochs=10,
        lr=1e-4,
        batch_size=16
    )
    logger.info(f"Imaging model metrics: {metrics}")
    return metrics


def write_results(tab_metrics, img_metrics=None):
    results_md = ROOT / "RESULTS.md"
    lines = [
        "# MediTwin AI — Review 3 Results\n",
        "## 1. Tabular Risk Model — UCI Heart Disease (Cleveland)\n",
        f"- **Dataset**: UCI Heart Disease (Cleveland), 303 samples, 13 features\n",
        f"- **Target**: Binary cardiac risk (0 = no disease, 1 = disease)\n",
        f"- **Model**: RandomForest + GradientBoosting soft-voting ensemble\n",
        f"- **Accuracy** : {tab_metrics.get('accuracy', 'N/A')}\n",
        f"- **Precision**: {tab_metrics.get('precision', 'N/A')}\n",
        f"- **Recall**   : {tab_metrics.get('recall', 'N/A')}\n",
        f"- **F1 Score** : {tab_metrics.get('f1', 'N/A')}\n",
        f"- **ROC-AUC**  : {tab_metrics.get('roc_auc', 'N/A')}\n",
        f"- **5-Fold CV ROC-AUC**: {tab_metrics.get('cv_roc_auc', 'N/A')}\n",
        f"\n```\n{tab_metrics.get('classification_report', '')}\n```\n",
    ]
    if img_metrics:
        lines += [
            "\n## 2. Imaging Model — Chest X-Ray (ResNet-18)\n",
            f"- **Dataset**: Kaggle Chest X-Ray Pneumonia\n",
            f"- **Architecture**: ResNet-18 (transfer learning, ImageNet pretrained)\n",
            f"- **Accuracy** : {img_metrics.get('test_accuracy', 'N/A')}\n",
            f"- **Precision**: {img_metrics.get('precision', 'N/A')}\n",
            f"- **Recall**   : {img_metrics.get('recall', 'N/A')}\n",
            f"- **F1 Score** : {img_metrics.get('f1', 'N/A')}\n",
            f"- **ROC-AUC**  : {img_metrics.get('roc_auc', 'N/A')}\n",
            f"\n```\n{img_metrics.get('classification_report', '')}\n```\n",
        ]
    else:
        lines += [
            "\n## 2. Imaging Model — Chest X-Ray\n",
            "> ⚠ Kaggle dataset not yet downloaded. To train:\n",
            "> 1. Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia\n",
            "> 2. Extract to: `data/chest_xray/` (folders: train/val/test with NORMAL/PNEUMONIA subfolders)\n",
            "> 3. Re-run: `python scripts/train_models.py --imaging`\n",
        ]
    with open(results_md, "w") as f:
        f.writelines(lines)
    logger.info(f"RESULTS.md written to {results_md}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--imaging", action="store_true", help="Also train imaging model")
    parser.add_argument("--xray-data", type=Path, default=ROOT / "data" / "chest_xray",
                        help="Path to chest_xray directory with train/val/test splits")
    args = parser.parse_args()

    tab_metrics = train_tabular()
    img_metrics = None

    if args.imaging:
        xray_dir = args.xray_data
        if not xray_dir.exists():
            logger.error(f"X-ray data directory not found: {xray_dir}")
            logger.error("Download dataset from Kaggle and extract to data/chest_xray/")
        else:
            img_metrics = train_imaging(xray_dir)

    write_results(tab_metrics, img_metrics)
    logger.info("Done! Check RESULTS.md for metrics summary.")

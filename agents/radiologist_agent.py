"""
MediTwin AI — Radiologist Agent

Primary model:
    TorchXRayVision DenseNet-121

Fallback model:
    Existing ResNet-18 chest X-ray classifier

The agent converts model outputs into a structured radiology result
for the multi-agent clinical reasoning pipeline.
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TorchXRayVision inference
# ─────────────────────────────────────────────────────────────────────────────
def _run_torchxrayvision(image) -> Dict[str, Any]:
    """
    Run TorchXRayVision DenseNet-121 on a chest X-ray.

    Returns all pathology scores.
    """
    from models.torchxray_model import TorchXRayVisionClassifier

    classifier = TorchXRayVisionClassifier()
    result = classifier.predict(image)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Existing ResNet-18 fallback
# ─────────────────────────────────────────────────────────────────────────────
def _run_resnet18(image) -> Dict[str, Any]:
    """
    Run the existing ResNet-18 model as a fallback.
    """
    model_path = Path("models/resnet18_xray.pt")

    if not model_path.exists():
        raise FileNotFoundError(
            "Existing ResNet-18 model not found: "
            "models/resnet18_xray.pt"
        )

    from models.imaging_model import XRayClassifier

    classifier = XRayClassifier(model_path)
    return classifier.predict(image)


# ─────────────────────────────────────────────────────────────────────────────
# Radiologist Agent
# ─────────────────────────────────────────────────────────────────────────────
def radiologist_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the available chest X-ray and produce a structured
    Radiologist Agent result.

    Preferred model:
        TorchXRayVision DenseNet-121

    Fallback:
        Existing ResNet-18 classifier
    """

    errors = state.get("errors", [])

    image_bytes = state.get("xray_image_bytes")
    image_path = state.get("xray_image_path")

    # ------------------------------------------------------------------
    # No image provided
    # ------------------------------------------------------------------
    if not image_bytes and not image_path:
        result = {
            "status": "not_provided",
            "agent": "Radiologist Agent",
            "modality": "Chest X-ray",
            "finding": "N/A",
            "prediction": "N/A",
            "confidence": 0.0,
            "findings": {},
            "note": "No chest X-ray image was provided.",
        }

        state["radiologist_result"] = result

        # Keep old UI/report compatibility
        state["imaging_result"] = result

        return state

    try:
        from PIL import Image

        # ------------------------------------------------------------------
        # Load image
        # ------------------------------------------------------------------
        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            path = Path(image_path)

            if not path.exists():
                raise FileNotFoundError(
                    f"Chest X-ray image not found: {path}"
                )

            image = Image.open(path).convert("RGB")

        # ------------------------------------------------------------------
        # Preferred model: TorchXRayVision
        # ------------------------------------------------------------------
        try:
            model_result = _run_torchxrayvision(image)

            findings = model_result.get("findings", {})

            # Prototype display threshold.
            # This is NOT a clinically validated diagnostic threshold.
            threshold = 0.50

            positive_findings = {
                name: score
                for name, score in findings.items()
                if float(score) >= threshold
            }

            # Sort findings by score
            positive_findings = dict(
                sorted(
                    positive_findings.items(),
                    key=lambda item: float(item[1]),
                    reverse=True,
                )
            )

            # Top finding for compatibility / summary
            if positive_findings:
                top_finding = next(iter(positive_findings))
                top_score = float(positive_findings[top_finding])
            elif findings:
                top_finding = max(
                    findings,
                    key=lambda name: float(findings[name])
                )
                top_score = float(findings[top_finding])
            else:
                top_finding = "No significant finding"
                top_score = 0.0

            result = {
                "status": "success",
                "agent": "Radiologist Agent",
                "modality": "Chest X-ray",

                # Main finding
                "finding": top_finding,
                "prediction": top_finding,
                "confidence": round(top_score, 4),

                # Multi-pathology results
                "findings": {
                    k: round(float(v), 4)
                    for k, v in findings.items()
                },

                "positive_findings": positive_findings,

                # Model information
                "model": model_result.get(
                    "model",
                    "TorchXRayVision DenseNet-121"
                ),
                "weights": model_result.get(
                    "weights",
                    "densenet121-res224-all"
                ),
                "device": model_result.get("device", "cpu"),

                # Explainability placeholder
                "xai": {
                    "method": "Grad-CAM",
                    "available": False,
                    "note": (
                        "TorchXRayVision multi-pathology inference is "
                        "currently enabled. Image explanation will be "
                        "added as a separate module."
                    ),
                },

                "interpretation_type": (
                    "AI-assisted chest X-ray finding"
                ),

                "clinical_note": (
                    "The result is an AI-assisted imaging analysis. "
                    "Pathology scores are model outputs and require "
                    "clinical correlation and professional review. "
                    "They must not be interpreted as an autonomous diagnosis."
                ),
            }

            state["radiologist_result"] = result

            # Keep current UI/report compatibility
            state["imaging_result"] = result

            logger.info(
                "[RadiologistAgent] TorchXRayVision | "
                "Top finding=%s | score=%.4f",
                top_finding,
                top_score,
            )

            return state

        except Exception as xrv_error:
            # ----------------------------------------------------------
            # TorchXRayVision failed → use ResNet-18 fallback
            # ----------------------------------------------------------
            logger.warning(
                "TorchXRayVision failed: %s. "
                "Falling back to ResNet-18.",
                xrv_error,
            )

            fallback_result = _run_resnet18(image)

            prediction = fallback_result.get(
                "prediction",
                "N/A"
            )

            confidence = float(
                fallback_result.get(
                    "confidence",
                    0.0
                )
            )

            result = {
                "status": "success_fallback",
                "agent": "Radiologist Agent",
                "modality": "Chest X-ray",

                "finding": prediction,
                "prediction": prediction,
                "confidence": confidence,

                "findings": {
                    prediction: round(confidence, 4)
                },

                "positive_findings": {
                    prediction: round(confidence, 4)
                },

                "probabilities": fallback_result.get(
                    "probabilities",
                    {}
                ),

                "model": "ResNet-18",
                "weights": "Project-trained checkpoint",

                "gradcam_b64": fallback_result.get(
                    "gradcam_b64"
                ),
                "cam_array": fallback_result.get(
                    "cam_array"
                ),

                "xai": {
                    "method": "Grad-CAM",
                    "available": bool(
                        fallback_result.get(
                            "gradcam_b64"
                        )
                    ),
                },

                "interpretation_type": (
                    "AI-assisted chest X-ray finding"
                ),

                "clinical_note": (
                    "TorchXRayVision inference was unavailable, "
                    "so the existing ResNet-18 model was used as "
                    "a fallback. Clinical correlation is required."
                ),
            }

            state["radiologist_result"] = result
            state["imaging_result"] = result

            return state

    except Exception as exc:
        error_message = f"RadiologistAgent error: {exc}"

        logger.exception(error_message)
        errors.append(error_message)

        result = {
            "status": "error",
            "agent": "Radiologist Agent",
            "modality": "Chest X-ray",
            "prediction": "ERROR",
            "finding": "ERROR",
            "confidence": 0.0,
            "findings": {},
            "positive_findings": {},
            "error": str(exc),
        }

        state["radiologist_result"] = result
        state["imaging_result"] = result
        state["errors"] = errors

    return state
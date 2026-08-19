from pathlib import Path

from agents.radiologist_agent import radiologist_agent


def main():
    image_path = Path("data/xray/images.jpg")

    if not image_path.exists():
        raise FileNotFoundError(
            f"X-ray not found: {image_path}"
        )

    state = {
        "patient_id": "TEST-001",
        "xray_image_path": str(image_path),
        "errors": [],
    }

    # Agent returns the complete updated state
    final_state = radiologist_agent(state)

    # Extract the actual agent result
    result = final_state.get("radiologist_result", {})

    print("\n=== RADIOLOGIST AGENT RESULT ===")
    print("Status:", result.get("status"))
    print("Model:", result.get("model"))
    print("Top finding:", result.get("finding"))
    print("Confidence:", result.get("confidence"))

    print("\nPositive findings:")
    for name, score in result.get("positive_findings", {}).items():
        print(f"  {name}: {score:.4f}")

    print("\nAll findings:")
    for name, score in result.get("findings", {}).items():
        print(f"  {name}: {score:.4f}")

    print("\nErrors:", final_state.get("errors", []))


if __name__ == "__main__":
    main()
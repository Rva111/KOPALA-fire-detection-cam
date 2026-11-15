#!/usr/bin/env python3
"""YOLO fire detection for a local webcam or video.

Install:
    python -m pip install -U ultralytics opencv-python huggingface_hub

Run:
    python fire_detector.py

Set SOURCE below to 0 for a webcam, 1 for another webcam, or a video filename.
Click the preview window and press Q to quit.

Model: https://huggingface.co/rabahdev/fire-smoke-yolov8n
Model license listed by its author: AGPL-3.0.
This demonstration has not been validated as a fire alarm.
"""

import cv2
import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

SOURCE = 0                  # Or "fire_video.mp4"
CONFIDENCE = 0.40            # Starting threshold; tune on your own footage
IMAGE_SIZE = 640


def main():
    # Download the fire-trained weights; later runs reuse the cached file.
    weights = hf_hub_download(
        repo_id="rabahdev/fire-smoke-yolov8n",
        filename="best.pt",
        revision="9841e3bc9373116d1a756969aa5be3f016ad3782",
    )
    model = YOLO(weights)

    # Get fire class IDs from the model instead of assuming a class number.
    fire_ids = [
        class_id
        for class_id, label in model.names.items()
        if str(label).strip().lower() in {"fire", "flame", "flames"}
    ]
    if not fire_ids:
        raise RuntimeError(f"No fire class in this model: {model.names}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model classes: {model.names}")
    print("Click the video window and press Q to stop.")

    cap = cv2.VideoCapture(SOURCE)
    try:
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open {SOURCE!r}. Check the video path or camera "
                "permissions, or try SOURCE = 1."
            )

        was_fire = False
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Video ended or the next frame could not be read.")
                break

            result = model.predict(
                source=frame,
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                classes=fire_ids,
                device=device,
                verbose=False,
            )[0]

            fire_detected = result.boxes is not None and len(result.boxes) > 0
            output = result.plot()  # Draw boxes, labels, and confidence scores.

            if fire_detected and not was_fire:
                print("FIRE DETECTED")
            was_fire = fire_detected

            status = "FIRE DETECTED" if fire_detected else "No fire detected"
            color = (0, 0, 255) if fire_detected else (255, 255, 255)
            cv2.putText(
                output, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2, cv2.LINE_AA,
            )
            cv2.imshow("YOLO - Fire Detection", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty("YOLO - Fire Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

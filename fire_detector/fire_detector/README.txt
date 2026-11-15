YOLO FIRE DETECTOR
==================

CONTENTS
- fire_detector.py: webcam/video fire detection script
- requirements.txt: Python dependencies
- README.txt: these instructions

QUICK START
1. Extract the ZIP.
2. Open the extracted fire_detector folder in a local desktop terminal
   (for example, the VS Code terminal).
3. Install the dependencies:

   python -m pip install -r requirements.txt

4. Run the script:

   python fire_detector.py

The default input is webcam 0. Click the preview window and press Q to stop.
The ZIP contains the code and setup files. The pretrained model weights are
not bundled: the script downloads them on its first run and caches them for
later use. Installation and the first model download need internet access.
Inference runs locally on your computer.

SETTINGS
Edit the constants near the top of fire_detector.py:

SOURCE = 0                    Default webcam
SOURCE = 1                    Another connected camera
SOURCE = "fire_video.mp4"      A video file (use its full path if needed)
CONFIDENCE = 0.40              Starting confidence threshold
IMAGE_SIZE = 640              YOLO inference image size

A lower confidence threshold includes more lower-scored detections.
A higher threshold filters out more detections and may miss real fire.
Tune the threshold on your own footage; 0.40 is a starting point.

The script automatically selects an NVIDIA GPU when PyTorch reports that
CUDA is available. Otherwise it uses the CPU.

TROUBLESHOOTING
- If the camera will not open, check permissions, close other camera apps,
  or try SOURCE = 1.
- For a video, make sure SOURCE points to an existing readable video file.
- Run this on a local computer with a desktop display. The preview window
  and webcam capture are not designed for a remote, headless notebook.

MODEL SOURCE
https://huggingface.co/rabahdev/fire-smoke-yolov8n

The model is YOLOv8n fine-tuned for fire and smoke. The script filters the
results to fire detections. The model author lists the weights as AGPL-3.0.
The script pins the downloaded model to a specific repository revision.

VALIDATION
The Python syntax and ZIP integrity were checked. Live inference was not
run in the preparation environment. This demonstration has not been
validated as a fire alarm.

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "./models/hand_landmarker.task"

capture = cv2.VideoCapture(0)


def print_result(
    result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    print(f"landmarker result: {result}")


options = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result,
)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, img = capture.read()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        landmarker.detect_async(mp_image, 1)

        cv2.imshow("Landmarker", img)
        cv2.waitKey(1)

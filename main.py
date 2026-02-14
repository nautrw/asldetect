import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils

model_path = "./models/hand_landmarker.task"

capture = cv2.VideoCapture(0)


# show_result is only called for capture_live_steam()
def show_result(
    result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    print(result)


def capture_live_steam():
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=show_result,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        frame = 1

        while True:
            success, img = capture.read()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            landmarker.detect_async(mp_image, frame)

            frame += 1

            cv2.imshow("Landmarker", img)
            cv2.waitKey(1)


def capture_image():
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        success, img = capture.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            print(get_landmarks_from_data(result))
            drawing_utils.draw_landmarks(img, result.hand_landmarks[0])

        cv2.imshow("Landmarker", img)
        cv2.waitKey(100000000)  # Prevents the window from closing immediately


def get_landmarks_from_data(data: vision.HandLandmarkerResult):
    return [(landmark.x, landmark.y, landmark.z) for landmark in data.hand_landmarks[0]]


capture_image()

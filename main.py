import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils

model_path = "./models/hand_landmarker.task"

capture = cv2.VideoCapture(0)


def show_result(
    result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int = 0
):
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            print(get_clean_landmarks(hand))
            drawing_utils.draw_landmarks(
                output_image, hand, vision.HandLandmarksConnections.HAND_CONNECTIONS
            )


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

            # show_result is called automatically for capture_live_steam

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

        show_result(result, img)

        cv2.imshow("Landmarker", img)
        cv2.waitKey(100000000)  # Prevents the window from closing immediately


def get_clean_landmarks(hand):
    return [(landmark.x, landmark.y, landmark.z) for landmark in hand]


capture_image()

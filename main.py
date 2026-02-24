import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
import os
import random
model_path = "./models/hand_landmarker.task"

capture = cv2.VideoCapture(0)

if not os.path.exists("./imgdata"):
    os.makedirs("imgdata")

def show_live_stream_result(
    result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int = 0
):
    output_image = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_BGR2RGB)
    annotated_image = output_image.copy()

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            print(get_clean_landmarks(hand))
            drawing_utils.draw_landmarks(
                annotated_image,
                hand,
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
            )
  
    
    cv2.imshow("Landmarker", annotated_image)

    keypress = cv2.waitKey(1) & 0xFF

    if keypress == ord('q'):
        cv2.destroyAllWindows()
        os._exit(1)
    elif keypress == ord('c'):
        if not os.path.exists("unprocessed_data"):
            os.makedirs("unprocessed_data")
       
        filename = ''.join([str(random.randint(0, 1000000)) for _ in range(10)]) # random placeholder for image file
        cv2.imwrite(os.path.join("./unprocessed_data", f"{''.join(filename)}.jpg"), output_image)

def capture_live_stream():
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=show_live_stream_result,
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

            # cv2.imshow("Landmarker", img)
            # cv2.waitKey(1)


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
            for hand in result.hand_landmarks:
                print(get_clean_landmarks(hand))
                drawing_utils.draw_landmarks(
                    img,
                    hand,
                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                )

        cv2.imshow("Landmarker", img)
        cv2.waitKey(0)


def get_clean_landmarks(hand):
    return [(landmark.x, landmark.y, landmark.z) for landmark in hand]


capture_live_stream()

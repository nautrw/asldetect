import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
import os
import random
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

model_path = "./models/hand_landmarker.task"

capture = cv2.VideoCapture(0)

if not os.path.exists("./imgdata"):
    os.makedirs("imgdata")
    
if not os.path.exists("./data.json"):
    with open("data.json", "x+") as f:
        f.write('{"data":[],"characters":[]}')

def show_live_stream_result(
    result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int = 0
):
    output_image = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_BGR2RGB)
    annotated_image = output_image.copy()

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            # print(get_clean_landmarks(hand))
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
    elif keypress == ord('s'):
        char = input('Character to save data for (`pass` to not save anything): ')

        if char == "pass":
            print("Passing; no data will be saved.")
        else:
            with open('data.json', 'r+') as f:
                data = json.load(f)

                data["data"].append(get_clean_landmarks(result.hand_landmarks[0]))
                data["characters"].append(char)

                f.seek(0)
                json.dump(data, f)

            print(f"Successfully saved data for character {char}")
    elif keypress == ord('p'):
        with open("model.pickle", "rb") as f:
            model = pickle.load(f)

        prediction = model.predict([np.asarray(get_clean_landmarks(result.hand_landmarks[0]))])
        print(prediction)

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
    res = []
    
    for landmark in hand:
        res.append(landmark.x)
        res.append(landmark.y)
        
    return res

def train_model():
    with open("data.json", "r") as f:
        data = json.load(f)
        
    cords = np.asarray(data["data"]) 
    chars = np.asarray(data["characters"])
    
    x_train, x_test, y_train, y_test = train_test_split(cords, chars, test_size=0.2, shuffle=True, stratify=chars)
    
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    
    predict_y = model.predict(x_test)
    accuracy = accuracy_score(predict_y, y_test)
    
    print(f"Trained with {accuracy * 100}% accuracy")
    
    with open("model.pickle", "wb") as f:
        pickle.dump(model, f)

capture_live_stream()
# train_model()

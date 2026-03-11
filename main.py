import json
import os
import pickle
import random

import cv2
import mediapipe as mp
import numpy as np
from google import genai
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

model_path = "./models/hand_landmarker.task"

sentence_raw = []

def show_live_stream_result(
    result, output_image, timestamp_ms = 0
):
    output_image = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_BGR2RGB)
    annotated_image = output_image.copy()

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            drawing_utils.draw_landmarks(
                annotated_image,
                hand,
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
            )

    cv2.imshow("Landmarker", annotated_image)

    keypress = cv2.waitKey(1) & 0xFF

    if keypress == ord("q"):
        cv2.destroyAllWindows()
        os._exit(1)
    elif keypress == ord("c"):
        if not os.path.exists("unprocessed_data"):
            os.makedirs("unprocessed_data")

        filename = "".join(
            [str(random.randint(0, 1000000)) for _ in range(10)]
        )
        cv2.imwrite(
            os.path.join("./unprocessed_data", f"{''.join(filename)}.jpg"), output_image
        )
    elif keypress == ord("s"):
        char = input("Character to save data for (`pass` to not save anything): ")

        if char == "pass":
            print("Passing; no data will be saved.")
        else:
            if not os.path.exists("./data.json"):
                with open("data.json", "x+") as f:
                    f.write('{"data":[],"characters":[]}')

            with open("data.json", "r+") as f:
                data = json.load(f)

                data["data"].append(get_clean_landmarks(result.hand_landmarks[0]))
                data["characters"].append(char)

                f.seek(0)
                json.dump(data, f)

            print(f"Successfully saved data for character `{char}`.")
    elif keypress == ord("p"):
        with open("model.pickle", "rb") as f:
            model = pickle.load(f)

        prediction = str(
            model.predict([np.asarray(get_clean_landmarks(result.hand_landmarks[0]))])[
                0
            ]
        )

        ask_append = input(
            f"Append character `{prediction}` to raw sentence array? [y/n]:  "
        )

        if ask_append == "y":
            sentence_raw.append(prediction)
            print(
                f"Appended character `{prediction}` to raw sentence array; array is now `{sentence_raw}`"
            )
        else:
            print("Appending nothing.")
    elif keypress == ord("o"):
        sentence_string = "".join(sentence_raw)

        print("Starting Gemini client")
        gemini_client = genai.Client()
        print("Started Gemini client")

        print(f"Sending prompt for `{sentence_string}`")

        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"Turn the following string of letters into proper english. Fill in any missing characters, and complete names if neccessary. Try not to rearrange letters as much as possible. Return ONLY the result.\n\n{sentence_string}",
        )

        print(f"Response: `{response.text}`")


def capture_live_stream():
    video_capture = cv2.VideoCapture(0)

    landmarker_options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=show_live_stream_result,
    )

    with vision.HandLandmarker.create_from_options(landmarker_options) as landmarker:
        frame = 1

        while True:
            _, img = video_capture.read()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            landmarker.detect_async(mp_image, frame) # Will call show_livestream_result, as seen in landmarker_options

            frame += 1


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

    x_train, x_test, y_train, y_test = train_test_split(
        cords, chars, test_size=0.2, shuffle=True, stratify=chars
    )

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    predict_y = model.predict(x_test)
    accuracy = accuracy_score(predict_y, y_test)

    print(f"Trained with {accuracy * 100}% accuracy")

    with open("model.pickle", "wb") as f:
        pickle.dump(model, f)


def main():
    choice = input(
        "1. Capture live stream\n2. Train model\n3. See training data statistics\nSelect an option: "
    )

    match choice:
        case "1":
            capture_live_stream()
        case "2":
            train_model()
        case "3":
            with open("data.json", "r") as f:
                data = json.load(f)["characters"]

                raw_counts = dict((key, data.count(key)) for key in set(data))

                for key, value in raw_counts.items():
                    print(f"`{key}``: {value}")

                print(f"Total of {len(data)} characters.")
        case _:
            exit("Exiting.")


main()

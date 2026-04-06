import json
import os
import pickle
from datetime import timedelta
from timeit import default_timer as timer

import cv2
import mediapipe as mp
import numpy as np
import pydot
from dotenv import dotenv_values, load_dotenv
from google import genai
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

model_path = "./models/hand_landmarker.task"

if load_dotenv():
    print("Loaded .env")
else:
    print("Failed to load .env")

char = input("Character to save data for (`pass` to not save anything): ")
print(
    "Not saving character data"
    if char.lower() == "pass"
    else f"Will save data for character {char}"
)
ask_append = input(f"Append characters to raw sentence array? [y/n]:  ")
print(
    f"Will{'' if ask_append.lower() == 'y' else ' not'} append detected characters to raw sentence array"
)
raw_sentence = []  # Must start this here even if it won't be used
new_data = {"data": [], "characters": []}

print("Starting Gemini client")
gemini_client = genai.Client(api_key=dotenv_values()["GEMINI_API_KEY"])
print("Started Gemini client")


def show_live_stream_result(result, output_image, timestamp_ms=0):
    output_image = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_BGR2RGB)
    annotated_image = output_image.copy()

    if landmarks_list := result.hand_landmarks:
        for hand in landmarks_list:
            drawing_utils.draw_landmarks(
                annotated_image,
                hand,
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
            )

    cv2.imshow("Landmarker", annotated_image)

    keypress = cv2.waitKey(1) & 0xFF

    if keypress == ord("q"):
        cv2.destroyAllWindows()

        if len(new_data["data"]) == 0:
            exit("Saving no new data")
        else:
            with open("data.json", "r+") as f:
                data = json.load(f)
                data["data"].extend(new_data["data"])
                data["characters"].extend(new_data["characters"])
                f.seek(0)
                json.dump(data, f)
            print("Saved data")

        os._exit(1)
    elif keypress == ord("s"):
        if not char == "pass":
            start_time = timer()

            if not os.path.exists("./data.json"):
                with open("data.json", "x+") as f:
                    f.write('{"data":[],"characters":[]}')

            new_data["data"].append(get_clean_landmarks(result.hand_landmarks[0]))
            new_data["characters"].append(char)

            end_time = timer()

            print(
                f"Successfully saved data for character `{char}`; {len(new_data['characters'])} characters saved this session (Took {timedelta(seconds=end_time-start_time)})"
            )
        else:
            print("Saving nothing.")
    elif keypress == ord("p"):
        with open("model.pickle", "rb") as f:
            model = pickle.load(f)

        prediction = str(
            model.predict([np.asarray(get_clean_landmarks(result.hand_landmarks[0]))])[
                0
            ]
        )

        print(f"Predicted: `{prediction}`")
        if ask_append == "y":
            raw_sentence.append(prediction)
            print(
                f"Appended character `{prediction}` to raw sentence array; array is now `{raw_sentence}`"
            )
        else:
            print("Appending nothing.")

    elif keypress == ord("o"):
        start_time = timer()

        sentence_string = "".join(raw_sentence)

        print(f"Sending prompt for `{sentence_string}`")

        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"Turn the following string of letters into proper english. Fill in any missing characters, and complete names if neccessary. Try not to rearrange letters as much as possible. Return ONLY the result.\n\n{sentence_string}",
        )

        end_time = timer()

        print(
            f"Response: `{response.text}` (Took {timedelta(seconds=end_time-start_time)})"
        )


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

            # Will call show_livestream_result, as seen in landmarker_options
            landmarker.detect_async(mp_image, frame)

            frame += 1


def get_clean_landmarks(hand):
    res = []

    for landmark in hand:
        res.append(landmark.x)
        res.append(landmark.y)

    return res


def train_model():
    print("Starting training")
    start_time = timer()

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
    print("Finished training; running accuracy test")
    accuracy = accuracy_score(predict_y, y_test)
    end_time = timer()

    print(
        f"Trained with {accuracy * 100:,.3f}% accuracy (Took {timedelta(seconds=end_time-start_time)})"
    )

    with open("model.pickle", "wb") as f:
        pickle.dump(model, f)


def model_png(model, filename):
    print(f"Number of estimators: {len(model.estimators_)}")
    tree = model.estimators_[0]
    print(f"Representation of first decision tree: {tree}")

    export_graphviz(
        tree, out_file=f"{filename}.dot", rounded=True, filled=True, precision=100
    )
    print("Exported graph visualization to .dot file")

    graph = pydot.graph_from_dot_file(f"{filename}.dot")
    graph[0].write_png(f"{filename}.png")
    print("Exported visualization as .png file")


def main():
    choice = input(
        "1. Capture live stream\n2. Train model\n3. See training data statistics\n4. Visualize model\nSelect an option: "
    )

    match choice:
        case "1":
            capture_live_stream()
        case "2":
            train_model()
        case "3":
            with open("data.json", "r") as f:
                data = json.load(f)["characters"]

                # TODO: Make sure that this is sorted alphabetically
                raw_counts = dict((key, data.count(key)) for key in set(data))

                for key, value in raw_counts.items():
                    print(f"`{key}``: {value}")

                print(f"Total of {len(data)} characters.")
        case "4":
            with open("model.pickle", "rb") as f:
                model = pickle.load(f)

            model_png(model, "model")
        case _:
            exit("Exiting.")


main()

if len(new_data["data"]) == 0:
    exit("Saving no new data")
else:
    with open("data.json", "r+") as f:
        data = json.load(f)
        data["data"].extend(new_data["data"])
        data["characters"].extend(new_data["characters"])
        f.seek(0)
        json.dump(data, f)
    print("Saved data")

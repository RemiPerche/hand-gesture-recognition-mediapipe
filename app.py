#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import os
import time
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

# Gesture action safety
#
# The classifier still predicts a gesture on every frame.
# We only sample one prediction every GESTURE_SAMPLE_INTERVAL seconds
# so that the history represents the movement over time instead of
# hundreds of frames captured in a fraction of a second.
GESTURE_SAMPLE_INTERVAL = 0.15       # 150 ms between samples
GESTURE_HISTORY_LENGTH = 5           # Number of samples used for validation
GESTURE_MIN_OCCURRENCES = 3          # Minimum occurrences in the history
GESTURE_CONFIRMATION_DELAY = 0.30    # Candidate must remain stable for 300 ms
GESTURE_COOLDOWN = 1.0               # Delay after an action before another one

POINT_HISTORY_LENGTH = 16

# VIDAA configuration
# Replace these values with your TV network information.
VIDAA_TV_IP = "192.168.1.XXX"
VIDAA_TV_MAC = "XX:XX:XX:XX:XX"

class HandState:
    def __init__(self):
        # History used to validate the detected gesture
        self.gesture_history = deque(maxlen=GESTURE_HISTORY_LENGTH)

        # History used by the finger gesture classifier
        self.finger_gesture_history = deque(maxlen=POINT_HISTORY_LENGTH)

        # Point trajectory history
        self.point_history = deque(maxlen=POINT_HISTORY_LENGTH)

        # Gesture sampling
        self.last_sample_time = 0.0

        # Gesture confirmation
        self.candidate_gesture = None
        self.candidate_since = 0.0

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument(
        "--model",
        help="path to the MediaPipe Hand Landmarker .task model",
        type=str,
        default="model/hand_landmarker.task",
    )

    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)

    args = parser.parse_args()

    return args


def execute_vidaa_action(action):
    if action == "Neutral":
        return True
    ACTION_TO_COMMAND = {
        "Power": tv.power,
        "Ok": tv.ok,
        "Left": tv.left,
        "Right": tv.right,
        "Up": tv.up,
        "Down": tv.down,
    }
    try:
        command = ACTION_TO_COMMAND.get(action)
        if command is None:
            print(f"Action inconnue : {action}")
            return False
        return bool(command())
    except FileNotFoundError:
        print("Erreur VIDAA : la commande 'tv' est introuvable.")
        return False
    except Exception as exc:
        print(f"Erreur VIDAA ({action}) : {exc}")
        return False

def translate_gesture_to_action(hand_label, hand_gesture, finger_gesture):
    RIGHT_HAND_GESTURE_TO_ACTION = {
        "Open" : "Power",
        "Close" : "Neutral",
        "Pointer" : "Selection",
        "Ok" : "Ok",
        "Stop" : "Neutral",
        "Left" : "Left",
        "Right" : "Right",
        "Up" : "Up",
        "Down" : "Down",
    }
    LEFT_HAND_GESTURE_TO_ACTION = {
        "Open" : "Power",
        "Close" : "Neutral",
        "Pointer" : "Selection",
        "Ok" : "Ok",
    }
    if hand_label == "right":
        action_detected = RIGHT_HAND_GESTURE_TO_ACTION.get(hand_gesture, "Neutral")
        if action_detected == "Selection":
            return RIGHT_HAND_GESTURE_TO_ACTION.get(finger_gesture, "Neutral")
    else:
        action_detected = LEFT_HAND_GESTURE_TO_ACTION.get(hand_gesture, "Neutral")
    return action_detected


def main():
    # Argument parsing
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # MediaPipe Hand Landmarker initialization
    # MediaPipe 1.x uses the Tasks API instead of the old mp.solutions API.
    model_path = args.model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Hand Landmarker model not found: {model_path}\n"
            "Download the official MediaPipe Hand Landmarker .task model "
            "and place it at this path, or use --model <path>."
        )

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    landmarker_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read keypoint labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Hands declarations
    hand_states = {
        "left": HandState(),
        "right": HandState(),
    }

    last_action_time = 0.0

    # Mode used (0 = normal, 1 = keypoint log, 2 = point history log)
    mode = 0

    # VIDEO mode is synchronous and uses MediaPipe tracking between frames.
    timestamp_ms = 0

    with HandLandmarker.create_from_options(landmarker_options) as hands:
        while True:
            fps = cvFpsCalc.get()

            # Process Key (ESC: end)
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = select_mode(key, mode)

            # Camera capture 
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # MediaPipe Tasks expects RGB input.
            rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_image,
            )

            # VIDEO mode requires strictly increasing timestamps.
            timestamp_ms = max(
                timestamp_ms + 1,
                int(time.monotonic() * 1000),
            )
            current_time = time.monotonic()
            results = hands.detect_for_video(mp_image, timestamp_ms)

            if results.hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.hand_landmarks,
                    results.handedness,
                ):
                    # Get the current hand
                    hand_label = get_handedness_label(handedness)
                    hand_state = hand_states.get(hand_label)

                    if hand_state is None:
                        continue
                    
                    brect = calc_bounding_rect(
                        debug_image, hand_landmarks
                    )

                    # Processing
                    landmark_list = calc_landmark_list(
                        debug_image, hand_landmarks
                    )
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list
                    )
                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, hand_state.point_history
                    )

                    # Log (if logging mode is selected)
                    logging_csv(
                        number,
                        mode,
                        pre_processed_landmark_list,
                        pre_processed_point_history_list,
                    )

                    hand_sign_id = keypoint_classifier(
                        pre_processed_landmark_list
                    )
                    current_gesture = keypoint_classifier_labels[hand_sign_id]

                    if hand_sign_id == 2:
                        hand_state.point_history.append(landmark_list[8])
                    else:
                        hand_state.point_history.append([0, 0])

                    # Finger gesture classification
                    finger_gesture_id = 0
                    
                    if len(pre_processed_point_history_list) == (POINT_HISTORY_LENGTH * 2):
                        finger_gesture_id = point_history_classifier(
                            pre_processed_point_history_list
                        )

                    # Calculates the gesture IDs in the latest detection
                    hand_state.finger_gesture_history.append(finger_gesture_id)
                    
                    most_common_fg_id = Counter(
                        hand_state.finger_gesture_history
                    ).most_common()

                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        get_handedness_label(handedness),
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[most_common_fg_id[0][0]],
                    )
                    debug_image = draw_point_history(debug_image, hand_state.point_history)

                    if (
                        current_time - hand_state.last_sample_time
                        >= GESTURE_SAMPLE_INTERVAL
                    ):
                        hand_state.gesture_history.append(current_gesture)
                        hand_state.last_sample_time = current_time

                        # Once the history is full, look for a majority gesture.
                        if len(hand_state.gesture_history) == GESTURE_HISTORY_LENGTH:
                            gesture_counts = Counter(hand_state.gesture_history)
                            candidate, occurrences = gesture_counts.most_common(1)[0]

                            if occurrences >= GESTURE_MIN_OCCURRENCES:
                                # A new candidate starts its own confirmation
                                # period. We do NOT trigger the action immediately.
                                if hand_state.candidate_gesture != candidate:
                                    hand_state.candidate_gesture = candidate
                                    hand_state.candidate_since = current_time

                                # The candidate must remain stable for the
                                # confirmation delay before executing VIDAА.
                                elif (
                                    current_time - hand_state.candidate_since
                                    >= GESTURE_CONFIRMATION_DELAY
                                    and current_time - last_action_time
                                    >= GESTURE_COOLDOWN
                                ):
                                    action = translate_gesture_to_action(
                                        hand_label,
                                        candidate,
                                        point_history_classifier_labels[most_common_fg_id[0][0]]
                                    )

                                    # Send the command only after every safety check.
                                    if execute_vidaa_action(action):
                                        print(
                                            f"Action VIDAA exécutée : {action}"
                                        )
                                        if action != "Neutral":
                                            last_action_time = time.monotonic()
                                    else:
                                        print(
                                            f"Échec de l'action VIDAA : {action}"
                                        )

                                    # Prevent the same history from triggering
                                    # another action immediately.
                                    hand_state.gesture_history.clear()
                                    hand_state.candidate_gesture = None
                                    hand_state.candidate_since = 0.0
                                    hand_state.point_history.clear()
                                    hand_state.finger_gesture_history.clear()
                
            else:
                # Reset values if no hands are detected
                for hand_state in hand_states.values():
                    hand_state.gesture_history.clear()
                    hand_state.candidate_gesture = None
                    hand_state.candidate_since = 0.0
                    hand_state.point_history.clear()
                    hand_state.finger_gesture_history.clear()

            debug_image = draw_info(debug_image, fps, mode, number)

            cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def draw_info_text(image, brect, hand_label, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = hand_label
    if hand_sign_text != "":
        info_text = info_text + ' : ' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture : " + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture : " + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image



def get_handedness_label(handedness):
    # Extract a readable handedness label from MediaPipe Tasks output
    if not handedness:
        return "unknown"

    category = handedness[0]
    
    label = (
        getattr(category, "category_name", None)
        or getattr(category, "display_name", None)
        or "unknown"
    )

    return label.lower()


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # Wrist 1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # Wrist 2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # Thumb：Base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # Thumb：First joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # Thumb：Fingertips
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # Index finger：Base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # Index finger：Second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # Index finger：First joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # Index finger：Fingertips
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # Middle finger：Base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # Middle finger：Second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # Middle finger：First joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # Middle finger：Fingertips
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # Ring finger：Base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # Ring finger：Second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # Ring finger：First joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # Ring finger：Fingertips
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # Little finger：Base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # Little finger：Second joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # Little finger：First joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # Little finger：Fingertips
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


if __name__ == '__main__':
    main()

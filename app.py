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
import subprocess

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

GESTURE_ACTIONS = {
    "Open" : "Allumer",
    "Close" : "Eteindre",
    "OK" : "Selectionner",
    "Up" : "Haut",
    "Down" : "Bas",
    "Left" : "Gauche",
    "Right" : "Droite"
}

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

# Two-hand safety system
#
# The LEFT hand must continuously perform the "OK" gesture to authorize
# actions from the RIGHT hand. As soon as the left hand is no longer
# recognized as OK, right-hand actions are ignored.
AUTHORIZATION_GESTURE = "OK"

# VIDAA configuration
# Replace these values with your TV network information.
VIDAA_TV_IP = "192.168.1.286"

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

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def execute_vidaa_action(gesture):
    """Execute a validated gesture through the vidaa-control CLI."""
    if gesture == "Open":
        command = ["tv", "--ip", VIDAA_TV_IP, "on"]
    elif gesture == "Close":
        command = ["tv", "--ip", VIDAA_TV_IP, "off"]
    else:
        key_mapping = {
            "OK": "OK",
            "Up": "UP",
            "Down": "DOWN",
            "Left": "LEFT",
            "Right": "RIGHT",
        }
        key = key_mapping.get(gesture)
        if key is None:
            print(f"Geste VIDAA inconnu : {gesture}")
            return False
        command = ["tv", "--ip", VIDAA_TV_IP, "key", key]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            return True

        error = result.stderr.strip() or result.stdout.strip()
        print(f"Erreur VIDAA ({gesture}) : {error}")
        return False

    except subprocess.TimeoutExpired:
        print(f"Timeout lors de l'envoi de l'action VIDAA : {gesture}")
        return False
    except FileNotFoundError:
        print("Erreur VIDAA : la commande 'tv' est introuvable.")
        return False
    except Exception as exc:
        print(f"Erreur VIDAA ({gesture}) : {exc}")
        return False


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # MediaPipe Hand Landmarker initialization ###############################
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

    # Read labels ###########################################################
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

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    # Gesture action validation history ######################################
    # This history is deliberately sampled much more slowly than the camera
    # FPS. This prevents the beginning of a hand movement from filling the
    # history with dozens of transient predictions in a few milliseconds.
    gesture_action_history = deque(maxlen=GESTURE_HISTORY_LENGTH)

    # Left-hand authorization history.
    authorization_history = deque(maxlen=GESTURE_HISTORY_LENGTH)

    last_gesture_sample_time = 0.0
    last_authorization_sample_time = 0.0
    last_action_time = 0.0

    candidate_gesture = None
    candidate_since = 0.0

    # True only while the left hand is continuously recognized as OK.
    detection_enabled = False

    # Text displayed on screen for the last triggered action.
    action_text = ""

    #  ########################################################################
    mode = 0

    # VIDEO mode is synchronous and uses MediaPipe tracking between frames.
    timestamp_ms = 0

    with HandLandmarker.create_from_options(landmarker_options) as hands:
        while True:
            fps = cvFpsCalc.get()

            # Process Key (ESC: end) #################################################
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = select_mode(key, mode)

            # Camera capture #####################################################
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
            results = hands.detect_for_video(mp_image, timestamp_ms)

            if results.hand_landmarks:
                # -------------------------------------------------------------
                # Identify the left and right hands.
                #
                # MediaPipe provides one handedness result for each detected
                # hand. We keep the two hands separate so that:
                #
                #   LEFT  hand = authorization ("OK")
                #   RIGHT hand = TV command
                # -------------------------------------------------------------
                left_hand_data = None
                right_hand_data = None

                for hand_landmarks, handedness in zip(
                    results.hand_landmarks,
                    results.handedness,
                ):
                    handedness_label = get_handedness_label(handedness).lower()

                    # L'image est une image miroir, il faut donc retourner les cases
                    if handedness_label == "right":
                        left_hand_data = (hand_landmarks, handedness)
                    elif handedness_label == "left":
                        right_hand_data = (hand_landmarks, handedness)

                current_time = time.monotonic()

                # =============================================================
                # LEFT HAND: AUTHORIZATION
                # =============================================================
                if left_hand_data is not None:
                    left_landmarks, left_handedness = left_hand_data

                    left_landmark_list = calc_landmark_list(
                        debug_image, left_landmarks
                    )
                    left_pre_processed_landmark_list = pre_process_landmark(
                        left_landmark_list
                    )

                    # The logging system must work for BOTH hands.
                    # The left hand is used for static gesture training, so we
                    # record its keypoints without involving authorization.
                    if mode == 1:
                        logging_csv(
                            number,
                            mode,
                            left_pre_processed_landmark_list,
                            [],
                        )

                    left_hand_sign_id = keypoint_classifier(
                        left_pre_processed_landmark_list
                    )
                    left_gesture = keypoint_classifier_labels[
                        left_hand_sign_id
                    ]

                    # Sample the left-hand prediction at the same temporal
                    # rate as the right-hand action history.
                    if (
                        current_time - last_authorization_sample_time
                        >= GESTURE_SAMPLE_INTERVAL
                    ):
                        authorization_history.append(left_gesture)
                        last_authorization_sample_time = current_time

                    # Authorization is active only while the recent history
                    # confirms OK.
                    if len(authorization_history) == GESTURE_HISTORY_LENGTH:
                        authorization_counts = Counter(
                            authorization_history
                        )
                        authorization_gesture, occurrences = (
                            authorization_counts.most_common(1)[0]
                        )

                        detection_enabled = (
                            authorization_gesture == AUTHORIZATION_GESTURE
                            and occurrences >= GESTURE_MIN_OCCURRENCES
                        )
                    else:
                        detection_enabled = False

                    # Draw the left-hand bounding box and landmarks.
                    left_brect = calc_bounding_rect(
                        debug_image, left_landmarks
                    )
                    debug_image = draw_bounding_rect(
                        use_brect, debug_image, left_brect
                    )
                    debug_image = draw_landmarks(
                        debug_image, left_landmark_list
                    )

                else:
                    # No left hand = no authorization.
                    authorization_history.clear()
                    detection_enabled = False
                    last_authorization_sample_time = current_time

                # =============================================================
                # RIGHT HAND: TV COMMAND
                # =============================================================
                if right_hand_data is not None:
                    right_landmarks, right_handedness = right_hand_data

                    brect = calc_bounding_rect(
                        debug_image, right_landmarks
                    )
                    landmark_list = calc_landmark_list(
                        debug_image, right_landmarks
                    )

                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list
                    )
                    pre_processed_point_history_list = pre_process_point_history(
                        debug_image, point_history
                    )

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
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(
                            pre_processed_point_history_list
                        )

                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(
                        finger_gesture_history
                    ).most_common()

                    # ---------------------------------------------------------
                    # ACTION SAFETY
                    #
                    # The right-hand gesture history is only filled while the
                    # left hand authorizes the system.
                    # ---------------------------------------------------------
                    if detection_enabled:
                        if (
                            current_time - last_gesture_sample_time
                            >= GESTURE_SAMPLE_INTERVAL
                        ):
                            gesture_action_history.append(current_gesture)
                            last_gesture_sample_time = current_time

                        # Once the history is full, look for a majority gesture.
                        if len(gesture_action_history) == GESTURE_HISTORY_LENGTH:
                            gesture_counts = Counter(gesture_action_history)
                            candidate, occurrences = (
                                gesture_counts.most_common(1)[0]
                            )

                            if (
                                candidate in GESTURE_ACTIONS
                                and candidate != AUTHORIZATION_GESTURE
                                and occurrences >= GESTURE_MIN_OCCURRENCES
                            ):
                                # A new candidate starts its own confirmation
                                # period. We do NOT trigger the action immediately.
                                if candidate_gesture != candidate:
                                    candidate_gesture = candidate
                                    candidate_since = current_time

                                # The candidate must remain stable for the
                                # confirmation delay before executing VIDAА.
                                elif (
                                    current_time - candidate_since
                                    >= GESTURE_CONFIRMATION_DELAY
                                    and current_time - last_action_time
                                    >= GESTURE_COOLDOWN
                                ):
                                    action_text = GESTURE_ACTIONS[candidate]

                                    # Send the command only after every safety check.
                                    if execute_vidaa_action(candidate):
                                        print(
                                            f"Action VIDAA exécutée : {action_text}"
                                        )
                                    else:
                                        print(
                                            f"Échec de l'action VIDAA : {action_text}"
                                        )

                                    last_action_time = current_time

                                    # Prevent the same history from triggering
                                    # another action immediately.
                                    gesture_action_history.clear()
                                    candidate_gesture = None
                                    candidate_since = 0.0
                    else:
                        # The authorization was removed, so all pending right
                        # hand information must be discarded.
                        gesture_action_history.clear()
                        candidate_gesture = None
                        candidate_since = 0.0
                        last_gesture_sample_time = current_time

                    # Draw the right-hand bounding box and landmarks.
                    debug_image = draw_bounding_rect(
                        use_brect, debug_image, brect
                    )
                    debug_image = draw_landmarks(
                        debug_image, landmark_list
                    )

                    # Display whether the two-hand safety system is active.
                    authorization_text = (
                        "AUTORISE" if detection_enabled else "VERROUILLE"
                    )

                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[
                            most_common_fg_id[0][0]
                        ],
                        action_text,
                    )

                    cv.putText(
                        debug_image,
                        "Controle : " + authorization_text,
                        (10, 140),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 0),
                        4,
                        cv.LINE_AA,
                    )
                    cv.putText(
                        debug_image,
                        "Controle : " + authorization_text,
                        (10, 140),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv.LINE_AA,
                    )

                else:
                    # No right hand: there is no command to process.
                    point_history.append([0, 0])

                    gesture_action_history.clear()
                    candidate_gesture = None
                    candidate_since = 0.0
                    last_gesture_sample_time = current_time

            else:
                # No hands detected: everything is locked.
                point_history.append([0, 0])
                authorization_history.clear()
                gesture_action_history.clear()

                detection_enabled = False
                candidate_gesture = None
                candidate_since = 0.0

                current_time = time.monotonic()
                last_authorization_sample_time = current_time
                last_gesture_sample_time = current_time

            debug_image = draw_point_history(debug_image, point_history)
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


def draw_info_text(image, brect, hand_sign_text,
                   finger_gesture_text, action_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = "Right"
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

    if action_text != "":
        cv.putText(image, "Action : " + action_text, (10, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Action : " + action_text, (10, 100),
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
        # landmark_z = landmark.z

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
    """Extract a readable handedness label from MediaPipe Tasks output."""
    if not handedness:
        return "Unknown"

    category = handedness[0]
    return (
        getattr(category, "category_name", None)
        or getattr(category, "display_name", None)
        or "Unknown"
    )


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
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


if __name__ == '__main__':
    main()
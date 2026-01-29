import cv2
import time
import random
import face_recognition
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

# Eye Aspect Ratio threshold (tuned empirically)
EYE_AR_THRESH = 0.20

# Number of consecutive frames required
BLINK_FRAMES_REQUIRED = 3
TURN_FRAMES_REQUIRED = 3

CHALLENGES = ["blink", "turn_left", "turn_right"]


# -----------------------------
# EAR (Eye Aspect Ratio)
# -----------------------------
def get_ear(eye_points):
    """
    Computes Eye Aspect Ratio (EAR)
    """
    p2_p6 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    p3_p5 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    p1_p4 = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))

    if p1_p4 == 0:
        return 1.0

    return (p2_p6 + p3_p5) / (2.0 * p1_p4)


# -----------------------------
# Head Turn Detection
# -----------------------------
def detect_head_turn(face_landmarks):
    """
    Determines head orientation based on relative nose position
    """
    nose_bridge = face_landmarks["nose_bridge"]
    chin = face_landmarks["chin"]

    nose_tip_x = nose_bridge[-1][0]
    face_left_x = chin[0][0]
    face_right_x = chin[16][0]

    total_width = face_right_x - face_left_x
    if total_width == 0:
        return "center"

    relative_nose_pos = (nose_tip_x - face_left_x) / total_width

    if relative_nose_pos < 0.35:
        return "turn_right"
    elif relative_nose_pos > 0.65:
        return "turn_left"
    else:
        return "center"


# -----------------------------
# Challenge Runner
# -----------------------------
def run_challenge(timeout=10):
    challenge = random.choice(CHALLENGES)
    logger.info(f"Anti-spoofing challenge issued: {challenge}")

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    blink_frames = 0
    turn_frames = 0

    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if not ret:
            continue

        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"ACTION REQUIRED: {challenge.upper()}",
            (30, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        # Speed optimization
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_landmarks_list = face_recognition.face_landmarks(rgb_small)

        if face_landmarks_list:
            # Scale landmarks back
            landmarks = {
                key: [(int(pt[0] * 2), int(pt[1] * 2)) for pt in val]
                for key, val in face_landmarks_list[0].items()
            }

            # -------- Blink Logic --------
            if challenge == "blink":
                if "left_eye" in landmarks and "right_eye" in landmarks:
                    left_ear = get_ear(landmarks["left_eye"])
                    right_ear = get_ear(landmarks["right_eye"])
                    avg_ear = (left_ear + right_ear) / 2

                    if avg_ear < EYE_AR_THRESH:
                        blink_frames += 1
                    else:
                        blink_frames = 0

                    if blink_frames >= BLINK_FRAMES_REQUIRED:
                        logger.info("Blink verified via EAR + temporal validation")
                        cap.release()
                        cv2.destroyAllWindows()
                        return True

            # -------- Head Turn Logic --------
            elif challenge in ["turn_left", "turn_right"]:
                current_turn = detect_head_turn(landmarks)

                if current_turn == challenge:
                    turn_frames += 1
                else:
                    turn_frames = 0

                if turn_frames >= TURN_FRAMES_REQUIRED:
                    logger.info(f"Head turn verified: {challenge}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return True

        cv2.imshow("Security Check", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.warning("Liveness check failed or timed out")
    return False


# -----------------------------
# Standalone Test
# -----------------------------
if __name__ == "__main__":
    result = run_challenge()
    print("Challenge result:", result)

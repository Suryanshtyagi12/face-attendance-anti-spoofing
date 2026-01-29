import os
import cv2
import numpy as np
import face_recognition
from logger import get_logger
from utils import extract_face_embedding, compute_distance, is_match
from spoof_check import run_challenge
from attendance import mark_attendance

logger = get_logger(__name__)

EMBEDDINGS_DIR = "data/embeddings"
THRESHOLD = 0.6  # justified in research notebook


def load_known_embeddings():
    known = {}

    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            user_id = file.replace(".npy", "")
            path = os.path.join(EMBEDDINGS_DIR, file)
            known[user_id] = np.load(path)

    logger.info(f"Loaded embeddings for {len(known)} users")
    return known


def recognize_face():
    known_embeddings = load_known_embeddings()

    if not known_embeddings:
        logger.error("No registered users found")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Webcam not accessible")
        return

    logger.info("Recognition system started")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) != 1:
            cv2.putText(
                frame,
                "Show exactly ONE face",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        embedding = extract_face_embedding(rgb_frame)
        if embedding is None:
            continue

        min_dist = float("inf")
        identity = "Unknown"

        for user_id, stored_embeddings in known_embeddings.items():
            for ref_emb in stored_embeddings:
                dist = compute_distance(embedding, ref_emb)
                if dist < min_dist:
                    min_dist = dist
                    identity = user_id

        if is_match(min_dist, THRESHOLD):
            label = f"Recognized: {identity}"
            logger.info(f"User recognized: {identity} | distance={min_dist:.4f}")

            cv2.putText(
                frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Face Recognition", frame)
            cv2.waitKey(500)

            # 🔐 Run Anti-Spoofing
            logger.info(f"Running anti-spoofing for {identity}")
            spoof_passed = run_challenge()

            if spoof_passed:
                logger.info(f"Anti-spoofing passed for {identity}")
                mark_attendance(identity)
            else:
                logger.warning(f"Anti-spoofing failed for {identity}")

            break  # stop after one successful attempt

        else:
            label = "Unknown"
            logger.warning(f"Unknown face attempt | distance={min_dist:.4f}")

        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Recognition system stopped")


if __name__ == "__main__":
    recognize_face()

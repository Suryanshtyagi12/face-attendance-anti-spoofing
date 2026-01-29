import cv2
import os
import numpy as np
import face_recognition
from logger import get_logger
from utils import extract_face_embedding

logger = get_logger(__name__)

EMBEDDINGS_DIR = "data/embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

SAMPLES_PER_USER = 15


def register_user(user_id):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Webcam not accessible")
        return

    embeddings = []
    sample_count = 0

    logger.info(f"Starting registration for user {user_id}")

    while sample_count < SAMPLES_PER_USER:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) != 1:
            cv2.putText(
                frame,
                "Ensure exactly ONE face is visible",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Register Face", frame)
            cv2.waitKey(1)
            continue

        embedding = extract_face_embedding(rgb_frame)

        if embedding is not None:
            embeddings.append(embedding)
            sample_count += 1
            logger.info(f"Captured sample {sample_count}/{SAMPLES_PER_USER}")

        cv2.putText(
            frame,
            f"Samples: {sample_count}/{SAMPLES_PER_USER}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Register Face", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if embeddings:
        embeddings = np.array(embeddings)
        save_path = os.path.join(EMBEDDINGS_DIR, f"{user_id}.npy")
        np.save(save_path, embeddings)
        logger.info(f"Registration completed for user {user_id}")
    else:
        logger.error("No embeddings captured; registration failed")


if __name__ == "__main__":
    user_id = input("Enter user ID (e.g., emp_001): ").strip()
    register_user(user_id)

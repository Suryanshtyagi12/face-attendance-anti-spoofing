import numpy as np
import face_recognition
from logger import get_logger

logger = get_logger(__name__)

# -----------------------------
# Face Preprocessing
# -----------------------------
def extract_face_embedding(image_rgb):
    """
    Takes an RGB image (numpy array)
    Returns a 128-D face embedding or None
    """
    try:
        encodings = face_recognition.face_encodings(image_rgb)

        if len(encodings) == 0:
            logger.warning("No face encoding found")
            return None

        return encodings[0]

    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        return None


# -----------------------------
# Distance Calculation
# -----------------------------
def compute_distance(emb1, emb2, metric="euclidean"):
    """
    Computes distance between two embeddings
    """
    if metric == "euclidean":
        return np.linalg.norm(emb1 - emb2)

    elif metric == "cosine":
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        return 1 - (dot / norm)

    else:
        raise ValueError("Unsupported distance metric")


# -----------------------------
# Threshold Decision
# -----------------------------
def is_match(distance, threshold):
    """
    Returns True if distance is within threshold
    """
    match = distance < threshold
    logger.info(
        f"Match decision | distance={distance:.4f}, threshold={threshold}, result={match}"
    )
    return match

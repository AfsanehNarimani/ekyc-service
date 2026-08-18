from enum import Enum


class FaceVerificationComparisonStatus(str, Enum):
    SUCCESS = "success"

    REFERENCE_IMAGE_LOAD_FAILED = "reference_image_load_failed"
    CANDIDATE_IMAGE_LOAD_FAILED = "candidate_image_load_failed"

    REFERENCE_FACE_NOT_DETECTED = "reference_face_not_detected"
    REFERENCE_MULTIPLE_FACES_DETECTED = "reference_multiple_faces_detected"
    CANDIDATE_FACE_NOT_DETECTED = "candidate_face_not_detected"
    CANDIDATE_MULTIPLE_FACES_DETECTED = "candidate_multiple_faces_detected"

    REFERENCE_FACE_ALIGNMENT_FAILED = "reference_face_alignment_failed"
    CANDIDATE_FACE_ALIGNMENT_FAILED = "candidate_face_alignment_failed"

    REFERENCE_EMBEDDING_FAILED = "reference_embedding_failed"
    CANDIDATE_EMBEDDING_FAILED = "candidate_embedding_failed"

    UNEXPECTED_ERROR = "unexpected_error"

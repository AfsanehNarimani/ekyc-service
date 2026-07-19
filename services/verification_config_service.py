from config import settings


class VerificationConfigService:
    def get_config(self) -> dict:
        return {
            "default_face_model": settings.default_face_model,
            "supported_actions": [
                "center",
                "left",
                "right",
                "up",
                "down",
            ],
            "face_match_levels": {
                "1": {
                    "name": "relaxed",
                    "threshold": settings.face_match_threshold_level_1,
                },
                "2": {
                    "name": "normal",
                    "threshold": settings.face_match_threshold_level_2,
                },
                "3": {
                    "name": "strict",
                    "threshold": settings.face_match_threshold_level_3,
                },
            },
            "liveness_levels": {
                "1": {
                    "name": "relaxed",
                    "min_required_frames": settings.liveness_min_frames_level_1,
                    "min_pose_variation_score": (
                        settings.liveness_pose_variation_level_1
                    ),
                },
                "2": {
                    "name": "normal",
                    "min_required_frames": settings.liveness_min_frames_level_2,
                    "min_pose_variation_score": (
                        settings.liveness_pose_variation_level_2
                    ),
                },
                "3": {
                    "name": "strict",
                    "min_required_frames": settings.liveness_min_frames_level_3,
                    "min_pose_variation_score": (
                        settings.liveness_pose_variation_level_3
                    ),
                },
            },
        }

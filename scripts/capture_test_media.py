import time
from pathlib import Path

import cv2

OUTPUT_DIR = Path.home() / "Desktop"
IMAGE_PATH = OUTPUT_DIR / "id_card.jpg"
VIDEO_PATH = OUTPUT_DIR / "user_video.mp4"


def main() -> None:
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    fps = 20
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    print("Webcam opened.")
    print("Press 'p' to capture id_card.jpg")
    print("Press 'q' to quit.")

    captured_photo = False

    while True:
        ok, frame = camera.read()

        if not ok:
            raise RuntimeError("Could not read frame from webcam.")

        preview = frame.copy()

        cv2.putText(
            preview,
            "Press 'p' to capture id_card.jpg",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Capture Test Media", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("p"):
            cv2.imwrite(str(IMAGE_PATH), frame)
            captured_photo = True
            print(f"Saved image: {IMAGE_PATH}")
            break

        if key == ord("q"):
            break

    if not captured_photo:
        camera.release()
        cv2.destroyAllWindows()
        return

    print()
    print("Now recording user_video.mp4 for 8 seconds.")
    print("Move your head: center, left, right, up, down.")

    writer = cv2.VideoWriter(
        str(VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    start_time = time.time()

    while time.time() - start_time < 8:
        ok, frame = camera.read()

        if not ok:
            break

        elapsed = time.time() - start_time
        remaining = max(0, int(8 - elapsed))

        preview = frame.copy()

        cv2.putText(
            preview,
            f"Recording... {remaining}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            preview,
            "Move: center -> left -> right -> up -> down",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        writer.write(frame)
        cv2.imshow("Capture Test Media", preview)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    writer.release()
    camera.release()
    cv2.destroyAllWindows()

    print(f"Saved video: {VIDEO_PATH}")


if __name__ == "__main__":
    main()

import math
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Landmark ids: fingertip and the joint below it (PIP) for index, middle, ring, pinky
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def count_fingers(lm):
    """Return number of raised fingers (0-5) from hand landmarks."""
    count = 0

    # Thumb: open if the tip is farther from the pinky base than the thumb's IP joint.
    # This works for left/right hands and either palm direction.
    if dist(lm[4], lm[17]) > dist(lm[3], lm[17]):
        count += 1

    # Other four fingers: tip is above its PIP joint (y grows downward in images)
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        if lm[tip].y < lm[pip].y:
            count += 1

    return count


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        total = 0
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=4),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2),
                )
                total = count_fingers(hand_lms.landmark)

        # Green box with count
        cv2.rectangle(frame, (20, 20), (330, 110), (0, 255, 0), cv2.FILLED)
        cv2.putText(
            frame,
            f"COUNT: {total}",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 0, 0),
            4,
        )

        cv2.imshow("AI Finger Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


import cv2
import mediapipe as mp
import pyautogui
import math

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

hand_detector = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
bounding_box = (50, 50, screen_width - 50, screen_height - 50)  # define bounding box

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output = hand_detector.process(rgb_frame)

    if output.multi_hand_landmarks:  # Check if any hands were detected
        for hand in output.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

            index_x, index_y, middle_x, middle_y, thumb_x, thumb_y, ring_x, ring_y = None, None, None, None, None, None, None, None
            for id, landmark in enumerate(hand.landmark):
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

                if id == 8:  # Index finger tip
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
                    index_x, index_y = int(x * screen_width / frame.shape[1]), int(y * screen_height / frame.shape[0])

                elif id == 12:  # Middle finger tip
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    middle_x, middle_y = int(x * screen_width / frame.shape[1]), int(y * screen_height / frame.shape[0])

                elif id == 4:  # Thumb tip
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    thumb_x, thumb_y = int(x * screen_width / frame.shape[1]), int(y * screen_height / frame.shape[0])

                elif id == 16:  # Ring finger tip
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    ring_x, ring_y = int(x * screen_width / frame.shape[1]), int(y * screen_height / frame.shape[0])


            if index_x and index_y and middle_x and middle_y:
                distance = math.sqrt((middle_x - index_x) ** 2 + (middle_y - index_y) ** 2)
                if distance < 75:
                    pyautogui.moveTo(middle_x, middle_y)

            if 'thumb_x' in locals() and 'thumb_y' in locals() and 'middle_x' in locals() and 'middle_y' in locals():
                distance = math.sqrt((middle_x - thumb_x) ** 2 + (middle_y - thumb_y) ** 2)
                if distance < 75:
                    pyautogui.rightClick()

        if 'index_x' in locals() and 'index_y' in locals() and 'thumb_x' in locals() and 'thumb_y' in locals():
            distance = abs(index_y - thumb_y)
            if distance < 50:
                pyautogui.click()

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import csv
import os

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# File setup
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

label = input("Enter the label for this data (e.g. A, B, hello): ")
csv_path = os.path.join(output_dir, f"{label}.csv")

# OpenCV setup
cap = cv2.VideoCapture(0)
print("Press 's' to save frame, 'q' to quit.")

with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = []
    for i in range(21):
        header += [f'x{i}', f'y{i}', f'z{i}']
    writer.writerow(header)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert color
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Save data on 's' keypress
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    data_row = []
                    for lm in hand_landmarks.landmark:
                        data_row += [lm.x, lm.y, lm.z]
                    writer.writerow(data_row)
                    print(f"Saved frame for label '{label}'")

        # Show frame
        cv2.imshow('Hand Tracking', frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()

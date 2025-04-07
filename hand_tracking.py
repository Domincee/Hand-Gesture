import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Start capturing video
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for MediaPipe processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        left_index = None
        right_index = None

        # Check if hands are detected
        if result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Get the coordinates of the index finger (landmark 8)
                x = int(hand_landmarks.landmark[8].x * frame.shape[1])
                y = int(hand_landmarks.landmark[8].y * frame.shape[0])

                # Determine if it's the left or right hand using multi_handedness
                hand_label = result.multi_handedness[i].classification[0].label       

                # Log the index finger position
                if hand_label == 'Left':
                    left_index = (x, y)
                elif hand_label == 'Right':
                    right_index = (x, y)

                # Optionally, you can also draw the landmarks on the frame
                for landmark in hand_landmarks.landmark:
                    cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # Draw a circle at the index finger position
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

        if left_index and right_index:
            # Draw a line connecting the left and right index fingers
            cv2.line(frame, left_index, right_index, (255, 0, 0), 2)

        # Show the frame with tracked landmarks
        cv2.imshow("Hand Tracking", frame)

        # Press 'ESC' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            print("Exiting...")
            break
finally:
    # Release resources when exiting
    cap.release()
    cv2.destroyAllWindows()
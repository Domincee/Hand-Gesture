import cv2
import numpy as np
import mediapipe as mp
import sys
import signal

# Signal handler for graceful exit (Ctrl+C)
def signal_handler(sig, frame):
    print("Gracefully shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Set up signal handling for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

# Setup MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Create a blank canvas
canvas = None
prev_x, prev_y = 0, 0  # To store previous finger position

def is_hand_closed(hand_landmarks, threshold=0.02):
    """
    Determines if the hand is closed with adjustable sensitivity.
    :param hand_landmarks: Hand landmarks from MediaPipe.
    :param threshold: Sensitivity for determining closure. Higher values allow more movement.
    :return: True if the hand is closed, False otherwise.
    """
    # Check each finger: tip y-coordinate should be below its base y-coordinate.
    index_closed = hand_landmarks.landmark[8].y > (hand_landmarks.landmark[6].y + threshold)
    middle_closed = hand_landmarks.landmark[12].y > (hand_landmarks.landmark[10].y + threshold)
    ring_closed = hand_landmarks.landmark[16].y > (hand_landmarks.landmark[14].y + threshold)
    pinky_closed = hand_landmarks.landmark[20].y > (hand_landmarks.landmark[18].y + threshold)
    thumb_closed = hand_landmarks.landmark[4].y > (hand_landmarks.landmark[2].y + threshold)

    # Return True if most fingers are closed (e.g., 4 out of 5)
    return index_closed and middle_closed and ring_closed and pinky_closed  # Ignore thumb for flexibility
   # Main loop to process webcam feed
try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)  # Mirror the image
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get index finger tip (landmark 8)
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)

                # Check if the hand is closed
                if is_hand_closed(hand_landmarks):
                    # Display text when the hand is closed
                    cv2.putText(frame, "Hand is Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    prev_x, prev_y = 0, 0  # Reset the drawing position
                else:
                    # Only draw if the hand is open
                    if prev_x != 0 and prev_y != 0:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 5)
                    prev_x, prev_y = x, y
        else:
            prev_x, prev_y = 0, 0  # Reset when no hand is detected

        # Combine canvas and webcam feed
        frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        # Show the frame in the OpenCV window
        cv2.imshow("Virtual Painter", frame)

        # Check for key press (using waitKey to detect it)
        key = cv2.waitKey(1) & 0xFF
        
        # Debugging: print key code to check if it's detecting keys
        print(f"Key pressed: {key}")
        
        if key == 27:  # ESC key to exit
            print("ESC key pressed. Exiting...")
            break

        # Check if the window is closed
        if cv2.getWindowProperty("Virtual Painter", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed manually. Exiting...")
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Ensure proper cleanup and release of resources
    print("Releasing camera and closing windows...")
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

    # Check if the webcam feed is still being displayed
    if cv2.getWindowProperty("Virtual Painter", cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow("Virtual Painter")
        print("Window properly destroyed.")
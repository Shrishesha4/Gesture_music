import cv2
import argparse
import os
from gesture_controller import GestureController
from camera_selector import CameraSelector

def main():
    """Main function to run the hand gesture music controller."""
    parser = argparse.ArgumentParser(description="Gesture-controlled audio player")
    parser.add_argument("--audio", type=str, default="music.wav", help="Path to audio file")
    parser.add_argument("--camera", type=int, default=0, help="Index of the camera to use")
    args = parser.parse_args()

    # Check if audio file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file '{args.audio}' not found.")
        return

    # Select camera
    camera_selector = CameraSelector()
    camera_index = camera_selector.select_camera()
    if camera_index is None:
        return

    # Initialize video capture
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Initialize gesture controller
    controller = GestureController(args.audio)
    print("\nGesture Controls:")
    print("- Use two hands: Move them apart/together to control playback speed")
    print("- Use one hand: Pinch gesture to control pitch")
    print("- Press 'q' to quit")
    print("- Press 'p' to play/pause")
    print("- Press 'r' to reset speed and pitch\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = controller.process_frame(frame)
        
        # Add control status overlay
        status_text = []
        status_text.append("Controls:")
        status_text.append("P: Play/Pause")
        status_text.append("R: Reset")
        status_text.append("Q: Quit")
        status_text.append(f"Speed: {controller.current_speed:.2f}x")
        status_text.append(f"Pitch: {controller.current_pitch:.1f}")
        status_text.append("Status: " + ("Paused" if controller.paused else "Playing"))

        # Display status text
        y_position = 30
        for text in status_text:
            cv2.putText(frame, text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_position += 25

        cv2.imshow("Hand Gesture Controller", frame)

        # Handle keyboard input with shorter wait time
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('p'):
            controller.toggle_play_pause()
        elif key == ord('r'):
            controller.reset_controls()

    controller.cleanup()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

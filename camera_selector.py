import cv2

class CameraSelector:
    """
    A utility class to help select the correct camera index,
    especially when using Continuity Camera on macOS.
    """

    def __init__(self):
        self.max_cameras_to_check = 10  # You can increase this if needed

    def get_available_cameras(self):
        """
        Checks for available cameras and returns a list of tuples containing
        the camera index and a boolean indicating if the camera opened successfully.
        """
        available_cameras = []
        for i in range(self.max_cameras_to_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append((i, True))
                cap.release()  # Release the camera immediately after checking
            else:
                available_cameras.append((i, False))
        return available_cameras

    def select_camera(self):
        """
        Attempts to use the default camera (index 0). If unavailable, prompts the user to select a camera.
        Returns the selected camera index or None if the user cancels.
        """
        default_camera_index = 0
        cap = cv2.VideoCapture(default_camera_index)
        if cap.isOpened():
            cap.release()
            return default_camera_index
        else:
            print(f"Default camera (index {default_camera_index}) not available.")

            available_cameras = self.get_available_cameras()

            if not any(status for _, status in available_cameras):
                print("No cameras found. Please ensure a camera is connected and accessible.")
                return None

            print("Available cameras:")
            for index, status in available_cameras:
                print(f"Index: {index}, {'Available' if status else 'Not Available'}")

            while True:
                try:
                    selected_index = int(input("Enter the index of the camera you want to use: "))
                    if 0 <= selected_index < self.max_cameras_to_check and available_cameras[selected_index][1]:
                        return selected_index
                    else:
                        print("Invalid camera index. Please choose an available camera from the list.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except IndexError:
                    print("Invalid input. Please enter a number between 0 and", self.max_cameras_to_check)

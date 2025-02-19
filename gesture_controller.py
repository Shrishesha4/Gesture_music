import cv2
import mediapipe as mp
from pydub import AudioSegment
import numpy as np
import threading
import pygame
import soundfile as sf
import os
import tempfile
import pyrubberband as pyrb
import time
from utils import calculate_distance

class GestureController:
    """
    A class to control music playback speed and pitch using hand gestures.
    """
    def __init__(self, audio_file, camera_index=0):
        """
        Initializes the GestureController with an audio file.

        Args:
            audio_file (str): Path to the audio file.
            camera_index (int): Index of the camera to use.
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2) # Increased confidence and hand number
        self.audio = AudioSegment.from_file(audio_file)
        self.playing = False
        self.current_speed = 1.0
        self.current_pitch = 0.0
        self.audio_process = None  # Store the audio process
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        self.is_playing = False
        self.audio_file = audio_file
        # Create a temporary output file for SoundStretch
        self.temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        self.camera_index = camera_index
        self.last_update_time = 0
        self.update_interval = 0.2  # Increased to 200ms for better stability
        self.last_speed = 1.0
        self.last_pitch = 0.0
        self.speed_threshold = 0.05  # Minimum change in speed to trigger update
        self.pitch_threshold = 0.5   # Minimum change in pitch to trigger update
        # Load audio file once
        self.audio_data, self.sample_rate = sf.read(audio_file)
        self.paused = False
        # Start playing immediately after initialization
        pygame.mixer.music.play(-1)  # -1 means loop indefinitely
        self.is_playing = True

    def process_frame(self, frame):
        """
        Processes a single frame from the webcam, detects hand gestures,
        and adjusts audio playback accordingly.

        Args:
            frame (numpy.ndarray): A frame from the webcam.

        Returns:
            numpy.ndarray: The processed frame with hand landmarks drawn on it.
        """
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = self.hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
        # Draw visualization bars
        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width // 2
        
        # Draw volume bars
        volume_level = int(pygame.mixer.music.get_volume() * 10)
        for i in range(10):
            bar_height = 20
            bar_y = frame_height // 2 - 100 + i * 25
            color = (0, 255, 0) if i < volume_level else (100, 100, 100)
            cv2.rectangle(frame, (center_x - 50, bar_y), 
                         (center_x + 50, bar_y + bar_height), color, -1)
    
        # Draw speed and frequency indicators
        cv2.putText(frame, f"speed", (50, frame_height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"{self.current_speed:.2f}", (50, frame_height // 2 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Frequency", (frame_width - 200, frame_height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frequency = int(self.sample_rate * self.current_pitch)
        cv2.putText(frame, f"{frequency}", (frame_width - 200, frame_height // 2 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if results.multi_hand_landmarks:
            self.hand_count = len(results.multi_hand_landmarks)
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                self.handle_gestures(hand_landmarks, results.multi_hand_landmarks)

        return frame

    def handle_gestures(self, hand_landmarks, all_hand_landmarks):
        """
        Handles gesture recognition and updates audio playback based on gestures.
        """
        if len(all_hand_landmarks) == 2:
            # Get landmarks for both hands
            hand1_landmarks = all_hand_landmarks[0]
            hand2_landmarks = all_hand_landmarks[1]
    
            # Calculate distance between hands for volume
            hand_distance = calculate_distance(
                hand1_landmarks.landmark[self.mp_hands.HandLandmark.WRIST],
                hand2_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            )
            
            # Map hand distance to volume (0.0 to 1.0)
            volume = np.clip(hand_distance, 0.0, 1.0)
            pygame.mixer.music.set_volume(volume)
    
            # Handle speed control with first hand
            thumb_tip1 = hand1_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip1 = hand1_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinch_distance1 = calculate_distance(thumb_tip1, index_tip1)
            
            # Map first hand pinch to speed
            new_speed = np.clip(1.0 + (pinch_distance1 - 0.1) * 2.0, 0.5, 2.0)
            if abs(new_speed - self.last_speed) > self.speed_threshold:
                self.current_speed = new_speed
                self.last_speed = new_speed
                self.adjust_audio()
    
            # Handle frequency control with second hand
            thumb_tip2 = hand2_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip2 = hand2_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinch_distance2 = calculate_distance(thumb_tip2, index_tip2)
            
            # Map second hand pinch to frequency
            frequency_shift = np.clip((pinch_distance2 - 0.1) * 48 - 12, -12, 12)
            if abs(frequency_shift - self.current_pitch) > self.pitch_threshold:
                self.current_pitch = frequency_shift
                self.adjust_audio()

        # Handle pitch control with one hand
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Calculate pinch distance
        pinch_distance = calculate_distance(thumb_tip, index_tip)
        
        # Map pinch distance to pitch with adjusted scaling
        new_pitch = np.clip((pinch_distance - 0.1) * 48 - 12, -12, 12)
        
        # Only update if change is significant
        if abs(new_pitch - self.last_pitch) > self.pitch_threshold:
            self.current_pitch = new_pitch
            self.last_pitch = new_pitch
            self.adjust_audio()
            print(f"Pitch updated: {self.current_pitch:.1f}")

    def adjust_audio(self):
        """
        Adjusts the audio playback speed and pitch using pyrubberband.
        """
        # Rate limiting to prevent too frequent updates
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        self.last_update_time = current_time

        try:
            # Process audio in a separate thread to prevent blocking
            threading.Thread(target=self._process_audio).start()
        except Exception as e:
            print(f"Error processing audio: {e}")

    def _process_audio(self):
        """
        Internal method to process audio with pyrubberband.
        """
        try:
            if not self.paused:
                # Process audio with pyrubberband
                modified_audio = pyrb.time_stretch(
                    self.audio_data, 
                    self.sample_rate, 
                    self.current_speed
                )
                modified_audio = pyrb.pitch_shift(
                    modified_audio, 
                    self.sample_rate, 
                    int(round(self.current_pitch))  # Convert pitch to nearest integer
                )

                # Save to temporary file
                sf.write(self.temp_output, modified_audio, self.sample_rate)

                # Play the processed audio
                pygame.mixer.music.load(self.temp_output)
                pygame.mixer.music.play()
                self.is_playing = True
                
                print(f"Audio processed - Speed: {self.current_speed:.2f}x, Pitch: {self.current_pitch:.1f}")

        except Exception as e:
            print(f"Error in audio processing: {e}")
            print(f"Current speed: {self.current_speed}, Current pitch: {self.current_pitch}")

    def stop_audio(self):
        """
        Stops the currently playing audio using pygame.
        """
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False

    def toggle_play_pause(self):
        """Toggles between play and pause states."""
        if self.is_playing:
            if self.paused:
                pygame.mixer.music.unpause()
                print("Resuming playback")
            else:
                pygame.mixer.music.pause()
                print("Pausing playback")
            self.paused = not self.paused
        else:
            pygame.mixer.music.play(-1)
            self.is_playing = True
            print("Starting playback")

    def reset_controls(self):
        """Resets speed and pitch to default values."""
        print("Resetting controls to default")
        self.current_speed = 1.0
        self.current_pitch = 0.0
        self._last_speed = 1.0
        self._last_pitch = 0.0
        self.adjust_audio()
        # Ensure music is playing after reset
        if not self.is_playing:
            pygame.mixer.music.play(-1)
            self.is_playing = True
            self.paused = False

    def cleanup(self):
        """Stops the audio playback and quits pygame."""
        self.stop_audio()
        pygame.mixer.quit()
        # Clean up the temporary file
        try:
            os.unlink(self.temp_output)
        except:
            pass

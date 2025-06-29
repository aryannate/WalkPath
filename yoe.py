# -*- coding: utf-8 -*-
import cv2
import tkinter as tk
from tkinter import messagebox, Label, Button, Frame
from PIL import Image, ImageTk
from ultralytics import YOLO
import google.generativeai as genai
import pyttsx3
import time
import threading
import os

# --- 1. SETUP & CONFIGURATION ---

# IMPORTANT: Load the API key from an environment variable for security.
# See instructions on how to set this up.
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not found.")
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    # Use messagebox for better visibility in a GUI app
    messagebox.showerror("API Key Error", f"Error configuring Gemini API: {e}\nPlease ensure you have set the GEMINI_API_KEY environment variable.")
    exit()

# --- 2. MAIN APPLICATION CLASS ---
class WalkPathNavApp:
    def __init__(self, window, title):
        self.window = window
        self.window.title(title)
        # Set a reasonable starting size
        self.window.geometry("1000x650")
        self.window.configure(bg="#f0f0f0") # A light grey background is easier on the eyes

        # --- State Variables ---
        self.is_running = False
        self.cap = None
        self.latest_frame = None
        self.ai_thread = None
        self.last_spoken_time = 0

        # --- AI & Engine Setup ---
        try:
            # NOTE: If 'yolov8s.pt' is not found, Ultralytics will attempt to
            # download it automatically. This requires an internet connection.
            self.yolo_model = YOLO("yolov8s.pt")
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
            self.tts_engine = pyttsx3.init()
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize models or TTS engine: {e}")
            self.window.destroy()
            return

        # --- GUI Setup ---
        self.setup_gui()

        # Handle window closing event gracefully
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_gui(self):
        # Main content frame
        content_frame = Frame(self.window, bg="#f0f0f0")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Left Side: Video Feed ---
        # The Label will hold the video frames.
        # We do NOT set a fixed width/height here; it will resize with the image.
        self.video_label = Label(content_frame, bg="black", text="Camera is off", fg="white", font=('Arial', 14))
        # Use 'fill=tk.BOTH' and 'expand=True' to make it take available space.
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # --- Right Side: Controls and Status ---
        right_frame = Frame(content_frame, width=300, bg="white")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        # Prevents the frame from shrinking to fit its contents
        right_frame.pack_propagate(False)

        self.btn_start = Button(right_frame, text="Start Navigation", font=('Arial', 14, 'bold'), command=self.start_navigation, bg="#4CAF50", fg="white")
        self.btn_start.pack(pady=20, padx=20, fill=tk.X)

        self.btn_stop = Button(right_frame, text="Stop Navigation", font=('Arial', 14, 'bold'), command=self.stop_navigation, state=tk.DISABLED, bg="#f44336", fg="white")
        self.btn_stop.pack(pady=10, padx=20, fill=tk.X)

        status_header = Label(right_frame, text="AI ASSISTANT STATUS", font=('Arial', 12, 'bold'), bg="white")
        status_header.pack(pady=(30, 5), padx=20, anchor=tk.W)

        self.ai_status_label = Label(right_frame, text="Standing by...", bg="white", fg="black", wraplength=260, justify=tk.LEFT, font=('Arial', 11))
        self.ai_status_label.pack(pady=5, padx=20, fill=tk.BOTH, expand=True, anchor=tk.NW)

    def start_navigation(self):
        if self.is_running:
            return

        # Attempt to open the default camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Could not access the webcam. Please ensure it is connected and not in use by another application.")
            return

        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.update_status_label("Activating systems...")
        self.speak("Navigation system activated.")

        # Start the AI processing in a separate thread to not freeze the GUI
        self.ai_thread = threading.Thread(target=self.run_gemini_assistant, daemon=True)
        self.ai_thread.start()

        # Start updating the video frame on the GUI
        self.update_frame()

    def stop_navigation(self):
        if not self.is_running:
            return

        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

        # Release the camera
        if self.cap:
            self.cap.release()
            self.cap = None

        # Reset the video label to its initial state
        self.video_label.config(image='', text="Camera is off", bg="black")
        self.update_status_label("System offline. Press Start to begin.")
        self.speak("Navigation system shutting down.")

    def on_close(self):
        """Called when the user closes the window."""
        self.stop_navigation()
        # Allow some time for threads to wind down before destroying window
        self.window.after(200, self.window.destroy)


    def update_frame(self):
        """Continuously gets a frame from the webcam and displays it."""
        # Only run if the system is active and camera is available
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Store a copy of the raw frame for the AI thread
                self.latest_frame = frame.copy()

                # Process with YOLO for object detection and get the annotated frame
                results = self.yolo_model(frame, verbose=False)
                annotated_frame = results[0].plot()

                # Convert the frame for Tkinter display
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)

                # Update the video label with the new frame
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

            # Schedule the next frame update
            self.window.after(20, self.update_frame)

    def run_gemini_assistant(self):
        """Runs in a separate thread, sending frames to Gemini for analysis."""
        nav_prompt = """
        You are an expert navigation assistant for a visually impaired user.
        Your task is to analyze the provided image of an indoor scene and give ONE SINGLE, SHORT, and CLEAR instruction for safe navigation.
        Focus on immediate obstacles and the clearest path forward.
        Use clock-face directions (e.g., "chair at 2 o'clock") when useful.
        Your response must be a direct command.

        Examples:
        - "Clear path ahead. Walk forward."
        - "Obstacle detected. Proceed with caution."
        - "Stairs ahead. Stop."
        - "Door is slightly to your left."
        - "Table at your 1 o'clock. go right."
        """

        while self.is_running:
            # Check if it's time to generate a new instruction (e.g., every 4 seconds)
            if (time.time() - self.last_spoken_time > 4) and self.latest_frame is not None:
                self.last_spoken_time = time.time()
                try:
                    self.update_status_label("Analyzing scene...")
                    # Convert frame to PIL Image for the API
                    pil_image = Image.fromarray(cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB))
                    
                    response = self.gemini_model.generate_content(
                        [nav_prompt, pil_image],
                        request_options={'timeout': 20} # Timeout after 20 seconds
                    )
                    advice = response.text.strip().replace("*", "") # Clean up response

                    self.update_status_label(f"AI Cue: {advice}")
                    self.speak(advice)

                except Exception as e:
                    print(f"[Gemini Error]: {e}")
                    self.update_status_label("AI Error: Could not get navigation cue.")
            
            # Sleep briefly to prevent high CPU usage
            time.sleep(0.5)

    def speak(self, text):
        """Speaks the given text using the TTS engine."""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"[TTS Error]: {e}")

    def update_status_label(self, text):
        """Safely updates the GUI label from any thread."""
        # Use 'after' to schedule the GUI update on the main thread
        self.window.after(0, lambda: self.ai_status_label.config(text=text))


# --- 3. MAIN ENTRY POINT ---
if __name__ == "__main__":
    root = tk.Tk()
    app = WalkPathNavApp(root, "AI Indoor Navigation Assistant")
    root.mainloop()
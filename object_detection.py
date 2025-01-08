import cv2
from ultralytics import YOLO
import numpy as np
import sys
import time

def main():
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    
    print("Attempting to access camera...")
    cap = cv2.VideoCapture(0)
    time.sleep(1)  # Give time for camera initialization
    
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        print("Please make sure you have granted camera permissions to the terminal/Python.")
        print("You may need to go to System Settings -> Privacy & Security -> Camera (on MacOS, at least)")
        sys.exit(1)
    
    print("Camera accessed successfully. Starting object detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
            
        try:
            results = model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow('YOLOv8 Object Detection', annotated_frame)
            
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            break
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
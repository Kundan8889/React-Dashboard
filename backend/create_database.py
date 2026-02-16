import cv2 as cv
import os
# temp = 0
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

video_path = "capture_data.mp4"
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video")
    exit()

def create_data(name):
    class_dir = os.path.join(DATA_DIR, str(name))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    print(f"Collecting data for {name}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to capture frame.")
            break

        if frame_count % 3 == 0:  # Save every 3rd frame
            file_path = os.path.join(class_dir, f'{saved_count}.jpg')
            cv.imwrite(file_path, frame)
            saved_count += 1
            cv.imshow('Saved Frame', frame)

        frame_count += 1

        # Optional: exit early by pressing 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user.")
            break

    print(f"Saved {saved_count} images.")

create_data("Birendar")

cap.release()
cv.destroyAllWindows()

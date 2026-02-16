import os 
import cv2 as cv

VIDEO_DIR = 'videos'
DATA_DIR = 'data'
SAVE_EVERY_N = 3

def extract_frames(video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    person_dir = os.path.join(DATA_DIR,name)
    os.makedirs(person_dir, exist_ok=True)
    
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    
    frame_count = 0
    saved_count = 0
    

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            break

        if frame_count % SAVE_EVERY_N == 0:
            filename = os.path.join(person_dir, f'{saved_count}.jpg')
            cv.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"Saved {saved_count} images to {person_dir}")




def main():
    if not os.path.exists(VIDEO_DIR):
        print(f"Video directory not found: {VIDEO_DIR}")
        return
    
    videos = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4','.avi','.mov','.mkv'))]
    if not videos:
        print(f"No video files found in {VIDEO_DIR}")
        return
    for video in videos:
        extract_frames(os.path.join(VIDEO_DIR, video))


if __name__ == "__main__":
    main()

import cv2 as cv
import os

myvideo = input("Enter your name : ")
output_path = r"C:\nexustechinnov-facial_attendence_management\videos"
os.makedirs(output_path, exist_ok=True) 
output_file = os.path.join(output_path, f"{myvideo}.mp4")




    


fps = 20

frame_width = 640
frame_height = 480

cap = cv.VideoCapture(0)

cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)


fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

print("Recording")
print("Press 'q' to stop")

while True:
    ret ,frame = cap.read()
    if not ret:
        break
    
    out.write(frame)
    cv.imshow('Recording', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break   


cap.release()
out.release()
cv.destroyAllWindows()
print(f'saved: {output_file}')
import cv2

# Trained classifier (only works with grayscale img)
face_cascade = cv2.CascadeClassifier(
    r'C:\Users\DELL\Documents\Python\Face_detection_using_OpenCV\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    if(img is not None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

    cv2.imshow('Face Detection via Webcam feed', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

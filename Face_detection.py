import cv2

# Trained classifier (only works with grayscale img)
face_cascade = cv2.CascadeClassifier(
    r'C:\Users\DELL\Documents\Python\Face_detection_using_OpenCV\haarcascade_frontalface_default.xml')

path = r'C:\Users\DELL\Documents\Python\Face_detection_using_OpenCV\images.jfif'
# Read the input image
img = cv2.imread(path)
print(img)
# or, to convert image 2 gray we can simply do --> img = cv2.imread(path,0)
if(img is not None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
res_gray = cv2.resize(gray, (0, 0), fx=1, fy=1)
res_img = cv2.resize(img, (0, 0), fx=1, fy=1)

faces = face_cascade.detectMultiScale(res_gray, 1.1, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(res_img, (x, y), (x+w, y+h), (255, 0, 0), 3)

cv2.imshow('img', res_img)
cv2.waitKey()

import cv2 as cv

face_Cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

img = cv.imread('Phoyos/cat.jpg')
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_Cascade.detectMultiScale(imgGray, 1.1, 4)

for (x, y, w, h) in faces:
  cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv.imshow("Result", img)
cv.waitKey(0)
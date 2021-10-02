import cv2 as cv

#img  = cv.imread('Photos/cat.jpg')

#cv.imshow('Cat', img)
#cv.waitKey(0)



capture = cv.VideoCapture(0)
# 0 is for webcam currently connected

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()


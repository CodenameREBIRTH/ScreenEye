import numpy as np
import cv2 as cv
import time
import pyttsx3

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

def distance_from_camera(actualwidth, focallength, apparentwidth):
    return (actualwidth * focallength) / apparentwidth

actualwidth = 6
actualdistance = 12

capture_video = cv.VideoCapture(0)
time.sleep(1)

while True:
    ret, frame = capture_video.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
    )

    for(x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        focallength = (w * actualdistance) / actualwidth
        dist = distance_from_camera(actualwidth, focallength, w)
        cv.putText(frame, str(dist), (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv.LINE_AA)

        if dist < 12:
            speech = pyttsx3.init()
            speech.say('Move head away from screen')
            speech.runAndWait()
            speech.stop()

    cv.imshow('result', frame)
    if cv.waitKey(10) == ord('q'):
        break

capture_video.release()
cv.destroyAllWindows()
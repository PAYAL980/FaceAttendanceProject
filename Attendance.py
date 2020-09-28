import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'StoredList'
images = []
names = []
myClass = os.listdir(path)

for student in myClass:
    currentImg = cv2.imread(f'{path}/{student}')
    images.append(currentImg)
    names.append(os.path.splitext(student)[0])
print(names)


def findEncodings(image):
    encodedList = []
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodedList.append(encode)
    return encodedList


encodeList = findEncodings(images)

cap = cv2.VideoCapture(0)


def markAttendance(name):
    with open('AttendanceList.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateTimeString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateTimeString}')


while True:
    res, frame = cap.read()
    frameScaled = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    frameScaled = cv2.cvtColor(frameScaled, cv2.COLOR_BGR2RGB)

    faceInFrame = face_recognition.face_locations(frameScaled)
    encodingsInFrame = face_recognition.face_encodings(frameScaled, faceInFrame)

    for encodeFace, faceLoc in zip(encodingsInFrame, faceInFrame):
        match = face_recognition.compare_faces(encodeList, encodeFace)
        faceDistance = face_recognition.face_distance(encodeList, encodeFace)

        matchIndex = np.argmin(faceDistance)

        if match[matchIndex]:
            name = names[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 220), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 220, 220), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam Pic', frame)
    cv2.waitKey(1)

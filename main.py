import cv2
import numpy as np
import face_recognition

imgSundar = face_recognition.load_image_file('imagesMain/SundarPichai.jpg')
imgSundar = cv2.cvtColor(imgSundar, cv2.COLOR_BGR2RGB)

imgTst = face_recognition.load_image_file('imagesMain/SundarPichaiTst.jpg')
imgTst = cv2.cvtColor(imgTst, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgSundar)[0]
encodeSundar = face_recognition.face_encodings(imgSundar)[0]
cv2.rectangle(imgSundar, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTst = face_recognition.face_locations(imgTst)[0]
encodeTst = face_recognition.face_encodings(imgTst)[0]
cv2.rectangle(imgTst, (faceLocTst[3], faceLocTst[0]), (faceLocTst[1], faceLocTst[2]), (255, 0, 255), 2)

result = face_recognition.compare_faces([encodeSundar], encodeTst)
print(result)

cv2.imshow('Sundar Pichai', imgSundar)
cv2.imshow('Sundar Pichai Test', imgTst)
cv2.waitKey(0)

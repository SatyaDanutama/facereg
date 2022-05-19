import cv2
import numpy
import os
import face_recognition


imgcoba = face_recognition.load_image_file('peserta/cropp.jpg')
imgcoba = cv2.cvtColor(imgcoba, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Peserta/Elon.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgcoba)[0]
encodeSat = face_recognition.face_encodings(imgcoba)[0]
cv2.rectangle(imgcoba, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeSat],encodeTest)
print(results)


#results = face_recognition.compare_faces([])

cv2.imshow('cropp',imgcoba)
cv2.imshow('crop1',imgTest)
cv2.waitKey(0)
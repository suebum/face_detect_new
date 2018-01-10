import cv2
import sys

imagePath = sys.argv[1]
cascPath = "haarcascade_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


img = cv2.imread(imagePath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_no = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(30, 30))

print("Successfully Found {0} unique face(s)".format(len(face_no)))

for (x, y, w, h) in face_no:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Result of Face Detection", img)
cv2.waitKey(0)

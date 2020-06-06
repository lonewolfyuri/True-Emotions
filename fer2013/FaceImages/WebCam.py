import cv2


def FaceRecognition(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find faces
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(48, 48)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img

cap = cv2.VideoCapture(0)
while(True):
  #Capture images
  ret, frame = cap.read()
  #Display
  cv2.imshow('Web camera', FaceRecognition(frame))
  #Click q to close
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
#Release camera
cap.release()
cv2.destroyAllWindows()
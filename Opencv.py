import cv2

face_Classifier = cv2.CascadeClassifier(r"C:\project\Facial Emotion Detection\haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

while True:
    ret, frame = cap.read()

   # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_Classifier.detectMultiScale(frame, 1.1, 3,minSize=(30, 30))

    for (x, y, w, h) in faces:
    
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,200, 0), 3) 

    cv2.imshow("face_detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2

model = cv2.face.LBPHFaceRecognizer_create()
model.read("model.xml")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

names = ["Kathirvel", "Person 1","Person 2"]  
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        
        label, confidence = model.predict(face)
        
        cv2.putText(frame, names[label], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
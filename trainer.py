import cv2
import os
import numpy as np

data_path = 'dataset'
faces = []
labels = []
names = {}
label_id = 0

for Kathir in os.listdir(data_path):
    person_path = os.path.join(data_path, Kathir)
    
    names[label_id] = Kathir
    
    for image in os.listdir(person_path):
        img_path = os.path.join(person_path, image)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        faces.append(img)
        labels.append(label_id)
    
    label_id += 1

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, np.array(labels))

model.save("model.xml")

print("Training Complete!")
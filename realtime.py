import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

model = load_model("food_model.h5")
class_names = os.listdir("food-101/images")

IMG_SIZE = 224

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[100:400, 100:400]
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    pred = model.predict(img)
    class_id = np.argmax(pred)
    label = class_names[class_id]

    cv2.rectangle(frame, (100,100), (400,400), (0,255,0), 2)
    cv2.putText(frame, label, (100,90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Food Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

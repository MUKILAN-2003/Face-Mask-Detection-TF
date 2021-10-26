import numpy as np
from mtcnn import MTCNN
import cv2
import tensorflow as tf

detector = MTCNN()

video_capture = cv2.VideoCapture(2)

faceCascade = cv2.CascadeClassifier(
    './haarcascades/haarcascade_frontalface_alt2.xml')

model = tf.keras.models.load_model("./model/MaskDetection.h5")

font = cv2.FONT_HERSHEY_SIMPLEX

while (True):
    ret, frame = video_capture.read()
    '''
    boxes = detector.detect_faces(frame)
    if boxes:
        box = boxes[0]['box']
        conf = boxes[0]['confidence']
        x, y, w, h = box[0], box[1], box[2], box[3]
        if conf > 0.5:
            mtcnn_face = frame[y:y + h, x:x + w]
            m_resize_to_pred = cv2.resize(mtcnn_face, (224, 224))
            print(model.predict(np.asarray(
                m_resize_to_pred).reshape(-1, 224, 224, 3)).round())
            cv2.imshow("mtcnn_face", m_resize_to_pred)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(
        64, 64), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        harcascade_face = frame[y:y + h, x:x + w]
        c_resize_to_pred = cv2.resize(harcascade_face, (224, 224))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        pred = model.predict(np.asarray(
            c_resize_to_pred).reshape(-1, 224, 224, 3)).round(decimals=3)
        print(pred)
        if (pred[0][2] > 0.2):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, 'Wear Mask', (x, y - 15),
                        font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        elif (pred[0][0] > 0.2 and pred[0][1] < 0.75):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, 'Wear Mask Properly', (x, y - 15),
                        font, 1, (0, 255, 255), 2, cv2.LINE_AA)

        elif (pred[0][1] > 0.98):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Mask Weared', (x, y - 15),
                        font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

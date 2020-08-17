import cv2
import numpy as np
from joblib import load

from sklearn.decomposition import PCA
from sklearn.svm import SVC

from time import time

HAAR_MODEL = 'D:\Rubina\AU 4th sem\ML\project\public\python\model-haar\haarcascade_frontalface_default.xml'
ANN_MODEL = 'D:\Rubina\AU 4th sem\ML\project\public\python\model-ann\gender-classify-v1.lib'

font1 = cv2.FONT_HERSHEY_TRIPLEX
font2 = cv2.FONT_HERSHEY_SIMPLEX
color = (205,55,0)

detector = cv2.CascadeClassifier(HAAR_MODEL)
classifier = load(ANN_MODEL)

capture = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 5, (1280,720))

start = time()
clock = time()
dic = {}
dic['[\'no\']'] = 0
max = 0
marked_key = ''

while True:
    clock = time()
    ret, frame = capture.read()
    resize = cv2.resize(frame, (1280,720), fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
    out.write(resize)
    image = resize.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)


    #cv2.putText(image, 'Welcome to AU Library!', (96, 106), font1, 1.6, (0,0,255), thickness=2)
    if clock-start < 3:
        for (x, y, w, h) in faces:
            testset = []
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face,(256,256), interpolation=cv2.INTER_LINEAR)
            testset.append(np.ravel(face_resized))

            pca = load('D:\Rubina\AU 4th sem\ML\project\public\python\model-ann\pca.lib')
            features = pca.transform(testset)

            pred = str(classifier.predict(features))
            prob = classifier.predict_proba(features)
            max_prob = 0
            for p in prob[0]:
                if p > max_prob:
                    max_prob = p
            threshold = 0.3

            if max_prob >= threshold:
                if (pred in dic) is False:
                    dic[pred] = 1
                else:
                    dic[pred] = dic[pred] +1 
                # # welcome = 'Welcome '.join(pred)
                # text = ''.join(pred[1:-1])
                # cv2.putText(image, 'Welcome to AU Library ' + (pred)[2:-2] + '!', (10, 106), font1, 2, (205,55,0), thickness=3)
                # # cv2.putText(image, text, (x,y-10), font2, 1, color, thickness=3)
            else:
                dic['[\'no\']'] = dic['[\'no\']'] + 1
                # cv2.putText(image, 'Not identified', (x,y-10), font2, 1, color, thickness=3) #cv2.rectangle(image, (x,y), (x+w,y+h), color, 2)
    else:
        for key,value in dic.items():
            if dic[key] > max:
                max = dic[key]
                marked_key = key
        result = marked_key[2:-2]

        if result == 'no':
            cv2.putText(image, 'Not identified', (x,y-10), font2, 1, color, thickness=3)
        else:
            cv2.putText(image, 'Welcome to AU Library ' + result + '!', (10, 106), font1, 2, (205,55,0), thickness=3)

    cv2.imshow('face classifier', image)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        key = cv2.waitKey(2000)
        break
capture.release()
cv2.destroyAllWindows()

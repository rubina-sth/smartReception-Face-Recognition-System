import cv2
import numpy as np
import pandas as pd
from joblib import dump, load
from pathlib import Path
 
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
 
HAAR_MODEL = 'D:\Rubina\AU 4th sem\ML\project\public\python\model-haar\haarcascade_frontalface_default.xml'

mainDir = 'D:\Rubina\AU 4th sem\ML\project\public'

INPUT_IMAGE_PATH = 'D:\Rubina\AU 4th sem\ML\project\public\images'

OUTPUT_CSV_FILE = 'D:\Rubina\AU 4th sem\ML\project\public\extra\images.csv'

PROCESSED_IMAGE_PATH = 'D:\Rubina\AU 4th sem\ML\project\public\extra\prc-faces'

PROCESSED_CSV_FILE = 'D:\Rubina\AU 4th sem\ML\project\public\extra\prc-faces.csv'

DETECTED_FACE_PATH = 'D:\Rubina\AU 4th sem\ML\project\public\extra\crp-faces'

DETECTED_CSV_FILE = 'D:\Rubina\AU 4th sem\ML\project\public\extra\crp-faces.csv'

OUTPUT_MODEL_NAME = 'D:\Rubina\AU 4th sem\ML\project\public\python\model-ann\gender-classify-v1.lib'

 
def create_csv(dataset_path, output_csv):
    root_dir = Path(dataset_path)
    items = root_dir.iterdir()
 
    filenames = []
    labels = []
    #print('reading image files ... ')
    for item in items:
        if item.is_dir():
            for file in item.iterdir():
                if file.is_file():
                    print(str(file))
                    filenames.append(file)
                    labels.append(item.name)
    raw_data = {'filename': filenames, 'label': labels}
    df = pd.DataFrame(raw_data, columns=['filename','label'])
    df.to_csv(output_csv)
    print(len(filenames), 'image file(s) read')
    #input("Press [ENTER] key to continue...")
 
def resize(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None:
        r = height/float(h)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))
    return cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
 
def process_image(input_csv, output_csv, output_path_name):
    dataset = pd.read_csv(input_csv, sep=',')
    ids = dataset.values[:,0]
    names = dataset.values[:,1]
    labels = dataset.values[:,2]
 
    output_path = Path(output_path_name)
    if not output_path.exists():
        output_path.mkdir()
 
    filenames = []
    #print('preprocessing images ... ')
    for item in names:
        input_path = Path(item)
        if input_path.is_file():
            output_name = output_path_name + '/image' + str(ids[len(filenames)]) + input_path.suffix
            #print(input_path, '->', output_name)
            image = cv2.imread(str(input_path))
            image = resize(image, width=256)
            cv2.imwrite(output_name, image)
            filenames.append(output_name)
    prc_data = {'filename': filenames, 'label': labels}
    df = pd.DataFrame(prc_data, columns=['filename', 'label'])
    df.to_csv(output_csv)
    # print(len(filenames), 'image file(s) processed')
    #input("Press [ENTER] key to continue...")
 
def detect_face(input_csv, output_csv, output_path_name):
    dataset = pd.read_csv(input_csv, sep=',')
    ids = dataset.values[:,0]
    names = dataset.values[:,1]
    labels = dataset.values[:,2]
 
    output_path = Path(output_path_name)
    if not output_path.exists():
        output_path.mkdir()
 
    clf = cv2.CascadeClassifier(HAAR_MODEL)
    face_filenames = []
    face_labels = []
    count = 0
    face_count = 0
    #print('detecting faces ... ')
    for item in names:
        image = cv2.imread(item)
        face_label = labels[count]
 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = clf.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cropped = image[y:y+h, x:x+w]
            output_file = output_path_name + '/face' + str(len(face_filenames)) + '.jpg'
            cv2.imwrite(output_file, cropped)
 
            face_filenames.append(output_file)
            face_labels.append(face_label)
        #print(item, '->', len(faces), ' face(s) detected')
        face_count += len(faces)
        count += 1
    crp_data = {'filename': face_filenames, 'label': face_labels}
    df = pd.DataFrame(crp_data, columns=['filename', 'label'])
    df.to_csv(output_csv)
    print('Total of', face_count, 'face(s) detected')
    #input("Press [ENTER] key to continue...")
 
def train_model(train_csv, output_model_name):
    dataset = pd.read_csv(train_csv, sep=',')
    ids = dataset.values[:,0]
    names = dataset.values[:,1]
    labels = dataset.values[:,2]
 
    images = []
    for item in names:
        image = cv2.imread(str(item), 0)
        resized = cv2.resize(image, (256,256), interpolation=cv2.INTER_LINEAR)
        images.append(np.ravel(resized))
    # print('Training recognition model ...')
 
    pca = PCA()
    features = pca.fit_transform(images)
    dump(pca, 'D:\Rubina\AU 4th sem\ML\project\public\python\model-ann\pca.lib')
 
    # plt.scatter(features[:, 0], features[:, 1], c=labels, s=30, cmap=plt.cm.Paired)
    # plt.show()
 
 
    clf = MLPClassifier(solver='lbfgs', activation='logistic', alpha=0.1, hidden_layer_sizes=(5,2), random_state=1)
    clf.fit(features, labels)
    dump(clf, output_model_name)
 
    print('Model created in', output_model_name)
    #input("Press [ENTER] key to continue...")
 
 
def validate_model(validate_csv, model_name):
    dataset = pd.read_csv(validate_csv, sep=',')
    ids = dataset.values[:,0]
    names = dataset.values[:,1]
    labels = dataset.values[:,2]
 
    images = []
    #print('Validating recognition model ...')
    for item in names:
        image = cv2.imread(str(item), 0)
        resized = cv2.resize(image, (256,256), interpolation=cv2.INTER_LINEAR)
        images.append(np.ravel(resized))
 
    pca = load('D:\Rubina\AU 4th sem\ML\project\public\python\model-ann\pca.lib')
    features = pca.transform(images)
    clf = load(model_name)
    y_p = cross_val_predict(clf, features, labels, cv=3)
    print('Accuracy Score:', '{0:.4g}'.format(accuracy_score(labels,y_p) * 100), '%')
    print('Confusion Matrix:')
    print(confusion_matrix(labels,y_p))
    print('Classification Report:')
    print(classification_report(labels,y_p))
 
create_csv(INPUT_IMAGE_PATH, OUTPUT_CSV_FILE)
process_image(OUTPUT_CSV_FILE, PROCESSED_CSV_FILE, PROCESSED_IMAGE_PATH)
detect_face(PROCESSED_CSV_FILE, DETECTED_CSV_FILE, DETECTED_FACE_PATH)
train_model(DETECTED_CSV_FILE, OUTPUT_MODEL_NAME)
validate_model(DETECTED_CSV_FILE, OUTPUT_MODEL_NAME)

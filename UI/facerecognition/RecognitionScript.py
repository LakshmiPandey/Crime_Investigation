# importing the libraries
import os
import cv2
import numpy as np



def sample():
   #  take the image input
   image = cv2.imread('image.jpg')

    # function call to predict the image
    # returning the image folder path as image_label
   image_label, predicted_image = predict(image)
   return image_label, predicted_image

subjects = []

# function to detect the face in the image
def detection( img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## face classifier used here ----------------- is lbpcascade
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]



#function to draw rectangle on image
#according to given (x, y) coordinates and
#given width and heigh
def draw_rectangle(img, rect):
 (x, y, w, h) = rect
 cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# training the model
def prepare_training_data(data_folder_path):
    i = -1
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        i = i + 1
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        subjects.append(subject_dir_path)
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            image_path = os.path.join(subject_dir_path, image_name)
            image = cv2.imread(image_path)
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(5)
            face, rect = detection(image)
            if face is not None:
                faces.append(face)
                labels.append(i)
    return faces, labels




# function to predict the new face
def predict(test_img):

    img = test_img.copy()
    face, rect = detection(img)
    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label]
    draw_rectangle(img, rect)
    return label_text, img

## preparing the data
faces, labels = prepare_training_data("Data_folder_path")


## For face recognition weare using the LBPH Face Recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

# function call to recognise the face
image_label, predicted_image = sample()
cv2.imshow('IMAGE', predicted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

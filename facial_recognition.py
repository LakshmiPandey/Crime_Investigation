import cv2
import numpy as np

def take_sample():
   
   #  take the image input
   image = raw_input("enter the name of the image file: ")
   
   # function call to predict the image
   image_label, predicted_image = predict(image)
   return image_label, predicted_image


def detection( img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
    ## face classifier used here -----------------  
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

# training the model
def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            face, rect = detection(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
    
    return faces, labels



def predict(test_img):

    img = test_img.copy()
    face, rect = detection(img)
    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label]
    draw_rectangle(img, rect)
    return label_text, img



## preparing the data 
faces, labels = prepare_training_data("training-data")
##("Data prepared")
# total faces and labels
#("Total faces: ", len(faces))
#("Total labels: ", len(labels))


## For face recognition weare using the LBPH Face Recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

image_label, predicted_image = take_sample()
cv2.waitKey(0)
cv2.destroyAllWindows()

limport cv2
import numpy as np

#face detection classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def detection( img1):
   faces = face_cascade.detectMultiScale(img1, 1.3, 5)
   for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            img1 = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            new_image = cv2.resize(gray1, (0,0), fx = 1.5,  fy=1.5)
            return(new_image)

#rtraining the model
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
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels

##print("Preparing data...")
faces, labels = prepare_training_data("training-data")
##print("Data prepared")
#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

## For face recognition weare using the LBPH Face Recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    face, rect = detection(img)
    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label]
    draw_rectangle(img, rect)
    return img




#sample image to be recognize
gray = cv2.imread('myee.jpg', 0)
img3 = detection(gray)
predicted_image = predict(img3)
cv2.imshow('sample',predicted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

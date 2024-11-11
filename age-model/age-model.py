from huggingface_hub import snapshot_download
from huggingface_hub import login
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.image import rgb_to_grayscale
import cv2
import numpy as np 
from keras_preprocessing.image import img_to_array
import os

os.environ["HF_HOME"] = r"main_folder"

login("token")

repo_path = snapshot_download(repo_id="Dhrumit1314/Live_Face_Detection", cache_dir=r"main_folde")

#repo_path = snapshot_download(repo_id="Dhrumit1314/Live_Face_Detection")

face_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

age_model = os.path.join(repo_path, "age_model_3epochs.h5")

custom_objects = {"mse": MeanSquaredError()}
age_model = load_model(age_model , custom_objects=custom_objects)

cap = cv2.VideoCapture(0)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
fps = cap.get(cv2.CAP_PROP_FPS)
i=0
j=0
pred = None
while cap.isOpened():
  ret, frame = cap.read()

  if ret == False:
    cap.release()
    break
  gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  faces=face_classifier.detectMultiScale(gray,1.3,5)

  for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        # Get image ready for prediction
        roi=roi_gray.astype('float')/255.0  # Scaling the image
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)  # Expand dims to get it ready for prediction (1, 48, 48, 1)
        roi_color=frame[y:y+h,x:x+w]
        roi_color=cv2.resize(roi_color,(200,200),interpolation=cv2.INTER_AREA)
        # Age
        age_predict = age_model.predict(np.array(roi_color).reshape(-1,200,200,3))
        age = round(age_predict[0,0])
        age_label_position=(x+h,y+h)
        cv2.putText(frame,"Age="+str(age),age_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Imagen con Texto', frame) 
        cv2.imwrite(f'gif/im_{j}.jpg', frame)
        j += 1

  if cv2.waitKey(27) & 0xFF == ord('q') or cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) < 1:
      cap.release()
      cv2.destroyAllWindows()
      break
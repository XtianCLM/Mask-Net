
from unicodedata import category
import streamlit as st
import os
import cv2
import imghdr
import numpy as np
from skimage.feature import hog
from skimage import exposure
from skimage.transform import resize
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import dlib

detector =  dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('dlibLM/shape_predictor_68_face_landmarks.dat')


@st.cache(hash_funcs={cv2.dnn_Net: hash})
def load_face_detector_and_model():
    prototxt_path = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weights_path = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    cnn_net = cv2.dnn.readNet(prototxt_path, weights_path)

    return cnn_net


@st.cache(allow_output_mutation=True)
def load_cnn_model():
    cnn_model = load_model("mask_detector.model")

    return cnn_model
##END OF model setup

##START OF initialization of the model aand other variable
net = load_face_detector_and_model()
model = load_cnn_model()
data_dir = 'demo'
image_exts = ['jpg','jpeg', 'bmp', 'png']

index_Value = 0

for Folder_path in os.listdir(data_dir):
    for Folder in os.listdir(os.path.join(data_dir, Folder_path)):
        for images in os.listdir(os.path.join(data_dir, Folder_path, Folder)):
            index_Value = index_Value + 1
            image_path = os.path.join(data_dir, Folder_path, Folder, images)
            try:
                    
                file_ext = os.path.splitext(images)[1]
                rename_img = '0'+str(index_Value)+'_'+Folder +file_ext 
                rename_path = os.path.join(data_dir, Folder_path, Folder)
                path = os.path.join(rename_path, rename_img)
                os.rename(image_path, path)
               

                img = cv2.imread(path)
                tip = imghdr.what(path)
            

                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                orig = image.copy()
                (h, w) = image.shape[:2]
                blob = cv2.dnn.blobFromImage(image, 1.0, (100, 100),
                                        (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()
            

                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        (startX, startY) = (max(0, startX), max(0, startY))
                        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                        face = img[startY:endY, startX:endX]
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face = cv2.resize(face, (224, 224))
                        face = img_to_array(face)
                        face = preprocess_input(face)
                        expanded_face = np.expand_dims(face, axis=0)
                        
                        (mask, withoutMask) = model.predict(expanded_face)[0]
          
                        if mask > withoutMask:
                            label = "Mask"
                        else:
                            label = "No Mask"
                        if label == "Mask":
        
                            dlib_rect = dlib.rectangle(startX, startY, endX, endY)

                            landmarks = predictor(image,dlib_rect)



                            e = landmarks.part(1).x
                            e1 = landmarks.part(26).x
                            i =  landmarks.part(20).y
                            i1 = landmarks.part(29).y
                       
                            
                         
                            
                            roi = image[i-10: i1, e: e1+5]
                            
                            roi_image = np.array(roi) 
                            
                        
                            cv2.imwrite(path, roi_image)
                       
                        
                        elif label == "No Mask":
                            os.remove(path)
                        else:
                            os.remove(path)

                if tip not in image_exts:
                    st.write('Image not in ext list')
                    os.remove(path)

            except Exception as e:
                st.write(e)





    





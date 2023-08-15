
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import exposure
from skimage.transform import resize
import json
from keras.optimizers import Adam, SGD
from keras.models import model_from_json
import requests  # pip install requests
from streamlit_lottie import st_lottie
import time
import dlib

detector =  dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('dlibLM/shape_predictor_68_face_landmarks.dat')

def Show_User_Testing():
    # with open('style.css') as f:
    #     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    json_file = open('Model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("Model/model.h5")


    optimizers = Adam(learning_rate=1e-4,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=1e-07)


    loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers,  metrics=['accuracy'])



    
    app_title = 'User Testing'
    with st.spinner("📝Now loading {}".format(app_title)):
        time.sleep(1)
        
        def load_lottieurl(url: str):
                r = requests.get(url)
                if r.status_code != 200:
                    return None
                return r.json()

        lottie_scan = " https://assets5.lottiefiles.com/packages/lf20_2z4qmxlj.json"
        lottie_face_scan = load_lottieurl(lottie_scan)
            

        

        ##START OF model setup
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

        ##START OF information or intro
    
        ##END OF information or intro

        ##START OF initialization of the model
        net = load_face_detector_and_model()
        model = load_cnn_model()
    ##END OF initialization of the model
        
    ##START OF global variables and sidebar inputs
        col1,col2 = st.columns([3,0.5])
        with col1:
            uploaded_image = st.file_uploader("Choose a JPG file", type="jpg")
            st.markdown("<p style='color:#0E2332; border-top: 1px solid #0E2332;'></p>",unsafe_allow_html=True)
            st.markdown("<p style='color:#0E2332;'>Try Live Feed</p>",unsafe_allow_html=True)
            run = st.checkbox('Start Video Capture')
            FRAME_WINDOW = st.image([])
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    
        ##END OF global variables and sidebar inputs

    

        if uploaded_image:

            with col2:

               
                
                st.image(uploaded_image, use_column_width=True)
                st.info('Uploaded image')

            ##START OF setup such as uploading image and declaring variables
            img = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
            image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            orig = image.copy()
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (100, 100),
                                        (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            ##END OF setup such as uploading image and declaring variables

            ##START OF face detection and determining the x and y value of the image and also the width and the height
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.20:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    face = image[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    expanded_face = np.expand_dims(face, axis=0)
                        
                    (mask, withoutMask) = model.predict(expanded_face)[0]
            ##END OF face detection and determining the x and y value of the image and also the width and the height    

                    ##START OF selecting region of interest and detecting eye inside region of interest
                    if mask > withoutMask:
                        label = "Mask"
                    else:
                        label = "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
                    if label == "Mask":
                        
                        Continues = st.button("Continue")
                        if Continues:
                            r_col1, r_col2, r_col3 = st.columns([2,2,2])
                            with r_col1:
                                st.markdown("<p style='font-size:1.3rem;color:#3B6978;text-align: center; font-weight:700;'>Upload Image<p>",unsafe_allow_html=True)
                            with r_col2:
                                st.markdown("<p style='font-size:1.3rem;color:#3B6978;text-align: center; font-weight:700;'>Face Detection<p>",unsafe_allow_html=True)
                            with r_col3:
                                st.markdown("<p style='font-size:1.3rem;color:#3B6978;text-align: center; font-weight:700;'>Prediction<p>",unsafe_allow_html=True)
                            my_bar = st.progress(0)
                            percent_complete = 0
                            my_bar.progress(percent_complete + 30)
                            
                            cv2.putText(image, label, (startX, startY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)

                            cv2.rectangle(image, (startX, startY), (endX,endY), color, 2)
                            
                            dlib_rect = dlib.rectangle(startX, startY, endX, endY)

                            landmarks = predictor(image,dlib_rect)

                            e = landmarks.part(1).x
                            e1 = landmarks.part(26).x
                            i =  landmarks.part(20).y
                            i1 = landmarks.part(29).y

                            roi = image[i-10: i1, e: e1+5]

                            
                            scol1, scol2, scol3,scol4,scol5 = st.columns([0.1,0.7,0.7,1,0.1])    
                            with scol2:
                                time.sleep(0.6)
                                st.markdown("<p style='border:1px solid red;height:114px;visibility:hidden;'>this is invisible<p>",unsafe_allow_html=True)
                                st.info("Face Detected:")
                                st.image(image, use_column_width=True)
                            with scol3:
                                time.sleep(0.5)
                                st.markdown("<p style='border:1px solid red;height:130px;visibility:hidden;'>this is invisible<p>",unsafe_allow_html=True)
                                st.image('images/Arrow_Right.png')
                            with scol4:
                                time.sleep(0.5)
                                st.markdown("<p style='border:1px solid red;height:114px;visibility:hidden;'>this is invisible<p>",unsafe_allow_html=True)
                                st.info("Region of Interest:")
                                st.image(roi, use_column_width=True)


                            emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4:"Surprise"}
                            target_names = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
                            
                            resize = cv2.resize(roi, (64, 64))
                            prediction = loaded_model.predict(np.expand_dims(resize/255, 0))
                            max_index = int(np.argmax(prediction))

                            col1, col2 = st.columns([1,1])

                            with col1:
                                st.info(emotion_dict[max_index])
                                st.image(image,width=200)
                                                
                                                
                                print("prediction={}".format(emotion_dict[max_index]))
                                st.write(prediction.shape)
                                prob = {}
                                emo = {}
                                w = 0.4
                            for label in prediction:
                                for i in label:
                                    iprob = round(i,2)
                                    prob[i] = iprob
                                    
                                   

                                    bar1 = np.arange(len(prob))
                                    fig = plt.figure(figsize=(3,3))
                                    plt.bar(bar1, list(prob.values()),w,color='#2e38f2' )
                                    plt.xticks(range(0,5), target_names, rotation ='vertical')
                                    plt.title("Probability Distribution")
                                    plt.tripcolor
                            with col2:                
                                st.pyplot(fig)
                               
                        
                    else:
                        scol1, scol2, scol3,scol4,scol5 = st.columns([0.1,0.7,1,0.7,0.1]) 
                        roi = None
                        with scol3:
                              st.info("Upload a Picture with mask:")
                              lottie_hello = load_lottieurl("https://assets6.lottiefiles.com/private_files/lf30_hdjld2cp.json")
                              st_lottie(lottie_hello)
                        
                    ##END OF selecting region of interest and detecting eye inside region of interest    
        

        
        while run:


            ##START OF setup such as uploading image and declaring variables
            _, frame = camera.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            orig = image.copy()
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (100, 100),
                                        (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            ##END OF setup such as uploading image and declaring variables

            ##START OF face detection and determining the x and y value of the image and also the width and the height
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    face = image[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    expanded_face = np.expand_dims(face, axis=0)
                        
                    (mask, withoutMask) = model.predict(expanded_face)[0]
            ##END OF face detection and determining the x and y value of the image and also the width and the height    

                    ##START OF selecting region of interest and detecting eye inside region of interest
                    if mask > withoutMask:
                        label = "Mask"
                    else:
                        label = "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                    if label == "Mask":
                        emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4:"Surprise"}

                        cv2.putText(image, label, (startX, startY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
                            # roi_size = st.slider("Adjust the slider to the specific region of interest", 1.0, 1.8)
                        cv2.rectangle(image, (startX, startY), (endX,endY), color, 2)

                        # FRAME_WINDOW.image(image)
                            
                        dlib_rect = dlib.rectangle(startX, startY, endX, endY)

                        landmarks = predictor(image,dlib_rect)

                        e = landmarks.part(1).x
                        e1 = landmarks.part(26).x
                        i =  landmarks.part(20).y
                        i1 = landmarks.part(29).y


                        roi = image[i-10: i1, e: e1+5]

                        resize = cv2.resize(roi, (64, 64))
                        prediction = loaded_model.predict(np.expand_dims(resize/255, 0))
                        max_index = int(np.argmax(prediction))

                        if emotion_dict[max_index] == 'Angry':
                            cv2.putText(image, emotion_dict[max_index], (e, i-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                            cv2.rectangle(image, (e,i-10),(e1+5,i1), (0,0,255),1)
                            imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            FRAME_WINDOW.image(imagergb)
                        elif emotion_dict[max_index] == 'Happy':
                            cv2.putText(image, emotion_dict[max_index], (e, i-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                            cv2.rectangle(image, (e,i-10),(e1+5,i1), (0,255,255),1)
                            imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            FRAME_WINDOW.image(imagergb)
                        elif emotion_dict[max_index] == 'Neutral':
                            cv2.putText(image, emotion_dict[max_index], (e, i-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (123,63,0), 1)
                            cv2.rectangle(image, (e,i-10),(e1+5,i1), (123,63,0),1)
                            imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            FRAME_WINDOW.image(imagergb)
                        elif emotion_dict[max_index] == 'Sad':
                            cv2.putText(image, emotion_dict[max_index], (e, i-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                            cv2.rectangle(image, (e,i-10),(e1+5,i1), (255,255,0),1)
                            imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            FRAME_WINDOW.image(imagergb)
                        elif emotion_dict[max_index] == 'Surprise':
                            cv2.putText(image, emotion_dict[max_index], (e, i-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                            cv2.rectangle(image, (e,i-10),(e1+5,i1), (255,0,255),1)
                            imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            FRAME_WINDOW.image(imagergb)

                        
                    else:
                        cv2.putText(image, label, (startX, startY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
                            # roi_size = st.slider("Adjust the slider to the specific region of interest", 1.0, 1.8)
                        cv2.rectangle(image, (startX, startY), (endX,endY), color, 2)
                        imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        FRAME_WINDOW.image(imagergb)
                        
                    ##END OF selecting region of interest and detecting eye inside region of interest    
        # else:
        #     # st.write('Stopped')
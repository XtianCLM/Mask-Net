
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



def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def Start_Testing():
    #Dlib Facial Landmarks
    detector =  dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('dlibLM/shape_predictor_68_face_landmarks.dat')

    #Emotion Detection Model
    json_file = open('Model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("Model/model.h5")

    #Optimizer
    optimizers = Adam(learning_rate=1e-4,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=1e-07)

    #Load Model Compile
    loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers,  metrics=['accuracy'])

    #Loading Spinner
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
        lottiefacedetector = load_lottiefile("lottiefiles/facedetector.json")

        @st.cache(hash_funcs={cv2.dnn_Net: hash})
        def load_face_detector_and_model():
            prototxt_path = os.path.sep.join(["face_detector", "deploy.prototxt"])
            weights_path = os.path.sep.join(["face_detector",
                                            "res10_300x300_ssd_iter_140000.caffemodel"])
            cnn_net = cv2.dnn.readNet(prototxt_path, weights_path)

            return cnn_net


        @st.cache(allow_output_mutation=True)
        def load_cnn_model():
            cnn_model = load_model("face_detector/mask_detector.model")

            return cnn_model
        ##END OF model setup

        ##START OF information or intro
    
        ##END OF information or intro

        ##START OF initialization of the model
        net = load_face_detector_and_model()
        model = load_cnn_model()

        st.markdown("<h1 style='color:#00476D;'>Emotion Detection</h1>",unsafe_allow_html=True)
        rcol1,rcol2 = st.columns([0.6,1])
        with rcol1:
            Testing_Choice = st.selectbox('Select action for testing', ['File Upload', 'Live Feed'])

        if Testing_Choice == 'File Upload':
            st.markdown("<p style='height:10px;padding:10px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
            r1col1, r1col2 = st.columns([3,0.5])
            with r1col1:
                uploaded_image = st.file_uploader("Choose a JPG file", type="jpg")

            if uploaded_image:
                st.markdown("<p style='height:10px;padding:10px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
                Continues = st.button("Continue")
                st.markdown("<p style='height:10px;padding:10px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)

                with r1col2:
                    st.image(uploaded_image, use_column_width=True)
                    st.markdown("<p style='font-size:1rem;color:#3B6978;text-align:center;'>Uploaded Image<p>",unsafe_allow_html=True)

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
                            
                            if Continues:
                                r_col1, r_col2, r_col3 = st.columns([2,2,2])
                                with r_col1:
                                    st.markdown("<p style='font-size:1.3rem;color:#3B6978;text-align: center; font-weight:700;'>Face Detection<p>",unsafe_allow_html=True)
                                with r_col2:
                                    st.markdown("<p style='font-size:1.3rem;color:#3B6978;text-align: center; font-weight:700;'>Selecting Region of Interest<p>",unsafe_allow_html=True)
                                with r_col3:
                                    st.markdown("<p style='font-size:1.3rem;color:#3B6978;text-align: center; font-weight:700;'>Making Prediction<p>",unsafe_allow_html=True)
                                my_bar = st.progress(0)
                                percent_complete = 0
                               
                                
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

                                emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4:"Surprise"}
                                target_names = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
                                
                                resize = cv2.resize(roi, (64, 64))
                                prediction = loaded_model.predict(np.expand_dims(resize/255, 0))
                                max_index = int(np.argmax(prediction))
                                prob = [round(max(i),2) for i in prediction]
                                
                                
                                scol1, scol2, scol3,scol4,scol5,scol6,scol7 = st.columns([0.1,0.7,0.7,1,0.7,0.7,0.1]) 
                                with st.container():   
                                    with scol2:
                                        time.sleep(0.30)
                                        my_bar.progress(percent_complete + 30)
                                        st.markdown("<p style='border:1px solid red;height:114px;visibility:hidden;'>this is invisible<p>",unsafe_allow_html=True)
                                        st.info("Detected: " + label)
                                        st.image(image, use_column_width=True)
                                    with scol3:
                                        time.sleep(0.40)
                                        st.markdown("<p style='border:1px solid red;height:130px;visibility:hidden;'>this is invisible<p>",unsafe_allow_html=True)
                                        st.image('images/Arrow_Right.png')
                                        
                                    with scol4:
                                        time.sleep(1.0)
                                        st.markdown("<p style='border:1px solid red;height:114px;visibility:hidden;'>this is invisible<p>",unsafe_allow_html=True)
                                        st.info("Region of Interest:")
                                        st.image(roi, use_column_width=True)
                                        my_bar.progress(percent_complete + 70)
                                        
                                    with scol5:
                                        time.sleep(1.10)
                                        st.markdown("<p style='border:1px solid red;height:130px;visibility:hidden;'>this is invisible<p>",unsafe_allow_html=True)
                                        st.image('images/Arrow_Right.png')
                                    with scol6:
                                        time.sleep(1.30)
                                        st.markdown("<p style='border:1px solid red;height:110px;visibility:hidden;'>this is invisible<p>",unsafe_allow_html=True)
                                        st.info("Classifying emotion")
                                        st.image(uploaded_image,use_column_width=True)
                                        my_bar.progress(percent_complete + 100)
                                        


                                st.markdown("<p style='border:1px solid red;height:100px;visibility:hidden;'>this is invisible<p>",unsafe_allow_html=True)
                                col1, col2,col3, col4,col5= st.columns([0.3,0.6,0.1,0.6,0.3])
                               
                                with st.container():    
                                    with col2:
                                            st.markdown("<p style='border:1px solid red;height:30px;visibility:hidden;'>this is invisible<p>",unsafe_allow_html=True)
                                            for label in prediction:
                                                for index, val in enumerate(label):
                                                    st.write(emotion_dict[index]+" "+str(round(val,2))+"%")
                                                    my_bar = st.progress(0)
                                                    percent_complete = 0
                                                    my_bar.progress(percent_complete + round(val,2))
                                                    time.sleep(1.50)
                                    
                                    with col4:      
                                        st.info("The predicted emotion is: "+emotion_dict[max_index])
                                        st.image(uploaded_image,use_column_width=True)
                                           
                                            # prob = {}
                                            # w = 0.4
                                            # for label in prediction:
                                            #     for i in label:
                                            #         iprob = round(i,2)
                                            #         prob[i] = iprob
                                                    
                                                

                                            #     bar1 = np.arange(len(prob))
                                            #     fig = plt.figure(figsize=(3,3))
                                            #     plt.bar(bar1, list(prob.values()),w,color='#2e38f2' )
                                            #     plt.xticks(range(0,5),target_names, rotation ='vertical')
                                            #     plt.title("Probability Distribution")
                                            #     plt.tripcolor          
                                            # st.pyplot(fig)
                                    
                                
                            
                        else:
                            scol1, scol2, scol3,scol4,scol5 = st.columns([0.1,0.7,1,0.7,0.1]) 
                            roi = None
                            with scol3:
                                st.info("Upload a Picture with mask:")
                                lottie_hello = load_lottieurl("https://assets6.lottiefiles.com/private_files/lf30_hdjld2cp.json")
                                st_lottie(lottie_hello)











        if Testing_Choice == 'Live Feed':
            st.markdown("<p style='height:10px;padding:10px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
            space1,r3col1,r3col2,r3col3 = st.columns([.4,1.3,1.8,0.1])
            st.markdown("<p style='height:25px;padding:30px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
            with r3col1:
                st.markdown("<p style='height:25px;padding:50px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
                st.image('images/realtime.png')
                st.markdown("<h5 style='width:340px;color:#15618a;text-align:justify;position:relative;right:4.7rem;top:2rem;text-align-last:center;'>Real-time emotion detector can detect if you are wearing a mask or not, as well as your emotional state even when you are wearing a mask</h5>",unsafe_allow_html=True)

            with r3col2:
                st.markdown("<p style='font-size:1.3rem; font-weight:700;color:#3B6978; text-align:center;'>LIVE FEED</p>",unsafe_allow_html=True)
                run = st.checkbox('Start Video Capture')
                FRAME_WINDOW = st.image([],use_column_width=True)
                
                camera = cv2.VideoCapture(0)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)

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
                            prob = [round(max(i),2) for i in prediction]
                            max_index = int(np.argmax(prediction))
                            
                            if emotion_dict[max_index] == 'Angry':
                                # if prob[0] < 0.20:
                                #     continue
                                #     time.sleep(0.5)
                                cv2.putText(image, str(prob[0])+"%" , (e+50, i-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                                cv2.putText(image, emotion_dict[max_index], (e, i-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                                cv2.rectangle(image, (e,i-10),(e1+5,i1), (0,0,255),1)
                                imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                FRAME_WINDOW.image(imagergb)
                                time.sleep(0.1)
                            elif emotion_dict[max_index] == 'Happy':
                                # if prob[0] < 0.20:
                                #     continue
                                cv2.putText(image, str(prob[0])+"%" , (e+50, i-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                                cv2.putText(image, emotion_dict[max_index], (e, i-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                                cv2.rectangle(image, (e,i-10),(e1+5,i1), (0,255,255),1)
                                imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                FRAME_WINDOW.image(imagergb)
                                time.sleep(1.0)
                            elif emotion_dict[max_index] == 'Neutral':
                                # if prob[0] < 0.20:
                                #     continue
                                cv2.putText(image, str(prob[0])+"%" , (e+50, i-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (123,63,0), 1)
                                cv2.putText(image, emotion_dict[max_index], (e, i-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (123,63,0), 1)
                                cv2.rectangle(image, (e,i-10),(e1+5,i1), (123,63,0),1)
                                imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                FRAME_WINDOW.image(imagergb)
                                time.sleep(0.1)
                            elif emotion_dict[max_index] == 'Sad':
                                # if prob[0] < 0.20:
                                #     continue
                                cv2.putText(image, str(prob[0])+"%" , (e+50, i-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                                cv2.putText(image, emotion_dict[max_index], (e, i-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                                cv2.rectangle(image, (e,i-10),(e1+5,i1), (255,255,0),1)
                                imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                FRAME_WINDOW.image(imagergb)
                                time.sleep(0.1)
                            elif emotion_dict[max_index] == 'Surprise':
                                # if prob[0] < 0.20:
                                #     continue
                                cv2.putText(image, str(prob[0])+"%" , (e+60, i-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                                cv2.putText(image, emotion_dict[max_index], (e, i-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                                cv2.rectangle(image, (e,i-10),(e1+5,i1), (255,0,255),1)
                                imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                FRAME_WINDOW.image(imagergb)
                                time.sleep(0.1)

                            
                        else:
                            cv2.putText(image, label, (startX, startY - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
                                # roi_size = st.slider("Adjust the slider to the specific region of interest", 1.0, 1.8)
                            cv2.rectangle(image, (startX, startY), (endX,endY), color, 2)
                            imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            FRAME_WINDOW.image(imagergb)
            
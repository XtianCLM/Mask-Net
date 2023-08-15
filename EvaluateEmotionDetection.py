import re
from unicodedata import category
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,ConfusionMatrixDisplay
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tkinter import Image
from tensorflow import keras
from keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import json
import array
import os
from pretty_confusion_matrix import pp_matrix
from keras.utils.np_utils import to_categorical 
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_confusion_matrix
import time


def Show_My_Model():
    app_title = 'Experimentation'
    
    # load json and create model
    json_file = open('Model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
    loaded_model.load_weights("Model/model.h5")


    with st.spinner("⚗️Now loading {}".format(app_title)):
        time.sleep(1)
        emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4:"Surprise"}
        target_names = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
        


    rcol1, rcol2, rcol3 = st.columns([1,0.1,1])
   
    with rcol1:
        st.markdown("<p style='height:20px;padding:5px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
        st.title("Model Evaluation")
        choice = st.selectbox("Select Report", ['Emotions','Age', 'Gender'])


    st.markdown("<p style='height:30px;padding:15px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)


    if choice == 'Emotions':
        if os.path.exists("LmCutDb/test"):
            
            validation_ds = ImageDataGenerator(rescale=1./255)


            # This is where you should set your directory path for the dataset
            validation_gen = validation_ds.flow_from_directory(
                                'LmCutDb/test',
                                target_size= (64,64),
                                batch_size = 16,
                                color_mode = 'rgb',
                                class_mode = 'categorical'

                                )

            optimizers = Adam(learning_rate=1e-4,
                                            beta_1=0.9,
                                            beta_2=0.999,
                                            epsilon=1e-07)

            loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers,  metrics=['categorical_accuracy'])

            (eval_loss, eval_accuracy) = loaded_model.evaluate( validation_gen, batch_size=16,verbose=1)

            predictions = loaded_model.predict(validation_gen)    

            c_matrix = confusion_matrix(validation_gen.classes, predictions.argmax(axis=1))
            Acc5Emotions = c_matrix.diagonal()/c_matrix.sum(axis=1)*100
            
            col1,col2,col3 = st.columns([0.3,1,1])
            with col2:
                st.image('images/C3.png')
            rrcol1, rrcol2 = st.columns([1.5,0.6])
            with rrcol1:
                # st.markdown("<p style='height:20px;padding:20px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
                st.markdown("<p style='font-size:1.8rem;color:#3B6978;text-align: center;'>Evaluation Accuracy<p>",unsafe_allow_html=True)
                df = pd.DataFrame({"Overall Accuracy": [eval_accuracy*100],
                                    "Overall Loss": [eval_loss]
                                })
                st.dataframe(df, use_container_width=True)
                
            with rrcol2:
                a = np.array([eval_accuracy*100]) 
                b = np.array([eval_loss])
                Target_Labels = ["Accuracy", "Loss"] 
                prob = {}
                w = 0.4
                for i in a:
                    iprob = round(i,2)
                    prob[i] = iprob
                                                            
                                                        

                bar1 = np.arange(len(prob))
                bar2 = [e+w+0.5 for e in bar1]
                fig = plt.figure(figsize=(3,3))
                plt.bar(bar1, list(prob.values()),w,color='#BCDFF2')
                plt.bar(bar2, b,w,color='#EFB1B1' )
                plt.xticks(range(0,2),Target_Labels, rotation ='horizontal')
                plt.title("Bar Graph")
                plt.tripcolor          
                st.pyplot(fig)
                # st.markdown("<p style='height:50px;padding:50px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
            
                # figs, ax = plot_confusion_matrix(conf_mat=c_matrix, figsize=(3, 3), cmap=plt.cm.Blues)
                # plt.xlabel('Predictions', fontsize=8)
                # plt.xticks(range(0,5), target_names, rotation ='vertical')
                # plt.yticks(range(0,5), target_names, rotation ='horizontal')
                # plt.ylabel('Actuals', fontsize=8)
                # plt.title('Confusion Matrix', fontsize=8)
                # st.pyplot(figs) 
            st.markdown("<p style='height:10px;padding:10px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
            with st.container():
                FiveEmoLabel = [i for i in target_names]
                FiveEmoProb = [k for k in Acc5Emotions]
                st.image('images/INTERPRETATINO.png')
            #st.markdown("<p style='font-size:1.8rem;color:#3B6978; text-align: center;'>Five Emotions Accuracy<p>",unsafe_allow_html=True)
                def load_data():
                        return  pd.DataFrame({"Emotion Type":FiveEmoLabel,
                                        "Emotion Accuracy":FiveEmoProb})
                results = load_data()
                st.dataframe(results, use_container_width=True)
        else:
            rcol1, rcol2,rcol3 = st.columns([0.1,1,0.1])

            with rcol2:
                st.error("Dataset Doesn't Exist in the Directory!!!")
                st.warning("Unable to upload and set data directory path will triger this error please check your file before running this module")
                st.balloons()

           
            



# AGE GROUP

    if choice == 'Age':
        if os.path.exists("Age/AgeCombined"):
            TestAgeCombined = ImageDataGenerator(rescale=1./255)
            TestAgeYoung = ImageDataGenerator(rescale=1./255)
            TestAgeOld = ImageDataGenerator(rescale=1./255)

            # This is where you should set your directory path for the dataset
            AgeCombined = TestAgeCombined.flow_from_directory(
                                    'Age/AgeCombined',
                                    target_size= (64,64),
                                    batch_size = 16,
                                    color_mode = 'rgb',
                                    class_mode = 'categorical'

                                    )
        

            AgeYoung = TestAgeYoung.flow_from_directory(
                                    'Age/Young',
                                    target_size= (64,64),
                                    batch_size = 16,
                                    color_mode = 'rgb',
                                    class_mode = 'categorical'

                                    )


            AgeOld = TestAgeOld.flow_from_directory(
                                    'Age/Old',
                                    target_size= (64,64),
                                    batch_size = 16,
                                    color_mode = 'rgb',
                                    class_mode = 'categorical'

                                    )


            optimizers = Adam(learning_rate=1e-4,
                                                beta_1=0.9,
                                                beta_2=0.999,
                                                epsilon=1e-07)

            loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers,  metrics=['accuracy'])

            (eval_lossAge, eval_accuracyAge) = loaded_model.evaluate( AgeCombined, batch_size=16,verbose=1)
            (eval_lossYoung, eval_accuracyYoung) = loaded_model.evaluate( AgeYoung, batch_size=16,verbose=1)
            (eval_lossOld, eval_accuracyOld) = loaded_model.evaluate( AgeOld, batch_size=16,verbose=1)

            
            AgePrediction = loaded_model.predict(AgeCombined)    
            Age_c_matrix = confusion_matrix(AgeCombined.classes, AgePrediction.argmax(axis=1))
            EmotionAgeAcc = Age_c_matrix.diagonal()/Age_c_matrix.sum(axis=1)*100
            YoungPrediction = loaded_model.predict(AgeYoung)    
            Young_c_matrix = confusion_matrix(AgeYoung.classes, YoungPrediction.argmax(axis=1))
            EmotionYoungAcc = Young_c_matrix.diagonal()/Young_c_matrix.sum(axis=1)*100
            OldPrediction = loaded_model.predict(AgeOld)    
            Old_c_matrix = confusion_matrix(AgeOld.classes, OldPrediction.argmax(axis=1))
            EmotionOldAcc = Old_c_matrix.diagonal()/Old_c_matrix.sum(axis=1)*100

        
            


            a = np.array([eval_accuracyAge*100,eval_accuracyYoung*100,eval_accuracyOld*100]) 
            b = np.array([eval_lossAge,eval_lossYoung,eval_lossOld])
            Label_acc = ["Age Group", "Young", "Old"]
            Acc_Labels = [i for i in Label_acc]
            Group_Acc = [i for i in a]
            Group_Loss = [i for i in b]

            col1,col2,col3 = st.columns([0.3,1,1])
            with col2:
                st.image('images/C3.png')
        
            r2col1, r2col2 = st.columns([1.5,0.6])
        
            with st.container():
                
                with r2col1:

                    st.markdown("<p style='font-size:1.8rem;color:#3B6978; text-align: center;'>Evaluation Accuracy<p>",unsafe_allow_html=True)
                    st.markdown("<p style='height:5px;padding:5px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
                    Age_df = pd.DataFrame({"Label":Acc_Labels,
                                            "Accuracy":Group_Acc,
                                            "Loss":Group_Loss
                    })
                    st.dataframe(Age_df, use_container_width=True)
                with r2col2:
            
                    Target_Labels = ["AgeGroup", "Young", "Old"] 
                    prob = {}
                    w = 0.4
                    for i in a:
                        iprob = round(i,2)
                        prob[i] = iprob
                                                                
                                                            

                    bar1 = np.arange(len(prob))
                    bar2 = [e+w for e in bar1]
                    fig = plt.figure(figsize=(3,3))
                    plt.bar(bar1, list(prob.values()),w,color='#BCDFF2')
                    plt.bar(bar2, Group_Loss,w,color='#EFB1B1' )
                    plt.xticks(range(0,3),Target_Labels, rotation ='horizontal')
                    plt.title("Bar Graph")
                    plt.tripcolor          
                    st.pyplot(fig)

            st.markdown("<p style='height:10px;padding:10px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)

            Column_Label = [i for i in target_names]    
            Age5Emotion = [i for i in EmotionAgeAcc]
            Young5Emotion = [i for i in EmotionYoungAcc]
            Old5Emotion = [i for i in EmotionOldAcc]

            with st.container():
                    st.image('images/INTERPRETATINO.png')
                    EmoAgedf = pd.DataFrame({"Label":Column_Label,
                                            "AgeGroup":Age5Emotion,
                                            "Young":Young5Emotion,
                                            "Old":Old5Emotion
                    })
                    st.dataframe(EmoAgedf,use_container_width=True)

        else:
            rcol1, rcol2,rcol3 = st.columns([0.1,1,0.1])

            with rcol2:
                st.error("Dataset Doesn't Exist in the Directory!!!")
                st.warning("Unable to upload and set data directory path will triger this error please check your file before running this module")
                st.balloons()
              


#GENDER GROUP   


    if choice == 'Gender':
        if os.path.exists("Gender/GenderCombined"):
            TestGenderCombined = ImageDataGenerator(rescale=1./255)
            TestGenderMale = ImageDataGenerator(rescale=1./255)
            TestGenderFemale = ImageDataGenerator(rescale=1./255)

            # This is where you should set your directory path for the dataset
            GenderCombined = TestGenderCombined.flow_from_directory(
                                    'Gender/GenderCombined',
                                    target_size= (64,64),
                                    batch_size = 16,
                                    color_mode = 'rgb',
                                    class_mode = 'categorical'

                                    )
        

            GenderMale = TestGenderMale.flow_from_directory(
                                    'Gender/Male',
                                    target_size= (64,64),
                                    batch_size = 16,
                                    color_mode = 'rgb',
                                    class_mode = 'categorical'

                                    )


            GenderFemale = TestGenderFemale.flow_from_directory(
                                    'Gender/female',
                                    target_size= (64,64),
                                    batch_size = 16,
                                    color_mode = 'rgb',
                                    class_mode = 'categorical'

                                    )


            optimizers = Adam(learning_rate=1e-4,
                                                beta_1=0.9,
                                                beta_2=0.999,
                                                epsilon=1e-07)

            loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers,  metrics=['accuracy'])
            (eval_lossGender, eval_accuracyGender) = loaded_model.evaluate( GenderCombined, batch_size=16,verbose=1)
            (eval_lossMale, eval_accuracyMale) = loaded_model.evaluate( GenderMale, batch_size=16,verbose=1)
            (eval_lossFemale, eval_accuracyFemale) = loaded_model.evaluate( GenderFemale, batch_size=16,verbose=1)

            
            GenderPrediction = loaded_model.predict(GenderCombined)    
            Gender_c_matrix = confusion_matrix(GenderCombined.classes, GenderPrediction.argmax(axis=1))
            EmotionGenderAcc = Gender_c_matrix.diagonal()/Gender_c_matrix.sum(axis=1)*100
            MalePrediction = loaded_model.predict(GenderMale)    
            Male_c_matrix = confusion_matrix(GenderMale.classes, MalePrediction.argmax(axis=1))
            EmotionMaleAcc = Male_c_matrix.diagonal()/Male_c_matrix.sum(axis=1)*100
            FemalePrediction = loaded_model.predict(GenderFemale)    
            Female_c_matrix = confusion_matrix(GenderFemale.classes, FemalePrediction.argmax(axis=1))
            EmotionFemaleAcc = Female_c_matrix.diagonal()/Female_c_matrix.sum(axis=1)*100

        
            


            a = np.array([eval_accuracyGender*100,eval_accuracyMale*100,eval_accuracyFemale*100]) 
            b = np.array([eval_lossGender,eval_lossMale,eval_lossFemale])
            Label_acc = ["Gender Group", "Male", "Female"]
            Acc_Labels = [i for i in Label_acc]
            Group_Acc = [i for i in a]
            Group_Loss = [i for i in b]
            
            col1,col2,col3 = st.columns([0.3,1,1])
            with col2:
                st.image('images/C3.png')
        
            r2col1, r2col2 = st.columns([1.5,0.6])
        
            with st.container():
                
                with r2col1:
                    st.markdown("<p style='font-size:1.8rem;color:#3B6978; text-align: center;'>Evaluation Accuracy<p>",unsafe_allow_html=True)
                    st.markdown("<p style='height:5px;padding:5px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
                    Gender_df = pd.DataFrame({"Label":Acc_Labels,
                                            "Accuracy":Group_Acc,
                                            "Loss":Group_Loss
                    })
                    st.dataframe(Gender_df, use_container_width=True)
                with r2col2:
            
                    Target_Labels = ["GenderGroup", "Male", "Female"] 
                    prob = {}
                    w = 0.4
                    for i in a:
                        iprob = round(i,2)
                        prob[i] = iprob
                                                                
                                                            

                    bar1 = np.arange(len(prob))
                    bar2 = [e+w for e in bar1]
                    fig = plt.figure(figsize=(3,3))
                    plt.bar(bar1, list(prob.values()),w,color='#BCDFF2')
                    plt.bar(bar2, Group_Loss,w,color='#EFB1B1' )
                    plt.xticks(range(0,3),Target_Labels, rotation ='horizontal')
                    plt.title("Bar Graph")
                    plt.tripcolor          
                    st.pyplot(fig)

            st.markdown("<p style='height:10px;padding:10px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
            r3col1, r3col2, r3col3 = st.columns([1.5,1.5,1.5])

            Column_Label = [i for i in target_names]    
            Gender5Emotion = [i for i in EmotionGenderAcc]
            Male5Emotion = [i for i in EmotionMaleAcc]
            Female5Emotion = [i for i in EmotionFemaleAcc]

            with st.container():
                    st.image('images/INTERPRETATINO.png')
                    #st.markdown("<p style='font-size:1.8rem;color:#3B6978; text-align: center;'>Five Emotions Accuracy<p>",unsafe_allow_html=True)
                    EmoGenderdf = pd.DataFrame({"Label":Column_Label,
                                            "GenderGroup":Gender5Emotion,
                                            "Male":Male5Emotion,
                                            "Female":Female5Emotion
                    })
                    st.dataframe(EmoGenderdf,use_container_width=True)
        else:
            rcol1, rcol2,rcol3 = st.columns([0.1,1,0.1])

            with rcol2:
                st.error("Dataset Doesn't Exist in the Directory!!!")
                st.warning("Unable to upload and set data directory path will triger this error please check your file before running this module")
                st.balloons()
              



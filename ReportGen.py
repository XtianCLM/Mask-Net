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
from keras.utils.np_utils import to_categorical 
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix
import os
from pretty_confusion_matrix import pp_matrix

def Show_Report():
    st.title("Report Generation")
    emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4:"Surprise"}
    target_names = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

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


    loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers,  metrics=['categorical_accuracy'])


    rcol1, rcol2 = st.columns([1,1])
    with rcol1:
        choice = st.selectbox("Select Report", ['Emotions','Age', 'Gender'])

    st.markdown("<p style='height:30px;padding:15px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
    if choice == 'Emotions':
            if os.path.exists("LmCutDb/test"):

                test_data_gen = ImageDataGenerator(rescale=1./255)
                # Preprocess all test images
                # This is where you should set your directory path for the dataset
                test_generator = test_data_gen.flow_from_directory(
                                    'LmCutDb/test',
                                    target_size=(64, 64),
                                    batch_size= 16,
                                    color_mode="rgb",
                                    class_mode='categorical')

                # do prediction on test data
                predictions2 = loaded_model.predict(test_generator)

                c_matrix = confusion_matrix(test_generator.classes, predictions2.argmax(axis=1))
                Acc5Emotions = c_matrix.diagonal()/c_matrix.sum(axis=1)*100
                

                predicted_class_indices=np.argmax(predictions2,axis=1)



                labels = (test_generator.class_indices)
                labels = dict((v,k) for k,v in labels.items())
                actual = [labels[i] for i in test_generator.labels]
                listprob = (np.array(result)*100 for result in predictions2)
                prob = [max(result)*100 for result in predictions2]
                predictions3 = [labels[k] for k in predicted_class_indices]
                FiveEmoLabel = [i for i in target_names]
                FiveEmoProb = [k for k in Acc5Emotions]

                
                def load_data():
                    return  pd.DataFrame({"Actual":actual,
                                    "Probabilities": listprob,
                                    "Highest Probability":prob,
                                    "Predictions":predictions3})
                results = load_data()
                # results.to_csv("FiveEmotionresults.csv",index=False)
                ccol1, ccol2 = st.columns([1,1])
                with ccol1:
                    

                    fig1, ax1 = plt.subplots()
                    ax1.pie(FiveEmoProb, labels= FiveEmoLabel, autopct="%1.1f%%", shadow=True, startangle=90)
                    ax1.axis("equal")

                    st.pyplot(fig1)
                
                with ccol2:
                    df_cm = pd.DataFrame(c_matrix, index=range(0, 5), columns=range(0, 5))
                    # colormap: see this and choose your more dear
                    cmap = 'Pastel2_r'
                    
                    figs = pp_matrix(df_cm, cmap=cmap)
                    plt.xlabel('Predictions', fontsize=8)
                    plt.xticks(range(0,5), target_names, rotation ='vertical')
                    plt.yticks(range(0,5), target_names, rotation ='horizontal')
                    plt.ylabel('Actuals', fontsize=8)
                    plt.title('Confusion Matrix', fontsize=20)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot(figs)
                    

                    # Display the dataframe and allow the user to stretch the dataframe
                st.checkbox("Use container width", value=False, key="use_container_width")
                st.dataframe(results, use_container_width=st.session_state.use_container_width)
            else:
                rcol1, rcol2,rcol3 = st.columns([0.1,1,0.1])

                with rcol2:
                    st.error("Dataset Doesn't Exist in the Directory!!!")
                    st.warning("Unable to upload and set data directory path will triger this error please check your file before running this module")
                    st.balloons()

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

            
                
                AgePrediction = loaded_model.predict(AgeCombined)    
                Age_c_matrix = confusion_matrix(AgeCombined.classes, AgePrediction.argmax(axis=1))
        
                YoungPrediction = loaded_model.predict(AgeYoung)    
                Young_c_matrix = confusion_matrix(AgeYoung.classes, YoungPrediction.argmax(axis=1))
                predicted_Young_indices=np.argmax(YoungPrediction,axis=1)

                OldPrediction = loaded_model.predict(AgeOld)    
                Old_c_matrix = confusion_matrix(AgeOld.classes, OldPrediction.argmax(axis=1))
                predicted_Old_indices=np.argmax(OldPrediction,axis=1)

            
                pred_acc_Age = accuracy_score(AgeCombined.classes,AgePrediction.argmax(axis=1))
                pred_acc_Young = accuracy_score(AgeYoung.classes,YoungPrediction.argmax(axis=1))
                pred_acc_Old = accuracy_score(AgeOld.classes,OldPrediction.argmax(axis=1))


                a = np.array([pred_acc_Age*100,pred_acc_Young*100,pred_acc_Old*100]) 
                Label_acc = ["Age Group", "Young", "Old"]



                
                ccol1, ccol2 = st.columns([1,1])
                with ccol1:
                    
                        
                    fig1, ax1 = plt.subplots()
                    ax1.pie(a, labels= Label_acc, autopct="%1.1f%%", shadow=True, startangle=90)
                    ax1.axis("equal")

                    st.pyplot(fig1)
                
                with ccol2:
                    df_cm = pd.DataFrame(Age_c_matrix, index=range(0, 5), columns=range(0, 5))
                    # colormap: see this and choose your more dear
                    cmap = 'Pastel2_r'
                    
                    figs = pp_matrix(df_cm, cmap=cmap)
                    plt.xlabel('Predictions', fontsize=8)
                    plt.xticks(range(0,5), target_names, rotation ='vertical')
                    plt.yticks(range(0,5), target_names, rotation ='horizontal')
                    plt.ylabel('Actuals', fontsize=8)
                    plt.title('Confusion Matrix', fontsize=20)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot(figs)
                
                rcol1, rcol2 = st.columns([1,1])
                with rcol1:
                    choosegen = st.selectbox("Select Report to Generate", ['Young', 'Old'])

                with rcol2:
                    st.markdown("")
                    st.markdown("")
                    st.button("Generate")

                if choosegen == "Young":
                    labels = (AgeYoung.class_indices)
                    labels = dict((v,k) for k,v in labels.items())
                    actual = [labels[i] for i in AgeYoung.labels]
                    listprob = (np.array(result)*100 for result in YoungPrediction)
                    prob = [max(result)*100 for result in YoungPrediction]
                    predictionsYoung = [labels[k] for k in predicted_Young_indices]

                    
                    def load_data():
                        return  pd.DataFrame({"Actual":actual,
                                        "Probabilities": listprob,
                                        "Highest Probability":prob,
                                        "Predictions":predictionsYoung})
                    results = load_data()
                    # results.to_csv("FiveEmotionresults.csv",index=False)

                    # Display the dataframe and allow the user to stretch the dataframe
                    st.checkbox("Use container width", value=False, key="use_container_width")
                    st.dataframe(results, use_container_width=st.session_state.use_container_width)
                if choosegen == "Old":
                    labels = (AgeOld.class_indices)
                    labels = dict((v,k) for k,v in labels.items())
                    actual = [labels[i] for i in AgeOld.labels]
                    listprob = (np.array(result)*100 for result in OldPrediction)
                    prob = [max(result)*100 for result in OldPrediction]
                    predictionsOld = [labels[k] for k in predicted_Old_indices]

                    filenames=AgeOld.filenames
                    def load_data():
                        return  pd.DataFrame({"Actual":actual,
                                        "Probabilities": listprob,
                                        "Highest Probability":prob,
                                        "Predictions":predictionsOld})
                    results = load_data()
                    # results.to_csv("FiveEmotionresults.csv",index=False)

                    # Display the dataframe and allow the user to stretch the dataframe
                    st.checkbox("Use container width", value=False, key="use_container_width")
                    st.dataframe(results, use_container_width=st.session_state.use_container_width)

            else:
                rcol1, rcol2,rcol3 = st.columns([0.1,1,0.1])

                with rcol2:
                    st.error("Dataset Doesn't Exist in the Directory!!!")
                    st.warning("Unable to upload and set data directory path will triger this error please check your file before running this module")
                    st.balloons()


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
                                        'Gender/Female',
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

            
                
                GenderPrediction = loaded_model.predict(GenderCombined)    
                Gender_c_matrix = confusion_matrix(GenderCombined.classes, GenderPrediction.argmax(axis=1))
        
                MalePrediction = loaded_model.predict(GenderMale)    
                Male_c_matrix = confusion_matrix(GenderMale.classes, MalePrediction.argmax(axis=1))
                predicted_Male_indices=np.argmax(MalePrediction,axis=1)

                FemalePrediction = loaded_model.predict(GenderFemale)    
                Female_c_matrix = confusion_matrix(GenderFemale.classes, FemalePrediction.argmax(axis=1))
                predicted_Female_indices=np.argmax(FemalePrediction,axis=1)

            
                pred_acc_Gender = accuracy_score(GenderCombined.classes,GenderPrediction.argmax(axis=1))
                pred_acc_Male = accuracy_score(GenderMale.classes,MalePrediction.argmax(axis=1))
                pred_acc_Female = accuracy_score(GenderFemale.classes,FemalePrediction.argmax(axis=1))


                a = np.array([pred_acc_Gender*100,pred_acc_Male*100,pred_acc_Female*100]) 
                Label_acc = ["Gender Group", "Male", "Female"]



                
                ccol1, ccol2 = st.columns([1,1])
                with ccol1:
                    
                        
                    fig1, ax1 = plt.subplots()
                    ax1.pie(a, labels= Label_acc, autopct="%1.1f%%", shadow=True, startangle=90)
                    ax1.axis("equal")

                    st.pyplot(fig1)
                
                with ccol2:
                    df_cm = pd.DataFrame(Gender_c_matrix, index=range(0, 5), columns=range(0, 5))
                    # colormap: see this and choose your more dear
                    cmap = 'Pastel2_r'
                    
                    figs = pp_matrix(df_cm, cmap=cmap)
                    plt.xlabel('Predictions', fontsize=8)
                    plt.xticks(range(0,5), target_names, rotation ='vertical')
                    plt.yticks(range(0,5), target_names, rotation ='horizontal')
                    plt.ylabel('Actuals', fontsize=8)
                    plt.title('Confusion Matrix', fontsize=20)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot(figs)
                
                rcol1, rcol2 = st.columns([1,1])
                with rcol1:
                    choosegen = st.selectbox("Select Report to Generate", ['Male', 'Female'])

                with rcol2:
                    st.markdown("")
                    st.markdown("")
                    st.button("Generate")

                if choosegen == "Male":
                    labels = (GenderMale.class_indices)
                    labels = dict((v,k) for k,v in labels.items())
                    actual = [labels[i] for i in GenderMale.labels]
                    listprob = (np.array(result)*100 for result in MalePrediction)
                    prob = [max(result)*100 for result in MalePrediction]
                    predictionsMale = [labels[k] for k in predicted_Male_indices]

                    
                    def load_data():
                        return  pd.DataFrame({"Actual":actual,
                                        "Probabilities": listprob,
                                        "Highest Probability":prob,
                                        "Predictions":predictionsMale})
                    results = load_data()
                    # results.to_csv("FiveEmotionresults.csv",index=False)

                    # Display the dataframe and allow the user to stretch the dataframe
                    st.checkbox("Use container width", value=False, key="use_container_width")
                    st.dataframe(results, use_container_width=st.session_state.use_container_width)
                if choosegen == "Female":
                    labels = (GenderFemale.class_indices)
                    labels = dict((v,k) for k,v in labels.items())
                    actual = [labels[i] for i in GenderFemale.labels]
                    listprob = (np.array(result)*100 for result in FemalePrediction)
                    prob = [max(result)*100 for result in FemalePrediction]
                    predictionsFemale = [labels[k] for k in predicted_Female_indices]

                    
                    def load_data():
                        return  pd.DataFrame({"Actual":actual,
                                        "Probabilities": listprob,
                                        "Highest Probability":prob,
                                        "Predictions":predictionsFemale})
                    results = load_data()
                    # results.to_csv("FiveEmotionresults.csv",index=False)

                    # Display the dataframe and allow the user to stretch the dataframe
                    st.checkbox("Use container width", value=False, key="use_container_width")
                    st.dataframe(results, use_container_width=st.session_state.use_container_width)

            else:
                rcol1, rcol2,rcol3 = st.columns([0.1,1,0.1])

                with rcol2:
                    st.error("Dataset Doesn't Exist in the Directory!!!")
                    st.warning("Unable to upload and set data directory path will triger this error please check your file before running this module")
                    st.balloons()

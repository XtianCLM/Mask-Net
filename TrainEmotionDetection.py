from random import shuffle
from unicodedata import category
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
import streamlit as st
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from msilib.schema import Binary
from tkinter import Image
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam, SGD
from keras.utils.np_utils import to_categorical 
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_confusion_matrix
import time

batch_size = 16
img_height = 64
img_width = 64

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

    
tf.config.list_physical_devices('GPU')

def Show_My_Model():
        app_title = 'Experimentation'
        with st.spinner("⚗️Now loading {}".format(app_title)):
                time.sleep(1) 

                emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4:"Surprise"}

                train_ds = ImageDataGenerator(rescale=1./255,rotation_range = 13,brightness_range=[0.6,1],horizontal_flip=True ,fill_mode = 'nearest')

                validation_ds = ImageDataGenerator(rescale=1./255)

                train_gen = train_ds.flow_from_directory(
                        'LmCutDb/train',
                        target_size= (img_height,img_width),
                        batch_size = batch_size,
                        color_mode = 'rgb',
                        seed = 123,
                        shuffle = True,
                        class_mode = 'categorical'

                        )

                validation_gen = validation_ds.flow_from_directory(
                        'LmCutDb/validation',
                        target_size= (img_height,img_width),
                        batch_size = batch_size,
                        color_mode = 'rgb',
                        shuffle = False,
                        class_mode = 'categorical'

                        )


                model = Sequential()
               
                model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding = 'valid', input_shape=(img_height, img_width, 3)))
                model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding = 'valid'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
                model.add(Dropout(0.2))

                model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
                model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
                model.add(Dropout(0.2))

                model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))
                model.add(Dropout(0.2))

                model.add(Flatten())
                model.add(Dense((256), activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(5, activation='softmax'))

                #optimizers = SGD(learning_rate=1e-4,momentum=0.9)
                optimizers = Adam(learning_rate=1e-4,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=1e-07)

                model.compile(loss='categorical_crossentropy', optimizer=optimizers,  metrics=['accuracy'])

                model.summary()
                



                my_callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=3),
                # tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
                tf.keras.callbacks.TensorBoard(log_dir='./logs'),
                ]

                hist = model.fit(train_gen,  epochs=80, validation_data=validation_gen, callbacks=my_callbacks)

                (eval_loss, eval_accuracy) = model.evaluate( 
                validation_gen, batch_size=batch_size,     verbose=1)

                model_json = model.to_json()
                with open("model.json", "w") as json_file:
                        json_file.write(model_json)

# save trained model weight in .h5 file
                model.save_weights('model.h5')

               
                print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100)) 
                print("[INFO] Loss: {}".format(eval_loss))

                


                col1,col2 = st.columns([1,1])
                with col1:
                        figs = plt.figure()
                        plt.plot(hist.history['loss'], color='teal', label='loss')
                        plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
                        figs.suptitle('Loss', fontsize=20)
                        plt.legend(loc="upper left")
                        plt.ylabel('accuracy') 
                        plt.xlabel('epoch')
                        st.pyplot(figs)
                        st.write("accuracy: {:.2f}%".format(eval_accuracy * 100))
                with col2:
                        fig = plt.figure()
                        plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
                        plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
                        fig.suptitle('Accuracy', fontsize=20)
                        plt.legend(loc="upper left")
                        plt.ylabel('accuracy') 
                        plt.xlabel('epoch')
                        st.pyplot(fig)
                        st.write("Loss: {:.3f}".format(eval_loss))
               
              

        #         test_data_gen = ImageDataGenerator(rescale=1./255)

        # # Preprocess all test images
        #         test_generator = test_data_gen.flow_from_directory(
        #                 'ExpData/validation',
        #                 target_size=(img_height, img_width),
        #                 batch_size= 15,
        #                 color_mode="grayscale",
        #                 class_mode='categorical')

        #         # do prediction on test data
        #         predictions = model.predict(test_generator)
              
                
        #         # see predictions
        #         for result in predictions:
        #         #     print(result)
        #             max_index = int(np.argmax(result))
                    
        #             print("prediction={}".format(emotion_dict[max_index]))

        #         print("-----------------------------------------------------------------")
        #         # confusion matrix
        #         c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))
                
        #         figs, ax = plot_confusion_matrix(conf_mat=c_matrix, figsize=(3, 3), cmap=plt.cm.Blues)
        #         plt.xlabel('Predictions', fontsize=8)
        #         plt.ylabel('Actuals', fontsize=8)
        #         plt.title('Confusion Matrix', fontsize=8)
        #         st.pyplot(figs)

        #         print("-----------------------------------------------------------------")
        #         print(classification_report(test_generator.classes, predictions.argmax(axis=1)))

        #         img = cv2.imread('TestImage/Testing.jpg')
        #         image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        #         resize = cv2.resize(image, (img_height, img_width))
        #         prediction = model.predict(np.expand_dims(resize/255, 0))
        #         max_index = int(np.argmax(prediction))

        #         col1, col2 = st.columns([1,1])

        #         with col1:
        #                 st.info(emotion_dict[max_index])
        #                 st.image(image,width=200)
                
                
        #         print("prediction={}".format(emotion_dict[max_index]))
        #         st.write(prediction.shape)
        #         prob = {}
        #         w = 0.4
        #         for label in prediction:
        #                 for i in label:
        #                         iprob = round(i,2)
        #                         prob[i] = iprob

        #                         bar1 = np.arange(len(prob))
        #                         fig = plt.figure(figsize=(3,3))
        #                         plt.bar(bar1, list(prob.values()),w,color='#2e38f2' )
        #                         # plt.xticks(bar1,emotion_dict[cont])
        #                         plt.title("Probability Distribution")
        #                         plt.tripcolor
        #         with col2:                
        #                 st.pyplot(fig)

                
              
               


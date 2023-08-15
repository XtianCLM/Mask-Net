from turtle import width
import streamlit as st
import streamlit.components.v1 as components
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from keras.utils.np_utils import to_categorical 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import time

batch_size = 32
img_height = 48
img_width = 48

def show_Dataset_info():
    app_title = 'Datasets'
    with st.spinner("💽Now loading {}".format(app_title)):
        time.sleep(1)

      
    if os.path.exists("LmCutDb/train"):

        # This is where you should set your directory path for the dataset
        train_ds = tf.keras.utils.image_dataset_from_directory(
        "LmCutDb/train",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
        
        
        class_names = train_ds.class_names
        #st.title('Datasets')
        st.markdown("<h2 style='font-size:43px;color:#00476D;line-height:1.2;'>DATASET</h2>",unsafe_allow_html=True)
        st.markdown("<p style='height:6px;padding:7px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
        col1,col2 = st.columns([1,1])
    
        with col1:
            # from tensorflow.python.client import device_lib 
            # print(device_lib.list_local_devices())
            st.markdown("<p><span style='font-size:1.8rem;color:#3B6978;'>Affectnet </span>is a large facial expression dataset with around 0.4 million images manually labeled for the presence of eight (neutral, happy, angry, sad, fear, surprise, disgust, contempt) facial expressions along with the intensity of valence and arousal. The researcher only use the 5 universal emotion for image classification<p>",unsafe_allow_html=True)
            st.markdown("<p>The training phase will use 80% of the data for training and let the model learn from the given data. The validation phase will use 10% of the data for evaluating the performance or the accuracy of the classifiers and the remaining 10% percent will be used for testing. To have an equal number of images in five different emotions the researcher will only choose 2355 random images for five emotion categories.<p>",unsafe_allow_html=True)
        st.markdown("")
        st.markdown("")
        with col2:
            st.markdown("<p><span style='font-size:1.8rem;color:#3B6978;'>Five Emotions</span> In 1969, after recognizing a universality among emotions in different groups of people despite the cultural differences, Ekman and Friesen classified six emotional expressions to be universal: happiness, sadness, disgust, surprise, anger, and neutral(Dharmaratne et. al., 2022).These 5 universal emotions will be used for image emotion detection to label images that are processed in the model.<p>",unsafe_allow_html=True)
            choice = st.selectbox("Select one from five different emotions", ['Happy', 'Sad', 'Surprise', 'Angry', 'Neutral'])
        st.markdown("")
        st.markdown("")
        
        rcol1, rcol2,rcol3 = st.columns([0.1,1,0.1])
        with rcol2:
            img_folder = ['Angry','Happy','Neutral', 'Sad', 'Surprise']
            nimgs = {}
            valnimgs = {}
            w = 0.4

            for i in img_folder:
                nimges = len(os.listdir('LmCutDb/train/'+i+'/'))
                valnimges = len(os.listdir('LmCutDb/validation/'+i+'/'))

                nimgs[i] = nimges
                valnimgs[i] = valnimges

                bar1 = np.arange(len(nimgs))
                bar2 = [e+w for e in bar1]
            fig = plt.figure(figsize=(11,6))
            plt.bar(bar1, list(nimgs.values()),w,color='#2e38f2' )
            plt.bar(bar2, list(valnimgs.values()),w,color='#f7f740' )
            plt.xticks(bar1, list(nimgs.keys()))
            plt.title("Distribution of different classes in train and validation Dataset")
            st.markdown("<p style='font-size:1.3rem;color:#3B6978;text-align: center; font-weight:700; padding-left:50px'> Train dataset Color: <span style ='font-size:5rem;color:#2e38f2'> . </span>  Validation dataset Color: <span style ='font-size:5rem;color:#f7f740'> . </span><p>",unsafe_allow_html=True)
            plt.tripcolor
            st.pyplot(fig)
            
        
        st.markdown("")
        st.markdown("")
        
        st.markdown("<p style='font-size:1.8rem;color:#3B6978;'>Face Detection and Removing Dodgy images<p>",unsafe_allow_html=True)
        st.markdown("<p>The researchers uses a pretrained model for face and mask detection since this model mostly focus in emotion classification. The face and mask detection model uses OpenCV to detect faces in the input images and a CNN as mask/no-mask binary classifier applied to the face ROI. The Deep Learning model currently used has the architecture suggested by Adrian Rosebrock <span><a href ='https://pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/' style='font-size:1.2rem; text-decoration: none; color:#3B6978;'>here</a></span> and has been trained using <span><a href ='https://github.com/prajnasb/observations/tree/master/experiements/data' style='font-size:1.2rem; text-decoration: none; color:#3B6978;'>this</a></span> image data set. The trained model has been shared in this repository. The face detector algorithm comes from <span><a href ='https://github.com/Shiva486/facial_recognition' style='font-size:1.2rem; text-decoration: none; color:#3B6978;'>here</a></span>: This face detector algorithm is called Caffe model. <p>",unsafe_allow_html=True)
        st.markdown("<p>After detecting the face and extracting region of interest in the images the researchers uses an os module to preprocess the images such as cropping the region of interest which is the upper area of the face and removing dodgy images that doesn't have the appropriate file extension, corrupted images and mislabeled images. Os module also used for renaming images in the datasets to apply category in each of the image</p>",unsafe_allow_html=True)
        st.markdown("")
        st.markdown("")

        

        r1col1, r1col2 = st.columns([1,1])
        with r1col1:
            st.markdown("<p style='font-size:1.8rem;color:#3B6978;'>Synthetic Mask<p>",unsafe_allow_html=True)
            st.markdown("<p>Since the activity requires a mask for classiying emotion of facial images it is necessary for the researcher to apply synthetic mask in each facial images in the datasets.<p>",unsafe_allow_html=True)
            st.markdown("<p>This process is done using the MaskTheFace algorithm or model <span><a href ='https://github.com/aqeelanwar/MaskTheFace/blob/master/faq.md' style='font-size:1.2rem; text-decoration: none; color:#3B6978;'>GITHUB</a></span>. MaskTheFace is computer vision-based script to mask faces in images. It uses a dlib based face landmarks detector to identify the face tilt and six key features of the face necessary for applying mask. Based on the face tilt, corresponding mask template is selected from the library of mask.</p>",unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")


            
        with r1col2:
      
            st.markdown("<p style='height:26px;padding:5px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
            st.image("images/SyntheticMask.png")

        st.markdown("<p style='font-size:1.8rem;color:#3B6978;'>Upper Face Detection<p>",unsafe_allow_html=True)
        st.markdown("<p> In this step, the researchers only wanted to extract the upper part of the masked facial images. The upper part of the images is extracted using the dlib 48 facial landmarks. In this step, the researcher wanted to make sure that the generated points covered the area outside of the eyes and eyebrows. By doing this, the computational complexity is significantly reduced.  <p>",unsafe_allow_html=True)
        pcol1,pcol2,pcol3 = st.columns([0.1,1,0.1])
        with pcol2:
            st.image("images/Process.png",use_column_width=True)


        st.markdown("<p style='font-size:1.8rem;color:#3B6978;'>Normalizing image inputs<p>",unsafe_allow_html=True)
        st.markdown("<p> Data normalization is an important step which ensures that each input parameter (pixel, in this case) has a similar data distribution. This makes convergence faster while training the network. Data normalization is done by subtracting the mean from each pixel and then dividing the result by the standard deviation. The distribution of such data would resemble a Gaussian curve centered at zero. For image inputs we need the pixel numbers to be positive, so we might choose to scale the normalized data in the range [0,1] or [0, 255]. For our data-set example, the following montage represents the normalized data.  <p>",unsafe_allow_html=True)
        

        st.markdown("<p style='font-size:1.8rem;color:#3B6978;'>Image Augmentation in Keras<p>",unsafe_allow_html=True)
        st.markdown("<p>Keras ImageDataGenerator class provides a quick and easy way to augment your images. It provides a host of different augmentation techniques like standardization, rotation, shifts, flips, brightness change, and many more. ImageDataGenerator class ensures that the model receives new variations of the images at each epoch. But it only returns the transformed images and does not add it to the original corpus of images.<p>",unsafe_allow_html=True)
        
        
        scol1,scol2 = st.columns([1,1])

        with scol1:
            st.markdown("<p> <span style='font-size:1.2rem;color:#3B6978;'>Random Rotation</span> Since our images orientation is not consistent and some images are slightly rotating, the reaserchers apply random rotation augmentation techniques to allows the model to become invariant to the orientation of the image. </p>",unsafe_allow_html=True)
            st.markdown("<p> <span style='font-size:1.2rem;color:#3B6978;'>Horizontal Flip</span> Flipping images is also a great augmentation technique and it makes sense to use it with a lot of different objects. ImageDataGenerator class has parameters horizontal_flip and vertical_flip  for flipping along the vertical or the horizontal axis. However, in this study the researcher only use horizontal_flip for flipping images along horizontal axis. </p>",unsafe_allow_html=True)
            # st.markdown("<p> <span style='font-size:1.2rem;color:#3B6978;'>Random Shift</span> In our dataset some faces did not happen to be in the center of the image. To overcome this problem the researcher can shift the pixels of the image either horizontally or vertically using Random shift augmentation. </p>",unsafe_allow_html=True)
            st.markdown("<p> <span style='font-size:1.2rem;color:#3B6978;'>Random Brightness</span> The researcher used Brightness augmentation to randomly changes the brightness of the image. It is also a very useful augmentation technique because not all of the image in the dataset has a perfect lighting condition. So, it becomes imperative to train our model on images under different lighting conditions.  </p>",unsafe_allow_html=True)
            st.markdown("<p> <span style='font-size:1.2rem;color:#3B6978;'>Fill mode</span> When applying those transformation such as rotation, shift, etc... some pixels will move outside the image and leave an empty area that needs to be filled in. You can fil this by using fill mode technique that has a default value of “nearest” which simply replaces the empty area with the nearest pixel values.  </p>",unsafe_allow_html=True)
        with scol2:
            img = load_img('TestImage/MaskedBig.jpg')
            # convert to numpy array
            data = img_to_array(img)
            # expand dimension to one sample
            samples = expand_dims(data, 0)

            datagen = ImageDataGenerator(rotation_range = 13,fill_mode = 'nearest')

                # iterator
            single_test = datagen.flow(samples,batch_size = 1)
                                        
            fig = plt.figure(figsize=(15,15))
            for i in range(3):
            # define subplot
                plt.subplot(330 + 1 + i)
                # generate batch of images
                batch = single_test.next()
                # convert to unsigned integers for viewing
                image = batch[0].astype('uint8')
                # plot raw pixel data
                plt.imshow(image)
        # show the figure
            st.pyplot(fig)

            st.markdown("<p style='height:26px;padding:5px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)

        # Horizontal Flip
            img2 = load_img('TestImage/MaskedBig.jpg')
            # convert to numpy array
            data2 = img_to_array(img2)
            # expand dimension to one sample
            samples2 = expand_dims(data2, 0)

            datagen2 = ImageDataGenerator(horizontal_flip=True)

                # iterator
            single_test2 = datagen2.flow(samples2,batch_size = 1)
                                        
            fig2 = plt.figure(figsize=(15,15))
            for i in range(3):
            # define subplot
                plt.subplot(330 + 1 + i)
                # generate batch of images
                batch2 = single_test2.next()
                # convert to unsigned integers for viewing
                image2 = batch2[0].astype('uint8')
                # plot raw pixel data
                plt.imshow(image2)
        # show the figure
            st.pyplot(fig2)

            st.markdown("<p style='height:26px;padding:5px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)

        #Brightness
            img3 = load_img('TestImage/MaskedBig.jpg')
            # convert to numpy array
            data3 = img_to_array(img3)
            # expand dimension to one sample
            samples3 = expand_dims(data3, 0)

            datagen3 = ImageDataGenerator(brightness_range=[0.6,1])

                # iterator
            single_test3 = datagen3.flow(samples3,batch_size = 1)
                                        
            fig3 = plt.figure(figsize=(15,15))
            for i in range(3):
            # define subplot
                plt.subplot(330 + 1 + i)
                # generate batch of images
                batch3 = single_test3.next()
                # convert to unsigned integers for viewing
                image3 = batch3[0].astype('uint8')
                # plot raw pixel data
                plt.imshow(image3)
        # show the figure
            st.pyplot(fig3)

    else:
            rcol1, rcol2,rcol3 = st.columns([0.1,1,0.1])

            with rcol2:
                st.error("Dataset Doesn't Exist in the Directory!!!")
                st.warning("Unable to upload and set data directory path will triger this error please check your file before running this module")
                st.balloons()


            
            
        
    
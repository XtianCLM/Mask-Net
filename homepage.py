import streamlit as st
import requests
import time
import json
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner




def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def show_homepage():
    

    
    #local file lottie
    lottiescanningface = load_lottiefile("lottiefiles/scan.json")
    lottiesfacemask = load_lottiefile("lottiefiles/facemask.json")
    lottiesaboutus = load_lottiefile("lottiefiles/about-us.json")

    st.markdown("<p style='height:50px;padding:50px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.1,1.3,1])
    with col2:

        st.markdown("<h2 style='font-size:53px;color:#14344C;line-height:1.2;text-align:left;position:relative;bottom:.5rem;'>CONVOLUTIONAL NEURAL NETWORK IN EMOTION DETECTION USING MASKED FACIAL IMAGES</h2>",unsafe_allow_html=True)
        st.markdown("<p style='font-size:20px;color:#22577E;'> The system will be able to classify images into five universal emotions namely Angry, Happy, Neutral, Sad, Surprise.<p>",unsafe_allow_html=True)
        st.button('Explore')   

    with col3:
        st_lottie(
            lottiesfacemask,
            speed=.8,
            reverse=False,
            loop=True,
            quality="high",
            
        )


    st.markdown("<p style='height:152px;padding:112px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)



    scol1, scol2, scol3 = st.columns([1.2,0.2,1.3])

    with scol1:
        st_lottie(
            lottiesaboutus,
            speed=.9,
            reverse=False,
            loop=True,
            quality="medium",
            width=600
            
        )

    with scol3:

        st.markdown("<p style='visibility:hidden;margin-top:5px;'>invisible</p>",unsafe_allow_html=True)
        st.markdown("<h2 style='color:#0E2332;'>ABOUT </h2>",unsafe_allow_html=True)
        st.markdown("<p style='font-size:17px;'>As the pandemic began, peope were required to wear masks to ensure their own safety. People's emotions are difficult to analyze in this new normal. As a result, we decided to develop a tool for emotion detection while wearing a mask in line with the theme of Community Building through innovation and a disciplined approach</p>",unsafe_allow_html=True)
            
    

    st.markdown("<p style='height:152px;padding:80px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)

    space1,title1,title2,title3 = st.columns([0.1,1,.5,1,])
    with title1:
        st.markdown("<h2 style='color:#0E2332'>HOW IT WORKS</h2>",unsafe_allow_html=True)
        st.markdown("<p style='font-size:18px;'>This is how we process your image</p>",unsafe_allow_html=True)
        st.markdown("<p style='height:30px;padding:32px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)

    space1,subtitle1,subtitle2,space2 = st.columns([.1,1,1,.1])

    with subtitle1:
        st.image('images/facedetection2.png')

        st.markdown("<p style='font-size:20px;'>Detect face in the image</p>",unsafe_allow_html=True)
        st.markdown("<p style='border:2px solid #112C3F;height:2px;width:32px;background-color:#112C3F'></p>",unsafe_allow_html=True)
        st.markdown("<h6 style='color:#22577E;font-size:16px;width:440px;text-align:justify;'>The first process in our emotion detection is face detection, in order to proceed to the next step, the researchers must first determine whether the image has a face </h6>",unsafe_allow_html=True)
        st.markdown("<h6 style='color:#22577E;font-size:16px;'></h6>",unsafe_allow_html=True)


        st.markdown("<p style='height:15px;padding:2px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
        st.image('images/region.png')

        st.markdown("<p style='font-size:20px;'>Get The Region of Interest</p>",unsafe_allow_html=True)
        st.markdown("<p style='border:2px solid #112C3F;height:2px;width:32px;background-color:#112C3F'></p>",unsafe_allow_html=True)
        st.markdown("<h6 style='color:#22577E;font-size:16px;text-align:justify;width:440px;'>After preprocessing your image it will be pass to the model to generate a prediction probability distribution and classify your image based on the highest probability result.</h6>",unsafe_allow_html=True)
# with subtitle2:
    with subtitle2:
        st.markdown("<p style='height:2px;padding:2px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)
        st.image('images/mask2.png')
        st.markdown("<p style='font-size:20px;'>Apply Synthetic Face Mask</p>",unsafe_allow_html=True)
        st.markdown("<p style='border:2px solid #112C3F;height:2px;width:32px;background-color:#112C3F'></p>",unsafe_allow_html=True)
        st.markdown("<h6 style='color:#22577E;font-size:16px;width:440px;text-align:justify;'>A synthetic mask needs to be applied to each facial image in the dataset since the research relies on it for emotion classification. This process is done using the MaskTheFace algorithm.</h6>",unsafe_allow_html=True)
    #with subtitle4:
        st.markdown("<p style='height:17px;padding:2px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)

        st.markdown("<p style='font-size:20px;'>Detect the Emotion</p>",unsafe_allow_html=True)
        st.markdown("<p style='border:2px solid #112C3F;height:2px;width:32px;background-color:#112C3F'></p>",unsafe_allow_html=True)
        st.markdown("<h6 style='color:#22577E;font-size:16px;text-align:justify;width:440px;'>After preprocessing your image it will be pass to the model to generate a prediction probability distribution and classify the emotion of the image based on the highest probability result.</h6>",unsafe_allow_html=True)

  
    st.markdown("<p style='height:152px;padding:80px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)



    space1,dcol1, dcol2, dcol3,space2 = st.columns([.1,1,1,1,.1])

    with dcol1:
        #st.markdown("<p style='font-size:1rem;color:#555555; text-align: center'>Our Service<p>",unsafe_allow_html=True)
        st.markdown("<h2 style='color:#0E2332'>SEE OUR SOLUTION</h2>",unsafe_allow_html=True)
    st.markdown("<p style='height:5px;visibility:hidden'>invisible comment</p>",unsafe_allow_html=True)

    space1,acol1, acol2, acol3,acol4,acol5,space2 = st.columns([.1,1,.1, 1, .1, 1,.1])

    with acol1:
        st.markdown("<p style='font-size:22px;color:#112C3F;'>Model Datasets<p>",unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.1rem;color:#1f4e71;text-align:justify;'><span style='font-size:1.1rem;color:#5800FF;'>Affectnet - </span> is a large facial expression dataset with around 0.4 million images manually labeled for the presence of eight (neutral, happy, angry, sad, fear, surprise, disgust, contempt) facial expressions along with the intensity of valence and arousal. The researcher only use the 5 universal emotion for image classification<p>",unsafe_allow_html=True)
        # st.markdown("")
        # st.markdown("")
        # st.markdown("<p style='font-size:1.3rem;color:#3B6978;'>Model Datasets<p>",unsafe_allow_html=True)
        # st.markdown("<p style='font-size:1rem;color:#3B6978;'><span style='font-size:1.1rem;color:#3B6990;'>Mask Fer-2013</span>  This dataset contains 21, 345 images having surgery masked on them with five emotions namely Anger, Happiness, Sadness, Surprise, and Neutral.<p>",unsafe_allow_html=True)

    with acol3:
        st.markdown("<p style='font-size:22px;color:#112C3F;'>Face Detection<p>",unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.1rem;color:#1f4e71;text-align:justify;'>We used a pre-trained CNN model for face detection since our main objective is to detect emotions. The face detector algorithm comes from This face detector algorithm is called the Caffe model. We cropped the region of interest in the image to preprocess for emotion detection..<p>",unsafe_allow_html=True)

    with acol5:
        st.markdown("<p style='font-size:22px;color:#112C3F;'>Convolutional Neural Network<p>",unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.1rem;color:#1f4e71;text-align:justify;'><span style='font-size:1.1rem;color:#5800FF;'>CNN - </span> The Convolutional Neural Network (CNN) is a Deep Learning algorithm that takes in an input image and assign important learnable weights and biases to various aspects and objects in the image and be able to distinguish between them. As compared to other classification algorithms, CNN requires much less pre-processing. .<p>",unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")


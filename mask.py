import cv2
import numpy as np
from skimage.transform import resize
import streamlit as st

# Reading an image
img = cv2.imread('TestImage/Maskedsmall.jpg')


# image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
# The kernel to be used for dilation 
# purpose
kernel = np.ones((5, 5), np.uint8)
  

# converting the image to HSV format
image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# hsv = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)
  
# # defining the lower and upper values
# # of HSV, this will detect yellow colour
# Lower_hsv = np.array([90,50,50])
# Upper_hsv = np.array([130,255,255])
  
# # creating the mask
# Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
# Mask = cv2.bitwise_not(Mask)


# contours, _ = cv2.findContours(Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if area >4000:
#         cv2.fillPoly(images, pts =[cnt], color=(0,0,0))
#         cv2.drawContours(images, [cnt], -1, (0,0,0),10)
  
# # Inverting the mask 

# face = cv2.resize(images, (128, 128))                     
# Displaying the image
st.image(image)

  
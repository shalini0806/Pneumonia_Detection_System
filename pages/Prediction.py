
import streamlit as st
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing import image
#from keras.preprocessing import image
from keras.models import load_model

from keras.preprocessing import image

from tensorflow.keras.utils import load_img, img_to_array 



from PIL import Image
import tensorflow as tf


st.set_page_config(layout = 'wide')
#import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
st.title(" Let's determine whether or not you have pneumonia! ")
col1, col2, col3 = st.columns([1, 5, 3])
with col2:
    image = Image.open("C:\\st\\pages\\image2.jpg")
    st.image(image, width = 400)



model = tf.keras.models.load_model("normal&pneumonia.h5")
uploaded_file = st.file_uploader("Choose a image file", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    test_image =load_img(uploaded_file, target_size = (128,128))
    
    st.image(test_image,width = 300)
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis =0)
    model = load_model("normal&pneumonia.h5")
    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        np.vstack([test_image])
        result = model.predict(test_image,verbose=1)
        result = result[0][0]
        if result == 0:
                
                st.title("This Patient is NORMAL")
                
        else:
                st.title("This Patient have PNEUMONIA") 


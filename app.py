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


import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
#st.set_option("deprecation.showfileUploaderEncoding", False)
#@st.cache(allow_output_mutation=True)


st.title("Pneumonia prediction")

#nav = st.sidebar.radio("navigator",["HOME"])
#if nav == "HOME":
    

image = Image.open("C:\\Users\\shali\\OneDrive\\Desktop\\premium_photo.jpg")
st.image(image, width = 700)
st.header("WHAT IS PNEUMONIA")
st.write("The air sacs in one or both lungs become inflamed when someone has pneumonia. The air sacs may swell with fluid or pus (purulent material), which can lead to a cough that produces pus or phlegm, a fever, chills, and difficulty breathing. Pneumonia can be brought on by a wide range of species, including bacteria, viruses, and fungus")
st.write("The severity of pneumonia can range from minor to potentially fatal. The most vulnerable groups include newborns and young children, adults over 65, and those with medical conditions or weaker immune systems.")
st.header("SYMPTOMS")
st.write("The signs and symptoms of pneumonia vary from mild to severe, depending on factors such as the type of germ causing the infection, and your age and overall health. Mild signs and symptoms often are similar to those of a cold or flu, but they last longer.")
st.subheader("Pneumonia may exhibit the following symptoms and signs:")
st.write("1.Chest pain when you breathe or coug")
st.write("2.Confusion or changes in mental awareness (in adults age 65 and older)")
st.write("3.Cough, which may produce phlegm")
st.write("4.Fatigue")    
st.write("5.Fever, sweating and shaking chills")
st.write("6.Lower than normal body temperature (in adults older than age 65 and people with weak immune systems)")
st.write("7.Nausea, vomiting or diarrhea")
st.write("8.Shortness of breath")
#if nav == "PREDICTION":
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
                
                st.title("This patient is NORMAL")
                
        else:
                st.title("This patient have PNEUMONIA")
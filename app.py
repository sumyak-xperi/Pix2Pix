import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import wget

def load_image(img):
    im = Image.open(img)
    im = im.resize((256,256))
    image = np.array(im)
    return image

def getImage(image):
    prediction_image = tf.expand_dims(image, axis=0)
    prediction_image = tf.cast(prediction_image, dtype=tf.float32)
    prediction_image = (prediction_image / 127.5) -1
    return prediction_image


st.image('xperi.png')
st.title("Blueprint to Satellite Image Generator using Pix2Pix-GAN Algorithm")
filepath = st.file_uploader(label = "Please Upload a Satellite Image", type=['jpg', 'png'])

if filepath:
    with st.spinner('Generating Map Image...'):
     url = "https://xperi-hackathon2k22.s3.ap-south-1.amazonaws.com/generator.h5"
     filename = wget.download(url)
     model = tf.keras.models.load_model(filename)
    img = load_image(filepath)
    col1,col2 = st.columns([3,3])
    col1.header("Blueprint")
    col1.image(img)

    predic_image = getImage(img)

    generated_image = model.predict(predic_image)
    generated_image = tf.cast(generated_image, dtype = tf.float32)
    generated_image = tf.squeeze(generated_image, axis=0)
    generated_image = (generated_image+1)/2.0
    generated_image = np.array(generated_image)

    
    col2.header("Satellite Image")
    col2.image(generated_image)


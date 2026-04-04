import streamlit as st
import PIL
import pickle
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
import numpy as np
import requests
from streamlit_lottie import st_lottie
with open(r'bin_best.pkl', 'rb') as f:
    loaded_weights = pickle.load(f)
print("wallahi")
def loadurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie = 'https://lottie.host/c006a0a3-f104-4d12-8b70-3dfb0c77eb11/Ez6J26UEBT.json'
def buildmodel():
    img_in = layers.Input(shape=(512, 512, 3))
    bckbn = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=img_in)
    bckbn.trainable = False
    shared = layers.GlobalAveragePooling2D()(bckbn.output)
    v = layers.Dense(128, activation='relu', name='v_dense')(shared)
    v = layers.Dropout(0.5)(v)
    v_out = layers.Dense(1, activation='sigmoid', name='validity_out')(v)
    ov = layers.Dense(128, activation='relu',name='ov_dense')(shared)
    ov = layers.Dropout(0.5)(ov)
    ov_out = layers.Dense(1, activation='sigmoid', name='overflow_out')(ov)
    return models.Model(inputs=img_in, outputs=[v_out, ov_out])
model = buildmodel()
model.get_layer('v_dense').set_weights(loaded_weights['v_dense'])
model.get_layer('validity_out').set_weights(loaded_weights['v_out'])
model.get_layer('ov_dense').set_weights(loaded_weights['ov_dense'])
model.get_layer('overflow_out').set_weights(loaded_weights['ov_out'])


st.set_page_config(page_title="DMC Overfill",layout='wide')
with st.container():
    left_col, right_col = st.columns(2)
    dmc_call = None
    Uploaded = None
    valid_percent = 0
    over_percent = 0
    with right_col:
        option = st.radio("Choose source:", ("Upload Photo", "Take Photo"),horizontal=True)
        if option == "Upload Photo":
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            Uploaded = True
        else:
            uploaded_file = st.camera_input("Take a picture")
            Uploaded = True
        if uploaded_file is not None:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                image = PIL.Image.open(uploaded_file)
                image.resize((800,800))
                st.image(image, caption="Bounded Image", width=300,)
            img = PIL.Image.open(uploaded_file)
            img = img.resize((512,512))
            img_array = np.array(img)
            if img_array.shape[-1] == 4:
                img = img.convert("RGB")
                img_array = np.array(img)
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
            img_dims = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_dims)
            valid_score = predictions[0][0][0]
            overflow_score = predictions[1][0][0]
            dmc_call= (valid_score > 0.5) * (overflow_score > 0.5)
            valid_percent = valid_score * 100
            over_percent = overflow_score * 100
    with left_col:
        st.subheader("Welcome User :wave:")
        st.title("IITK DMC Overfill Portal")
        st.write("To intimate DMC utilities, upload or click a picture of Overfilled Dustbin area:")
        st.write("File's Uploaded :", Uploaded)
        st.write("Validity Percentage Confidence :",valid_percent)
        st.write("Overflow Percentage Confidence :",over_percent)
        st.write("Final Verdict on Notification : ",dmc_call)
        with st.container():
            col1,col2,col3 = st.columns([1,2,1])
            with col2:
                st_lottie(lottie,height=400)

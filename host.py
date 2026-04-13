'''
MADE BY GITHUB USER: HelloThere-3 & Keke47
Dataset: Roboflow allinone v1
'''
import streamlit as st
import PIL
import pickle
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import requests
from streamlit_lottie import st_lottie
import cv2


############################################################################################################################

#VISIT THE WEB LINK BELOW FOR AN ONLINE HOSTED SITE:

# https://dmchktn-asz5iommxbns4z4rrs7kgu.streamlit.app/

##############################################################################################################################

with open(r'best_bins_weights.pkl', 'rb') as f:
    loaded_weights = pickle.load(f)
print(cv2.__version__)
def loadurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie = 'https://lottie.host/c006a0a3-f104-4d12-8b70-3dfb0c77eb11/Ez6J26UEBT.json'

def model1():
    def buildmodel():
        img_in = layers.Input(shape=(512, 512, 3))
        bckbn = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=img_in)
        bckbn.trainable = False
        shared = layers.GlobalAveragePooling2D()(bckbn.output)
        v = layers.Dense(128, activation='relu', name='v_dense')(shared)
        v = layers.Dropout(0.5)(v)
        v_out = layers.Dense(1, activation='sigmoid', name='validity_out')(v)
        ov = layers.Dense(128, activation='relu', name='ov_dense')(shared)
        ov = layers.Dropout(0.5)(ov)
        ov_out = layers.Dense(1, activation='sigmoid', name='overflow_out')(ov)
        return models.Model(inputs=img_in, outputs=[v_out, ov_out])
    model = buildmodel()
    model.get_layer('v_dense').set_weights(loaded_weights['v_dense'])
    model.get_layer('validity_out').set_weights(loaded_weights['v_out'])
    model.get_layer('ov_dense').set_weights(loaded_weights['ov_dense'])
    model.get_layer('overflow_out').set_weights(loaded_weights['ov_out'])
    return model

def model2():
    IMG_SIZE = 512
    def weighted_bce(y_true, y_pred):
        # weights: [valid, overflow]
        weights = tf.constant([0.9, 1.1])
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean(bce * weights)

    def build_model():
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        base_model.trainable = False
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128,activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64)(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(2, activation='sigmoid')(x)  # [valid, overflow]
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=weighted_bce,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )
        return model
    model = build_model()
    model.load_weights(r'best_weights.weights.h5')
    return model
def model3():
    IMG_SIZE = 512
    def focal_loss(gamma=2.0, alpha=0.6):
        def focal_loss_fixed(y_true, y_pred):
            y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
            bce = -y_true * K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred)
            p_t = (y_true * y_pred) + ((1.0 - y_true) * (1.0 - y_pred))
            loss = K.pow(1.0 - p_t, gamma) * bce
            return K.mean(loss)

        return focal_loss_fixed

    def weighted_bce(y_true, y_pred):
        # weights: [valid, overflow]
        weights = tf.constant([1.1, 1.0])
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean(bce * weights)

    def build_model():
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        base_model.trainable = False
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(2, activation='sigmoid')(x)  # [valid, overflow]
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=focal_loss(),  # weighted_bce,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )
        return model

    model = build_model()
    model.load_weights(r'tuned_weights.weights.h5')
    return model
def model4():
    #script_dir = os.path.dirname(__file__)
    IMG_SIZE = 384

    def focal_loss(gamma=2.0, alpha=0.65):
        def focal_loss_fixed(y_true, y_pred):
            y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
            p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
            bce = -y_true * K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred)
            alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
            loss = alpha_t * K.pow(1.0 - p_t, gamma) * bce
            return K.mean(loss)

        return focal_loss_fixed

    def patch_mil_head(feature_map, name, proj_dim=128, l2_val=1e-4):
        from keras import ops  # keras.ops works in both tf.keras 2.x and Keras 3

        reg = regularizers.l2(l2_val)

        # ── projection ──
        feat = layers.Conv2D(proj_dim, 1, padding="same", activation="relu",
                             kernel_regularizer=reg, name=f"{name}_proj")(feature_map)
        feat = layers.BatchNormalization(name=f"{name}_bn")(feat)
        feat = layers.SpatialDropout2D(0.2, name=f"{name}_sdrop")(feat)

        # ── attention gate: two-layer gated attention ──
        v = layers.Conv2D(proj_dim, 1, activation="tanh", padding="same",
                          kernel_regularizer=reg, name=f"{name}_V")(feat)
        u = layers.Conv2D(proj_dim, 1, activation="sigmoid", padding="same",
                          kernel_regularizer=reg, name=f"{name}_U")(feat)
        vu = layers.Multiply(name=f"{name}_vu")([v, u])

        # scalar attention score per patch  [B, H, W, 1]
        attn_logits = layers.Conv2D(1, 1, padding="same",
                                    name=f"{name}_attn_logits")(vu)

        # ── FIX 1: remove tf.shape() calls — they are unused and illegal on KerasTensors ──
        # flatten spatial dims → softmax → reshape back  [B, H*W, 1]
        attn_flat = layers.Reshape((-1, 1), name=f"{name}_attn_flat")(attn_logits)
        attn_soft = layers.Softmax(axis=1, name=f"{name}_attn_soft")(attn_flat)

        # ── FIX 2: reshape back using Lambda + ops (static shape is None for dynamic spatial dims) ──
        def _reshape_attn(inputs):
            flat, orig = inputs
            # orig shape: [B, H, W, 1]  — recover H, W at runtime via ops.shape
            shape = ops.shape(orig)
            return ops.reshape(flat, (shape[0], shape[1], shape[2], 1))

        attn_map = layers.Lambda(
            _reshape_attn, name=f"{name}_attn_map"
        )([attn_soft, attn_logits])

        # ── FIX 3: use ops.sum instead of tf.reduce_sum ──
        weighted = layers.Multiply(name=f"{name}_weighted")([feat, attn_map])
        pooled = layers.Lambda(
            lambda x: ops.sum(x, axis=[1, 2]), name=f"{name}_pooled"
        )(weighted)

        # classification head
        pooled = layers.Dense(64, activation="relu", kernel_regularizer=reg,
                              name=f"{name}_fc")(pooled)
        pooled = layers.Dropout(0.3, name=f"{name}_drop")(pooled)
        out = layers.Dense(1, activation="sigmoid", name=name)(pooled)
        return out, attn_map

    # ── 5d. Assemble model ────────────────────────────────────────────────────────
    def build_model(img_size=IMG_SIZE, trainable_backbone=False):
        inputs = layers.Input(shape=(img_size, img_size, 3), name="input_image")

        # x = make_augmentation()(inputs)  # augment only during .fit(); skipped at .predict()

        backbone = EfficientNetB3(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3),
            # drop_connect_rate=0.3   # uncomment to add stochastic depth
        )
        backbone.trainable = trainable_backbone

        features = backbone(inputs, training=False)  # keep BN stats frozen

        # shared feature refinement
        shared = layers.Conv2D(256, 3, padding="same", activation="relu",
                               kernel_regularizer=regularizers.l2(1e-4),
                               name="shared_conv")(features)
        shared = layers.BatchNormalization(name="shared_bn")(shared)
        shared = layers.SpatialDropout2D(0.2, name="shared_sdrop")(shared)

        valid_out, valid_attn = patch_mil_head(shared, "valid")
        overflow_out, overflow_attn = patch_mil_head(shared, "overflow")

        outputs = layers.Concatenate(name="predictions")([valid_out, overflow_out])

        model = models.Model(inputs, outputs, name="PatchMIL_EfficientNetB3")

        # attention sub-models for visualisation
        attn_model = models.Model(
            inputs,
            [valid_attn, overflow_attn],
            name="attention_maps"
        )

        return model, attn_model

    model, attn_model = build_model()
    model.load_weights(r'patchmil_efficientnetb3_final.weights.h5')
    return model

st.set_page_config(page_title="DMC Overfill",layout='wide')
with st.container():
    st.write("use firefox for best experience :exclamation:")
    left_col, right_col = st.columns(2)
    dmc_call = None
    Uploaded = None
    valid_percent = 0
    over_percent = 0
    with right_col:
        option_model = st.radio("Choose ML model:", ("Model_1.0", "Model_2.0","Model_Tuned","Final_Model"), horizontal=True)
        Chosen_Model = None
        if option_model == "Model_1.0":
            model = model1()
            Chosen_Model = "Model_1.0"
        elif option_model == "Model_2.0":
            model = model2()
            Chosen_Model = "Model_2.0"
        elif option_model == "Model_Tuned":
            model = model3()
            Chosen_Model = "Model_Tuned"
        else:
            model = model4()
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
            if option_model=="Final_Model":
                def apply_clahe(img_array_uint8):
                    lab = cv2.cvtColor(img_array_uint8, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # softer than equalizeHist
                    l_eq = clahe.apply(l)
                    lab = cv2.merge((l_eq, a, b))
                    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

                img = PIL.Image.open(uploaded_file)
                img = img.resize((384, 384))
                img_array = np.array(img)
                if img_array.shape[-1] == 4:
                    img = img.convert("RGB")
                    img_array = np.array(img, dtype=np.uint8)
                arr = apply_clahe(img_array)  # consistent CLAHE
                arr = tf.keras.applications.efficientnet.preprocess_input(arr.astype("float32"))
                inp = np.expand_dims(arr, axis=0)
                predictions = model.predict(inp, verbose=0)[0]
                valid_score = predictions[0]
                overflow_score = predictions[1]
                dmc_call = (valid_score > 0.54) * (overflow_score > 0.4)
                valid_percent = valid_score * 100
                over_percent = overflow_score * 100
            else:
                img = PIL.Image.open(uploaded_file)
                img = img.resize((512,512))
                img_array = np.array(img)
                if img_array.shape[-1] == 4:
                    img = img.convert("RGB")
                    img_array = np.array(img,dtype=np.uint8)
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_clahe = clahe.apply(l)
                lab = cv2.merge((l_clahe, a, b))
                img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
                img_dims = np.expand_dims(img_array, axis=0)
                predictions = model.predict(img_dims)
                if option_model == "Model_2.0":
                    valid_score = predictions[0][0]
                    overflow_score = predictions[0][1]
                    dmc_call= (valid_score > 0.47) * (overflow_score > 0.62)
                    valid_percent = valid_score * 100
                    over_percent = overflow_score * 100
                elif option_model == "Model_1.0":
                    valid_score = predictions[0][0][0]
                    overflow_score = predictions[1][0][0]
                    dmc_call = (valid_score > 0.61) * (overflow_score > 0.55)
                    valid_percent = valid_score * 100
                    over_percent = overflow_score * 100
                else:
                    valid_score = predictions[0][0]
                    overflow_score = predictions[0][1]
                    dmc_call = (valid_score > 0.5) * (overflow_score > 0.48)
                    valid_percent = valid_score * 100
                    over_percent = overflow_score * 100
    with left_col:
        st.subheader("Welcome User :wave:")
        st.title("IITK DMC Overfill Portal")
        st.write("To intimate DMC utilities, upload or click a picture of Overfilled Dustbin area:")
        st.write("File's Uploaded :", Uploaded)
        st.write("Validity Percentage Confidence :",valid_percent,valid_percent > 47)
        st.write("Overflow Percentage Confidence :",over_percent, over_percent > 62)
        st.write("Final Verdict on Notification : ",dmc_call)
        st.write("Thank's for playing your part in keeping the campus clean! :wastebasket:")
        with st.container():
            col1,col2,col3 = st.columns([1,2,1])
            with col2:
                st_lottie(lottie,height=400)
st.cache_data.clear()
st.cache_resource.clear()
############################################################################################################
# VISIT THE LINK BELOW FOR ONLINE HOSTED WEBSITE
# https://dmchktn-asz5iommxbns4z4rrs7kgu.streamlit.app/

#############################################################################################################

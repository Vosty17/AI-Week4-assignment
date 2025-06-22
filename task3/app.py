import streamlit as st
import joblib
import cv2
import numpy as np

# App title
st.title("Medical Image Classifier")

# 1. Load the pre-trained model
model = joblib.load('task3/breast_cancer_detector_grayscale.pkl')
# 2. File uploader
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 3. Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    # 4. Preprocess image (convert to grayscale and resize)
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_array = cv2.resize(img_array, (128, 128)).flatten().reshape(1, -1)
    
    # 5. Make prediction when button is clicked
    if st.button("Classify"):
        prediction = model.predict(img_array)[0]
        probabilities = model.predict_proba(img_array)[0]
        
        # 6. Show results
        st.write("## Results")
        st.write(f"**Prediction:** {'Benign' if prediction == 0 else 'Malignant'}")
        st.write(f"**Confidence:** {max(probabilities)*100:.1f}%")
        st.write(f"- Benign probability: {probabilities[0]*100:.1f}%")
        st.write(f"- Malignant probability: {probabilities[1]*100:.1f}%")

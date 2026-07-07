# app.py
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")

st.title("🔢 MNIST Digit Recognition (ANN)")
st.write("FastAPI backend aur Streamlit frontend ka integration.")

st.markdown("---")

# File uploader widget
uploaded_file = st.file_uploader("Koi bhi handwritten digit (0-9) ki image upload karein...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Image ko display karna
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("🔄 Backend API se prediction li ja rahi hai...")
    
    # Image ko bytes mein convert karna API ko bhejne ke liye
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format if image.format else 'PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # FastAPI Endpoint URL
    api_url = "http://127.0.0.1:8000/predict"
    
    try:
        # API Request send karna
        files = {"file": (uploaded_file.name, img_byte_arr, uploaded_file.type)}
        response = requests.post(api_url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            
            # Results display karna
            st.success("🎉 Prediction Successful!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Digit", value=result["prediction"])
            with col2:
                st.metric(label="Confidence Level", value=result["confidence"])
        else:
            st.error(f"Backend se error aya: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Backend server (FastAPI) nahi chal raha. Pehle `main.py` ko run karein.")
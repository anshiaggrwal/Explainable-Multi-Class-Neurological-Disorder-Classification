import streamlit as st
import requests #to send image to backend
from PIL import Image
import io # to handle image data

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title = " Neurological Disorder Classification ",
    layout = "centered"
)

st.title("🧠 Brain MRI Disorder Classification System")
st.write("Upload your brain MRI to check for potential neurological disorders. This system will analyse the image and provide a prediction along with an explaination of the disorder.")

uploaded_file = st.file_uploader(
    "Upload your brain MRI. Accepted formats: (.jpg, .jpeg, .png)",
    type = ["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption = "Uploaded Brain MRI", width = 250)
    
    with st.spinner("Analysing youe MRI..."):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }
        response = requests.post(API_URL, files = files)
        if response.status_code == 200:
            data = response.json()
            st.success("Analysis Completed")

            #prediction result
            st.subheader("Prediction Result")
            st.write(f"**Your predicted case is** {data['prediction']}")
            st.write(f"**Confidence:** {data['confidence']}%")

            # Explaination 
            st.subheader("📝 What This Result Means")
            st.write(data["explaination"])

            # Visualization
            st.subheader("🔍 Where the Model Focused")
            gradcam_url = f"http://localhost:8000{data["gradcam_image"]}"
            st.image(gradcam_url, caption = "Highlighted MRI Areas", width = 250)
        else:
            st.error("Error connecting to backend API. Please try again.")


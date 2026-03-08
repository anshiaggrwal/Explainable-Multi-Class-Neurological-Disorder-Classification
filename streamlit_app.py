import streamlit as st
import requests #to send image to backend
from PIL import Image
import io # to handle image data

API_URL = "https://anshiagarwal-brain-mri-api.hf.space/predict"

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
    
    with st.spinner("Analyzing your MRI..."):
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

            # Warnings 
            if data["prediction"] == "Normal":
                dementia_confidence = (
                    data["all_probs"].get("AD_VeryMildDemented", 0) +
                    data["all_probs"].get("AD_MildDemented", 0) +
                    data["all_probs"].get("AD_ModerateDemented", 0)
                )
                if dementia_confidence > 20:
                    st.warning(
                        "⚠️ Although predicted case is Normal, there is a noticeable probability of dementia-related changes. "
                        f"The combined probability of Mild or Moderate Dementia is {dementia_confidence}%."
                        " Further medical evaluation is recommended."
                    )
                else:
                    st.warning(
                        "⚠️ While the model predicts a normal case, it's important to remember that no AI model is perfect. "
                        "There still might be a chance of Mild or Moderate Dementia. "
                        "If you have any symptoms or concerns, please consult a healthcare professional for a comprehensive evaluation."
                    )
            elif data["confidence"] < 70:
                st.warning(
                    "⚠️ The model's confidence in this prediction is relatively low. "
                    "Consider consulting a healthcare professional for a more comprehensive evaluation."
                )
                
            # Explaination 
            st.subheader("📝 What This Result Means")
            st.write(data["explanation"])

            # Visualization
            st.subheader("🔍 Where the Model Focused")
            gradcam_url = f"http://localhost:8000{data['gradcam_image']}"
            st.image(gradcam_url, caption = "Highlighted MRI Areas", width = 250)

            st.subheader("🩺 Need a Specialist?")
            st.write(
                "If you have any concerns about your results or symptoms, search for highly rated neurologists near your city."
            )
            city = st.text_input("Enter you city name to find neurologists near you: ")

            if city.strip() != "":
                search_url = f"https://www.google.com/maps/search/top+rated+neurologists+near+{city.replace(' ', '+')}"
                st.markdown(f"🔎[Find top rated neurologists near {city}](%s)" % search_url)
        else:
            st.error("Error connecting to backend API. Please try again.")


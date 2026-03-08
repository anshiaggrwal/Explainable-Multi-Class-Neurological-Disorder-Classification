# 🧠 Brain MRI Dementia Screening System

An AI-assisted web application for analyzing brain MRI scans and
identifying potential dementia stages using deep learning and
explainable AI.

This project integrates a **PyTorch deep learning model**, **FastAPI
backend**, and **Streamlit frontend** to provide an end-to-end system
for MRI image analysis.

------------------------------------------------------------------------
## Check Live

🌐 **Live Demo:** [neurological-disorder-identification.streamlit.app](https://neurological-disorder-identification.streamlit.app/)

🔗 **Backend API:** [anshiagarwal-brain-mri-api.hf.space](https://anshiagarwal-brain-mri-api.hf.space/docs)



------------------------------------------------------------------------

## 🚀 Features

-   MRI image classification using a **PyTorch CNN model**
-   **FastAPI backend** for model inference
-   **Streamlit frontend** for user-friendly interaction
-   **Grad-CAM explainability** to visualize important brain regions
-   **Confidence-aware warnings** for uncertain predictions
-   **Dementia probability analysis** across multiple classes
-   **Neurologist search support** via Google Maps integration
-   Interactive MRI visualization

------------------------------------------------------------------------

## 🧠 Dementia Classes

The model predicts the following categories:

-   Normal\
-   AD Very Mild Demented\
-   AD Mild Demented\
-   AD Moderate Demented\
-   Multiple Sclerosis (MS)\
-   Brain Tumor -- Glioma\
-   Brain Tumor -- Meningioma\
-   Brain Tumor -- Pituitary

------------------------------------------------------------------------

## 🏗 Project Architecture

User Upload MRI\
↓\
Streamlit Frontend\
↓\
FastAPI Backend\
↓\
PyTorch Model Prediction\
↓\
Grad-CAM Visualization + Results

------------------------------------------------------------------------

## 📁 Project Structure

repo\
│\
├── app\
│ ├── main.py \# FastAPI application\
│ ├── predict.py \# Model inference logic\
│ ├── model_loader.py \# Load trained PyTorch model\
│ ├── gradcam.py \# Grad-CAM visualization\
│ └── disorder_info.py \# Disorder explanations\
│\
├── saved_model\
│ └── rexnet_model_best\_\*.pth\
│\
├── static\
│ └── gradcam \# Generated Grad-CAM images\
│\
├── streamlit_app.py \# Streamlit frontend\
├── requirements.txt \# Project dependencies\
├── .gitignore\
└── README.md

------------------------------------------------------------------------

## ⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/mri-dementia-screening.git

cd mri-dementia-screening

Install dependencies:

pip install -r requirements.txt

------------------------------------------------------------------------

## ▶️ Run Backend (FastAPI)

uvicorn app.main:app --reload

API will start at:

http://127.0.0.1:8000

Swagger docs:

http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## ▶️ Run Frontend (Streamlit)

streamlit run streamlit_app.py

The web interface will open in your browser.

------------------------------------------------------------------------

## 🔥 Grad-CAM Explainability

Grad-CAM highlights brain regions influencing model predictions, helping
interpret how the AI model makes decisions.

This improves **transparency and trust** in medical AI systems.

------------------------------------------------------------------------

## ⚠️ Medical Disclaimer

This AI tool is intended for **screening and educational purposes
only**.\
It does **not replace professional medical diagnosis or consultation**.

Users are advised to consult qualified medical professionals for
clinical decisions.

------------------------------------------------------------------------

## 🌐 Deployment

Backend deployed using:

-   **Hugging Face Spaces**

Frontend deployed using:

-   **Streamlit Cloud**

------------------------------------------------------------------------

## 👩‍💻 Author

**Anshi Agarwal**\
BTech Computer Science Engineering (AIML)

------------------------------------------------------------------------

## 📜 License

This project is intended for **educational and research purposes**.

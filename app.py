import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import onnxruntime as ort
from PIL import Image
import easyocr

# ---- Load CSV with All Columns ----
@st.cache_data
def load_data():
    df = pd.read_csv("medicines.csv").dropna(how='all')
    df.dropna(axis=1, how='all', inplace=True)
    df.fillna('', inplace=True)
    df['combined_text'] = df.astype(str).apply(lambda row: ' '.join(row.values), axis=1).str.lower()
    return df, df['combined_text'].tolist()

df, text_list = load_data()

# ---- Load Tokenizer & ONNX Model ----
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    ort_session = ort.InferenceSession("sentence_model.onnx")
    return tokenizer, ort_session

tokenizer, ort_session = load_model()

# ---- EasyOCR Extract Text Function (your original) ----
def extract_text_from_image(image_file):
    reader = easyocr.Reader(['en'], gpu=False)
    image = Image.open(image_file).convert("RGB")
    result = reader.readtext(np.array(image))
    return " ".join([text[1] for text in result])

# ---- Encode input text using ONNX model ----
def encode_text_with_onnx(text):
    inputs = tokenizer(text, return_tensors='np', padding='max_length', truncation=True, max_length=128)
    input_ids = inputs['input_ids'].astype('int64')  # ONNX requires int64
    outputs = ort_session.run(None, {'input_ids': input_ids})
    return np.mean(outputs[0], axis=1)

# ---- Precompute embeddings for all medicines ----
@st.cache_resource
def get_all_embeddings():
    return np.vstack([encode_text_with_onnx(text) for text in text_list])

embeddings = get_all_embeddings()

# ---- Recommender ----
def recommend_meds(symptom_text, top_k=5):
    query_emb = encode_text_with_onnx(symptom_text)
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return df.iloc[top_indices].drop(columns=['combined_text'])

# ---- Streamlit UI ----
st.set_page_config(page_title="Medicine Recommender", page_icon="💊")
st.title("💊 Smart Medicine Recommender")
st.markdown("Upload a **medicine image** or enter **symptoms** to get top recommended medicines.")

# Input selection
option = st.radio("Choose Input Type:", ["📷 Upload Image", "✍️ Enter Symptoms"])

if option == "📷 Upload Image":
    uploaded_file = st.file_uploader("Upload Prescription Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="📸 Uploaded Image", use_column_width=True)
        with st.spinner("🔍 Extracting text from image..."):
            extracted_text = extract_text_from_image(uploaded_file)
        st.text_area("🧠 Extracted Text from OCR", extracted_text, height=100)
        if st.button("💡 Recommend Medicines"):
            results = recommend_meds(extracted_text)
            st.success("✅ Recommendations based on extracted content:")
            st.dataframe(results)

elif option == "✍️ Enter Symptoms":
    user_text = st.text_input("📝 Describe symptoms (e.g., fever, cough, body pain)")
    if st.button("💡 Recommend Medicines"):
        results = recommend_meds(user_text)
        st.success("✅ Recommendations based on your symptoms:")
        st.dataframe(results)

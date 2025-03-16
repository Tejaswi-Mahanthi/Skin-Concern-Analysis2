import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Load YOLOv8 models
skin_type_model = YOLO(r"C:\Users\HP\Downloads\best (8).pt")  # Classification Model for Skin Type
skin_problem_model = YOLO(r"C:\Users\HP\Downloads\best (9).pt")  # Segmentation Model for Skin Problems

# Firebase Setup
firebase_key_path = r"C:\Users\HP\PycharmProjects\Skin Disease Prediction\firebase_key.json"
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_key_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

CATEGORY_NAMES = {
    0: "Acne", 1: "Dark Circle", 2: "Dark Spot", 3: "Dry Skin",
    4: "Normal Skin", 5: "Oily Skin", 6: "Pores",
    7: "Skin Redness", 8: "Wrinkles"
}

CATEGORY_COLORS = {
    0: (0, 0, 255), 1: (128, 0, 128), 2: (0, 0, 128), 3: (165, 42, 42),
    4: (0, 255, 0), 5: (255, 165, 0), 6: (255, 255, 0),
    7: (255, 0, 0), 8: (192, 192, 192)
}

# Streamlit UI Configuration
st.set_page_config(page_title="AI Skin Analysis", layout="wide")
st.markdown("""
    <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run Skin Type Classification
    classification_results = skin_type_model(image)
    skin_type = classification_results[0].names[classification_results[0].probs.top1]

    # Run Skin Problem Detection
    results = skin_problem_model(image)

    # Create two columns: one for image, one for analysis summary
    col1, col2 = st.columns([1, 1])

    with col1:
        resized_image = cv2.resize(image, (250, 200))
        st.subheader("Original Image")
        st.image(Image.fromarray(resized_image), caption="Uploaded Image", use_container_width=False)

    with col2:
        st.subheader("📝 Analysis Summary")
        category_masks = {cls: np.zeros_like(image, dtype=np.uint8) for cls in CATEGORY_NAMES.keys()}
        detected_conditions = set()

        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                for seg_mask, cls in zip(result.masks.xy, result.boxes.cls):
                    points = np.array([seg_mask], np.int32)
                    cls_id = int(cls)
                    color = CATEGORY_COLORS.get(cls_id, (255, 255, 255))
                    cv2.fillPoly(category_masks[cls_id], points, color)
                    detected_conditions.add(cls_id)

        skin_problems = [CATEGORY_NAMES[cls_id] for cls_id in detected_conditions if
                         CATEGORY_NAMES[cls_id] not in ["Normal Skin", "Oily Skin", "Dry Skin"]]

        col_skin_type, col_skin_problems = st.columns(2)

        with col_skin_type:
            st.markdown("### 🌿 Skin Type")
            st.markdown(f"- ✅ {skin_type}")

        with col_skin_problems:
            st.markdown("### ⚠️ Skin Problems")
            for problem in skin_problems:
                st.markdown(f"- ❌ {problem}")
            if not skin_problems:
                st.success("No major skin problems detected!")

    for cls_id in detected_conditions:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader(f"Segmented Image for {CATEGORY_NAMES[cls_id]}")
            segmented_img = cv2.addWeighted(image, 0.7, category_masks[cls_id], 0.3, 0)
            resized_segmented_img = cv2.resize(segmented_img, (250, 200))
            st.image(Image.fromarray(resized_segmented_img), caption=f"{CATEGORY_NAMES[cls_id]}",
                     use_container_width=False)

        with col2:
            st.subheader(f"🛙️ Recommended Products for {CATEGORY_NAMES[cls_id]}")
            products_ref = db.collection('Products').where("Problem", "==", CATEGORY_NAMES[cls_id])
            products = list(products_ref.stream())

            if products:
                # Pagination logic for showing 3 products at a time
                product_index = st.session_state.get(f'product_index_{cls_id}', 0)
                num_products = len(products)
                num_cols = 3  # Show 3 products in a row

                # Navigation buttons
                cols_nav = st.columns([1, 3, 1])
                with cols_nav[0]:
                    if st.button("⬅️", key=f'prev_{cls_id}'):
                        product_index = (product_index - num_cols) % num_products

                with cols_nav[2]:
                    if st.button("➡️", key=f'next_{cls_id}'):
                        product_index = (product_index + num_cols) % num_products

                st.session_state[f'product_index_{cls_id}'] = product_index

                # Display products in a row
                product_columns = st.columns(num_cols)
                for i in range(num_cols):
                    product_idx = product_index + i
                    if product_idx < num_products:
                        product_data = products[product_idx].to_dict()
                        with product_columns[i]:
                            st.markdown(f"**{product_data.get('Product', 'Unnamed Product')}**")
                            st.image(product_data.get('Img_URL', ''), width=150)
                            st.write(product_data.get('Description', 'No description available.'))
                            st.write(f"💰 Price: ₹{product_data.get('Price', 'N/A')}")
                            st.markdown(f"[🛒 Buy Now]({product_data.get('Prod_URL', '#')})", unsafe_allow_html=True)

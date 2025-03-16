import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import firebase_admin
from firebase_admin import credentials, firestore
import base64
import os


st.set_page_config(page_title="Skin Analysis App", layout="wide")


# ✅ Function to Convert Image to Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# ✅ Convert Pic4.webp to Base64
pic4_path = "static/images/Pic4.webp"
pic4_base64 = get_base64_image(pic4_path)


# ✅ Convert Pic5.jpg to Base64
pic5_path = "static/images/Pic5.jpg"
pic5_base64 = get_base64_image(pic5_path)

# ✅ Inject CSS & HTML
st.markdown(f"""
    <style>
        .block-container {{ padding: 0 !important; margin: 0 !important; }}
        header {{ visibility: hidden; height: 0px; }}  

        .navbar {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-color: #A94064;
            padding: 8px 20px;
            box-shadow: 0px 3px 5px rgba(0, 0, 0, 0.1);
            z-index: 9999;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .service-number {{ color: white; font-size: 16px; font-weight: bold; margin-left: 15px; }}
        .brand-name {{ color: white; font-size: 22px; font-weight: bold; text-align: center; flex-grow: 1; }}
        .right-section {{ display: flex; align-items: center; gap: 15px; margin-right: 15px; }}
        .search-bar {{
            padding: 6px 12px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            width: 250px;
        }}
        .icon {{ font-size: 20px; color: white; cursor: pointer; }}

        .content {{ margin-top: 0px; }}

        /* ✅ 3rd Section - Image Carousel */
        .carousel-container {{
            width: 100vw;
            height: 90vh;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            position: relative;
        }}

        .carousel-image {{
            position: absolute;
            width: 100vw;
            height: 90vh;
            object-fit: cover;
            opacity: 0;
            animation: fade 9s infinite;
        }}

        .carousel-image:nth-child(1) {{ animation-delay: 0s; }}
        .carousel-image:nth-child(2) {{ animation-delay: 3s; }}
        .carousel-image:nth-child(3) {{ animation-delay: 6s; }}

        @keyframes fade {{
            0% {{ opacity: 0; }}
            10% {{ opacity: 1; }}
            30% {{ opacity: 1; }}
            40% {{ opacity: 0; }}
            100% {{ opacity: 0; }}
        }}

        /* ✅ 4th Section: Full-Width Background Image */
        .background-section {{
            position: relative;
            width: 100vw;
            height: 90vh;
            background: url("data:image/webp;base64,{pic4_base64}") no-repeat center center/cover;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            color: white;
        }}

        /* ✅ Left-aligned ANSKIN Text */
        .brand-overlay {{
            position: absolute;
            top: 20%;
            left: 15%;
            font-size: 60px;
            font-weight: bold;
            color: white;
            text-shadow: 3px 3px 5px rgba(0, 0, 0, 0.9);
            opacity: 0;
            animation: fadeIn 2s ease-in-out forwards 1s;
        }}

        /* ✅ Left-aligned Skin Analysis Text */
        .skin-analysis-text {{
            position: absolute;
            top: 45%;
            left: 15%;
            font-size: 24px;
            font-weight: bold;
            color: white;
            line-height: 1.5;
            text-align: left;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
            opacity: 0;
            animation: fadeIn 2s ease-in-out forwards 1.5s;
        }}

        /* ✅ Quote Overlay */
        .quote-overlay {{
            position: absolute;
            top: 50%;
            left: 78%;
            transform: translate(-50%, -50%);
            padding: 15px;
            border-radius: 10px;
            font-size: 25px;
            font-weight: bold;
            line-height: 1.4;
            text-align: left;
            width: 80%;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.9);
            opacity: 0;
            animation: fadeIn 2s ease-in-out forwards 1s, neonGlow 3s infinite alternate ease-in-out;
        }}

        @keyframes fadeIn {{
            0% {{ opacity: 0; transform: translate(-50%, -55%); }}
            100% {{ opacity: 1; transform: translate(-50%, -50%); }}
        }}

        @keyframes neonGlow {{
            0% {{ text-shadow: 0 0 5px #ff80ab, 0 0 10px #ff4081, 0 0 15px #ff80ab; }}
            100% {{ text-shadow: 0 0 10px #ff4081, 0 0 20px #ff80ab, 0 0 30px #ff4081; }}
        }}
        .browse-button {{
            margin-top: 10px;
            padding: 10px 20px;
            font-size: 20px;
            font-weight: bold;
            color: white;
            background-color: #A94064;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }}
        
        /* ✅ 5th Section: Full-Width Background Image */
        .background-section-5 {{
            position: relative;
            width: 100vw;
            height: 90vh;
            background: url("data:image/jpeg;base64,{pic5_base64}") no-repeat center center/cover;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
        }}

        .capture-button {{
            position: absolute;
            top: 70%;
            left: 90%;
            transform: translate(-50%, -50%);
            padding: 15px 30px;
            font-size: 22px;
            font-weight: bold;
            color: white;
            background-color: #A94064;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.3s ease-in-out;
        }}

        .capture-button:hover {{
            background-color: #8b324f;
        }}
    </style>
""", unsafe_allow_html=True)

# ✅ Navigation Bar
st.markdown("""
    <div class="navbar">
        <div class="service-number">📞 Service Number: 1234567890</div>
        <div class="brand-name">AnSkin</div>
        <div class="right-section">
            <input type="text" class="search-bar" placeholder="🔍 Search">
            <span class="icon">❤️</span>
            <span class="icon">🛋</span>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='content'></div>", unsafe_allow_html=True)

# ✅ 3rd Section - Image Carousel with Working Animation
image_folder = "static/images"
image_files = ["Pic1.webp", "Pic2.webp", "Pic3.webp"]
image_b64_list = [get_base64_image(os.path.join(image_folder, img)) for img in image_files]

carousel_html = f"""
<div class="carousel-container">
    {''.join(f'<img class="carousel-image" src="data:image/webp;base64,{img}">' for img in image_b64_list)}
</div>
"""
st.markdown(carousel_html, unsafe_allow_html=True)

# ✅ 4th Section: Full-Width Background
st.markdown("""
    <div class="background-section">
        <div class="brand-overlay">ANSKIN</div>  
        <div class="skin-analysis-text">
            Analyse your <br>
            skin problems and <br>
            skin types and <br>
            know your right <br>
            skincare routine
        <form action="/Browse">
                <button class="browse-button" type="submit">Browse</button>
        </form>
        </div>
        <div class="quote-overlay">
            The best makeup<br> foundation<br>
            you can wear<br>
            is glowing skin<br>
        </div>
        
    </div>
""", unsafe_allow_html=True)

# ✅ 5th Section: Full-Width Background
st.markdown(f"""
    <div class="background-section-5">
        <form action="/Capture">
            <button class="capture-button" type="submit">Capture</button>
        </form>
    </div>
""", unsafe_allow_html=True)
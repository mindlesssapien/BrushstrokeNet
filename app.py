import streamlit as st
from PIL import Image
from main import *

st.title('BrushstrokeNet')

style_image = st.file_uploader("Choose a style image...", type=["jpg", "png"])
content_image = st.file_uploader("Choose a content image...", type=["jpg", "png"])

if style_image is not None and content_image is not None:
    style_img = Image.open(style_image)
    content_img = Image.open(content_image)
    col1, col2 = st.columns(2)
    col1.image(style_img, caption='Uploaded Style Image.', use_container_width=True)
    col2.image(content_img, caption='Uploaded Content Image.', use_container_width=True)
    st.write("")
    st.write("Generating new image...")

    target_img = train(style_image,content_image)

    st.write(f"Final Result: {target_img}")

import streamlit as st
from downloader import download_instagram_photos
from face_sorter import sort_faces
from style_analyzer import generate_prompt

st.title("AI Style Inspector")

username = st.text_input("Enter Instagram username (without @):")
run_button = st.button("Analyze Style")

if run_button and username:
    st.write("Downloading photos...")
    download_instagram_photos(username)

    st.write("Detecting main face and sorting photos...")
    sort_faces()

    st.write("Generating style description...")
    prompt = generate_prompt("clothing_visible")
    st.success("Style prompt generated:")
    st.text(prompt)

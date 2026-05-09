import streamlit as st
from PIL import Image

# streamlit page configuration
st.set_page_config(page_title="Emotion Music Recommender", layout="wide")


st.title("Emotion-Based Music Recommendation System 🎵 ")
st.markdown("---")

# layout with two columns
col1, col2 = st.columns([1, 1])

with col1:

    st.header("Input Image 📸")

    input_mode = st.radio("Choose input source:", ["Upload Image", "Live Camera"])

    # remains None until user provides image
    image = None

    if input_mode == "Upload Image":
        # file uploader for local files
        uploaded_file = st.file_uploader("Select a photo from your device...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            # open uploaded image using pillow
            image = Image.open(uploaded_file)

    else:
        # camera input for real-time photos
        camera_photo = st.camera_input("Take a picture of your face")

        if camera_photo:
            # convert captured image into pillow object
            image = Image.open(camera_photo)

    # display image if provided
    if image:
        st.image(image, caption="Selected Image for Analysis", width='stretch')

with col2:

    st.header("Analysis & Recommendations 📊")

    if image:
        # container for future emotion detection results
        st.subheader("Detected Emotion")
        st.info("The model analysis will be displayed here in Step 3.")
        
        # progress bar placeholder for probabilities
        st.markdown("**Probability Distribution:**")
        st.progress(0)
        
        # container for future music recommendations
        st.markdown("---")
        st.subheader("Music Recommendation")
        st.info("Music suggestions based on your mood will appear here in Step 4.")
    else:
        # feedback for empty state
        st.warning("Please provide an image using the controls on the left to start.")
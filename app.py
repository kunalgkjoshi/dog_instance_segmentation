import streamlit as st
from instance_segmentation import *

def detect_on_image(x):
    detector = Detector(model_type=x)
    image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
    if image_file is not None:
        file_details = {"FileName":image_file.name,"FileType":image_file.type}
        st.write(file_details)
        img = Image.open(image_file)
        st.image(img, caption='Uploaded Image.')
        with open(image_file.name,mode = "wb") as f: 
            f.write(image_file.getbuffer())         
        st.success("Saved File")
        detector.onImage(image_file.name)
        img_ = Image.open("result.jpg")
        st.image(img_, caption='Proccesed Image.')


def main():
    with st.expander("About the App"):
        st.markdown( '<p style="font-size: 30px;"><strong>Welcome to my Instance Segmentation App!</strong></p>', unsafe_allow_html= True)
        

    option = st.selectbox(
     'What Type of File do you want to work with?',
     ('Images', ' '))

    
    if option == "Images":
        st.title('Instance Segmentation for Images')
        st.subheader("""
    This takes an image as an input, and provides image with bounding box and mask as an output.
    """)
        detect_on_image('instance_segmentation')


if __name__ == '__main__':
		main()
import streamlit as st
import random
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import process
#hide 
def main():
    import warnings
    warnings.filterwarnings("ignore")



    # set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
    st.set_page_config(
        page_title="Traditional ML App",
        page_icon="🧊",
        initial_sidebar_state='auto'
    )
    file = st.file_uploader("", type=["jpg", "png"])
    if file is None:
        st.text("Please upload an image file")
    else:
        os.makedirs("temp", exist_ok=True)
        image = Image.open(file)
        st.image(image, use_column_width=False)

        with open(os.path.join("temp", file.name), "wb") as f:
            f.write(file.getbuffer())
        
        # Get the file path of the saved file
        file_path = os.path.join("temp", file.name)

        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_img, name = process.run(img)
        st.image(processed_img, use_column_width=False)
        st.write(f"Predicted Traffic Sign: {name}")
        #shutil.rmtree("temp")




if __name__ == "__main__":
    main()



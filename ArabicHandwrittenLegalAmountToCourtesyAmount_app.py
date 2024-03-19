import streamlit as st
import numpy as np
from PIL import Image
import torch
import os
import pytesseract
import pandas,csv
import re
from LegaltoCourtesyAmount import *


st.set_page_config(layout="wide")
st.sidebar.header("Arabic Handwritten Legal Amount Sentences Recognition")
st.sidebar.markdown("---")
container=st.container()
image_url=''
with container:
   uploaded_file = st.sidebar.file_uploader('Upload a Legal Amount image', type=['jpg'])
   if uploaded_file is not None:
     image = Image.open(uploaded_file)
     image.save(os.path.join('uploaded_file',uploaded_file.name ))
     image_url=os.path.join('uploaded_file',uploaded_file.name)
     legal_amount_path=image_url
     ou_path="runs/OutputWords/"+str(uploaded_file.name)
     st.write("**Orignal Legal Amount**")
     st.image(image)
     if st.button('Recognize and convert'):
       st.markdown("---")
       st.write("**The detection and cropping of legal Amount Words.**")
       legal_word_path=ou_path+"/*.jpg"
       imageWithboxes=legelcrop(legal_amount_path,ou_path)
       st.image(imageWithboxes)
       st.markdown("---")
       st.write("**The vlaue of courtesy amount from legal amount**")
       val_legal=str(get_courtesy_from_legal(legal_word_path))+" R"
       st.write(val_legal)
           








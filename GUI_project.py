import streamlit as st
import cv2
from PIL import Image
import numpy as np
from dropboxes import *

import RRDBNet_arch as arch
import torch


icon = Image.open("icons/logo_TIM.jpg")
st.set_page_config(page_title="FotoChipi", page_icon=icon, layout="wide")

WIDTH = 500

image = Image.open('icons/logo_TIM-removebg.png')
#st.image(image, use_column_width=True, width=10)

col1, col2 = st.columns([1,6])  # Créer deux colonnes

with col1:
    st.image(image, width=150)  # Afficher l'image dans la première colonne

with col2:
    st.write("")



device = torch.device('cuda')

#@st.cache(allow_output_mutation=True)
st.cache_resource()
def load_model():
    model_path = 'models/RRDB_ESRGAN_x4.pth'
    #device = torch.device('cuda')
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model

def upscale_image(model, img):
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    return output



def main():
    page = st.sidebar.selectbox(
        "Select a page:",
        [
        "Home",
        "Image Enhancement 'Gray Edition'",
        "Image Enhancement 'Color Edition'",
        "Noise Reduction",
        "Edge Detection",
        "==> ESRGAN <=="
        ]
    )
    if page=="Home":
        st.header("Welcome To Foto Chipi chipi!!!")
        st.image("icons/logo_TIM-removebg.png")
        """
        ## **From:**
        ### Amine MAALEM
        ### Boualem MOKEDEM
        """
    
    
    
    if page== "Image Enhancement 'Gray Edition'":
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg'])
        if uploaded_file is not None:
            #fig = plt.figure(figsize=(7,5))
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            #image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            col1, col2, col4 = st.columns(3)

            # Afficher l'image originale dans la première colonne
            col1.image(image, caption='Image téléchargée.', width=WIDTH)

            # Afficher l'image en niveaux de gris dans la dernière colonne
            col4.image(image_gray, caption='Image téléchargée en gray scale.', width=WIDTH)
            drop_box_gray(image_gray, uploaded_file)
            

        
    if page=="Image Enhancement 'Color Edition'":
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Image téléchargée.', width=WIDTH)
            drop_box_color(image, uploaded_file)
            
    if page=="Noise Reduction":
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Image téléchargée.', width=WIDTH)
            drop_box_noise(image, uploaded_file)
            
    if page=="Edge Detection":
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Image téléchargée.', width=WIDTH)
            drop_box_edge(image, uploaded_file)
    if page=="==> ESRGAN <==":
        model = load_model()

        st.title('ESRGAN Super Resolution')
        uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg","bmp"])
        if uploaded_file is not None:
            img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            st.image(img, channels="BGR", width=600)
            st.write("")
            st.write("Processing...")
            output = upscale_image(model, img)
            output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            st.image(output_pil, width=600)
            save_image(output_pil, uploaded_file, "ESRGAN")








if __name__ == "__main__":
    main()
import streamlit as st
import cv2
import numpy as np
from image_processing import image_enhancement as ie
from PIL import Image







WIDTH = 500

def save_image(image, uploaded_file, transformation_name):
    if st.button('Save the image'):
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray((image * 255).astype(np.uint8))
        """
        image = Image.fromarray((image).astype(np.uint8))
        file_name = uploaded_file.name.split(".")[0]
        
        image.save(file_name +"_"+ transformation_name +'.png')
        st.write(f'Image saved successfully! as {file_name}_{transformation_name}.png')


def drop_box_gray(image_gray, uploaded_file):
    if image_gray is not None:
        dropbox = st.selectbox("select a trasformation",
                                [
                                    "None",
                                    "image quatization",
                                    "windowing",
                                    "logarithmic intensity",
                                    "gamma intensity",
                                    "inverse"
                                ]
                                )
        if dropbox == "image quatization":
            pixel_depth = st.slider('Choisissez la profondeur des pixels', min_value=1, max_value=256, value=4)
            image = ie.image_quatization(image_gray,pixel_depth)
            image = image.astype(np.float32) / 255.0
            st.image(image, caption=f'avec quatization {pixel_depth}', width=WIDTH)
            save_image(image, uploaded_file, "quatization")
        if dropbox == "windowing":
            upper_bound = st.slider('Choisissez la borne max', min_value=2, max_value=256, value=128)
            lower_bound = st.slider('Choisissez la borne min', min_value=1, max_value=upper_bound, value=int(upper_bound/2))

            image = ie.windowing(image_gray,lower_bound=lower_bound,upper_bound=upper_bound)
            st.image(image, caption=f'Upper Bound {upper_bound} and Lower Bound {lower_bound}', width=WIDTH)
            save_image(image, uploaded_file, "windowing")
            
        if dropbox == "logarithmic intensity":
            image = ie.transf_log_intensity(image_gray)
            image = np.clip(image, 0, 255)
            image = image.astype(np.float32) / 255.0
            st.image(image, caption=f"trasformation logarithmic", width=WIDTH)
            save_image(image, uploaded_file, "logarithmic")
            
        if dropbox == "gamma intensity":
            gamma = st.slider("valeur de gamma :", min_value=0.0, max_value=2.0, value=1.0,step=0.1)
            image = ie.gamma(image_gray, gamma=gamma)
            image = np.clip(image, 0, 255)
            image = image.astype(np.float32) / 255.0
            #image = image.astype(np.float32) / 255.0
            st.image(image, caption=f"transformation with gamma = {gamma}", width=WIDTH)
            save_image(image, uploaded_file, "gamma")
        if dropbox == "inverse":
            image = ie.inverse(image_gray)
            st.image(image, caption=f"inverse color of the image", width=WIDTH)
            save_image(image, uploaded_file, "inverse")

def drop_box_color(image, uploaded_file):
    if image is not None:
        image_np = np.array(image)
        image_yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
    
        dropbox = st.selectbox("select a trasformation",
                                [
                                    "None",
                                    "image quatization",
                                    "windowing",
                                    "logarithmic intensity",
                                    "gamma intensity",
                                    "inverse",
                                    "all channels",
                                    "all inverse",
                                    "histogram equalizer mk1",
                                    "histogram equalizer mk2",
                                    "histogram stretching"
                                    ])
        
        if dropbox == "image quatization":
            pixel_depth = st.slider('Choisissez la profondeur des pixels', min_value=1, max_value=256, value=4)
            image_yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
            image_yuv[:,:,0] = ie.image_quatization(image_yuv[:,:,0],pixel_depth)
            image = cv2.cvtColor(image_yuv,cv2.COLOR_YUV2RGB)
            image = image.astype(np.float32) / 255.0
            st.image(image, caption=f'avec quatization {pixel_depth}', width=WIDTH)
            save_image(image, uploaded_file, "quatization")
        if dropbox == "windowing":
            upper_bound = st.slider('Choisissez la borne max', min_value=2, max_value=256, value=128)
            lower_bound = st.slider('Choisissez la borne min', min_value=1, max_value=upper_bound, value=int(upper_bound/2))

            image_yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
            image_yuv[:,:,0] = ie.windowing(image_yuv[:,:,0],lower_bound=lower_bound,upper_bound=upper_bound)
            image = cv2.cvtColor(image_yuv,cv2.COLOR_YUV2RGB)
            st.image(image, caption=f'Upper Bound {upper_bound} and Lower Bound {lower_bound}', width=WIDTH)
            save_image(image, uploaded_file, "windowing")
            
        if dropbox == "logarithmic intensity":
            image_yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
            image_yuv[:,:,0] = ie.transf_log_intensity(image_yuv[:,:,0])
            image = cv2.cvtColor(image_yuv,cv2.COLOR_YUV2RGB)
            image = np.clip(image, 0, 255)
            image = image.astype(np.float32) / 255.0
            st.image(image, caption=f"trasformation logarithmic", width=WIDTH)
            save_image(image, uploaded_file, "logarithmic")
            
        if dropbox == "gamma intensity":
            gamma = st.slider("valeur de gamma :", min_value=0.0, max_value=2.0, value=1.0,step=0.1)
            image_yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
            image_yuv[:,:,0] = ie.gamma(image_yuv[:,:,0], gamma=gamma)
            image = cv2.cvtColor(image_yuv,cv2.COLOR_YUV2RGB)
            image = np.clip(image, 0, 255)
            image = image.astype(np.float32) / 255.0
            #image = image.astype(np.float32) / 255.0
            st.image(image, caption=f"transformation with gamma = {gamma}", width=WIDTH)
            save_image(image, uploaded_file, "gamma")
        if dropbox=="inverse":
            image = ie.inverse(image_np)
            st.image(image, caption="the inverse of the image", width=WIDTH)
            save_image(image, uploaded_file, "inverse")
        
        if dropbox == "all channels":
            r, g, b = cv2.split(image_np)
            r_color = cv2.merge([r, np.zeros_like(r), np.zeros_like(r)])
            g_color = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
            b_color = cv2.merge([np.zeros_like(b), np.zeros_like(b), b])
        
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.image(r_color, caption="the Red channel", width=WIDTH//2)
            #save_image(r_color, uploaded_file, "red_channel")
            
            col3.image(g_color, caption="the Green channel", width=WIDTH//2)
            #save_image(g_color, uploaded_file, "green_channel")
            
            col5.image(b_color, caption="the Blue channel", width=WIDTH//2)
            #save_image(b_color, uploaded_file, "blue_channel")
        
        if dropbox=="all inverse":
            r,g,b = ie.inverse_all(image_np)
            r_color = cv2.merge([r, np.zeros_like(r), np.zeros_like(r)])
            g_color = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
            b_color = cv2.merge([np.zeros_like(b), np.zeros_like(b), b])
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.image(r_color, caption="the Red channel", width=WIDTH//2)
            #save_image(image, uploaded_file, "inverse_red_channel")
            
            col3.image(g_color, caption="the Green channel", width=WIDTH//2)
            #save_image(image, uploaded_file, "inverse_green_channel")
            
            col5.image(b_color, caption="the Blue channel", width=WIDTH//2)
            #save_image(image, uploaded_file, "inverse_blue_channel")
        
        if dropbox=="histogram equalizer mk1":
            image = ie.hist_equalizer_mk1(image_np)
            st.image(image, caption="the equalized image", width=WIDTH)
            save_image(image, uploaded_file, "hiqtEqualizedMk1")
            
        if dropbox=="histogram equalizer mk2":
            image_mk2 = ie.hist_equalizer_mk2(image_np)
            st.image(image_mk2, caption="the equalized image", width=WIDTH)
            save_image(image_mk2, uploaded_file, "hiqtEqualizedMk2")
            
        if dropbox=="histogram stretching":
            image = ie.hist_stretch(image_np)
            st.image(image, caption="the stretch image", width=WIDTH)
            save_image(image, uploaded_file, "hiqtStretch")

def drop_box_noise(image, uploaded_file):
    if image is not None:
        image = np.array(image)
    
        dropbox = st.selectbox("select a trasformation",
                                [
                                    "None",
                                    "Mean",
                                    "Gaussain",
                                    "Median",
                                ]
                                )
        
        if dropbox == "Mean":
            kernel_size = st.slider("choose the size for the kernel :", min_value=1, max_value=100, value=5)
            #max_value=np.min(image.shape[0],image.shape[1])
            image = cv2.blur(image, (kernel_size, kernel_size))
            st.image(image, caption=f"your image with the mean filter with kernel {kernel_size}x{kernel_size}", width=WIDTH)
            save_image(image, uploaded_file, "mean_filter")
        if dropbox == "Gaussain":
            kernel_size = st.slider("choose the size for the kernel :", min_value=1, max_value=101, step=2)
            std_v = st.slider("choose the std varience:", min_value=0, max_value=100)
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), std_v)
            st.image(image, caption=f"your image with the gaussain filter with kernel {kernel_size}x{kernel_size} and std={std_v}", width=WIDTH)
            save_image(image, uploaded_file, "gaussain_filter")
        if dropbox == "Median":
            kernel_size = st.slider("choose the size for the kernel :", min_value=1, max_value=101, step=2)
            image = cv2.medianBlur(image, kernel_size)
            st.image(image, caption=f"your image with the median filter with kernel {kernel_size}x{kernel_size}", width=WIDTH)
            save_image(image, uploaded_file, "median_filter")
            
def drop_box_edge(image, uploaded_file):
    if image is not None:
        image = np.array(image)
    
        dropbox = st.selectbox("select a trasformation",
                                [
                                    "None",
                                    "Canny",
                                    "",
                                    "",
                                ]
                                )
        if dropbox == "Canny":
            
            max = st.slider("choose the upperbound of canny",min_value=0,max_value=255, value=200)
            min = st.slider("choose the lowerbound of canny",min_value=0,max_value=max, value=int(max/2))
            
            
            image = cv2.Canny(image, threshold1=min, threshold2=max)
            st.image(image, caption=f"canny detector between {min} and {max}", width=WIDTH)
            save_image(image, uploaded_file, "canny_detector")
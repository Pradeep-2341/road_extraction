import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from skimage.morphology import skeletonize

st.title("Road Extraction from Satellite Images")

input_file_name = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]).name

if input_file_name is not None:

	image = cv2.imread(input_file_name)
	st.image(image, caption="Satellite Image", use_column_width=True)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
	edges = cv2.Canny(gray_blur, 50 , 150)
	_, thresh = cv2.threshold(edges, 50, 150, cv2.THRESH_BINARY)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
	closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	thinned = skeletonize(closed)
	thinned_int = (thinned * 255).astype(np.uint8)  # Convert to integer type
	st.image(thinned_int, caption="Extracted Road", use_column_width=True)

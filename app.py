import streamlit as st
from PIL import Image, ExifTags
from ultralytics import YOLO
import pytesseract
import cv2
import numpy as np

st.title("Traitement automatique des notes de frais")
model = YOLO("best.pt")


uploaded_file = st.file_uploader("Déposez votre note de frais : ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    classes_cibles = [0, 1, 2, 3]
    
    results = model.predict(image, conf=0.2, imgsz = 736, save = True)
    bboxes = results[0].boxes.xyxy
    labels = results[0].boxes.cls
    
    texts = {"Date": "", "Lieu": "", "Total": "", "TVA": ""}
    
    for classe in classes_cibles:
        bboxes_classe = bboxes[labels == classe]
        if classe == 0:
            for bbox in bboxes_classe:
                xmin, ymin, xmax, ymax = map(int, bbox)
                sub_image = img[ymin:ymax, xmin:xmax]
                texte = pytesseract.image_to_string(sub_image, lang='fra', config = '--psm 6')
                texts["Date"] += texte + " "

        elif classe == 1:
            for bbox in bboxes_classe:
                xmin, ymin, xmax, ymax = map(int, bbox)
                sub_image = img[ymin:ymax, xmin:xmax]
                texte = pytesseract.image_to_string(sub_image, lang='fra')
                texts["Lieu"] += texte + " "

        elif classe == 2:
            for bbox in bboxes_classe:
                xmin, ymin, xmax, ymax = map(int, bbox)
                sub_image = img[ymin:ymax, xmin:xmax]
                texte = pytesseract.image_to_string(sub_image, lang='fra')
                texts["Total"] += texte + " "

        elif classe == 3:
            for bbox in bboxes_classe:
                xmin, ymin, xmax, ymax = map(int, bbox)
                sub_image = img[ymin:ymax, xmin:xmax]
                texte = pytesseract.image_to_string(sub_image, lang='fra')
                texts["TVA"] += texte + " "
    
    img_with_boxes = Image.fromarray(img)

    st.image(img_with_boxes, caption='Image', use_column_width=True)

    
    st.write("Résultats :")
    texts["Date"] = st.text_area("Date", texts["Date"])
    texts["Lieu"] = st.text_area("Lieu", texts["Lieu"])
    texts["Total"] = st.text_area("Total", texts["Total"])
    texts["TVA"] = st.text_area("TVA", texts["TVA"])

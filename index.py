import cv2
import numpy as np
import pytesseract
img = cv2.VideoCapture(2)
while(True):
    ret, leer = img.read()#cargar realtime
    img_gray = cv2.cvtColor(leer,cv2.COLOR_RGB2GRAY)
    cv2.imshow('imagen', img_gray)
    #-----------------------------------------------------------------------------efecto en tiempo real
    noise_removal = cv2.bilateralFilter(img_gray,9,75,75)#eliminar los ruidos
    equal_histogram = cv2.equalizeHist(noise_removal)#histograma
    cv2.imshow("Histograma",equal_histogram)
    #------------------------------------------------------------------------------------------silueta 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))#Apertura morfológica(cuadricular)
    morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=15)
    sub_morp_image = cv2.subtract(equal_histogram,morph_image)#Sustracción de imagen
    ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)#Umbralizar la imagen

    canny_image = cv2.Canny(thresh_image,250,255)#detección de bordes(detección Canny Edge)
    canny_image = cv2.convertScaleAbs(canny_image)

    kernel = np.ones((3,3), np.uint8)#dilatación del grosor del bordes
    dilated_image = cv2.dilate(canny_image,kernel,iterations=1)#Creando el núcleo para la dilatación
    mask = np.zeros(img_gray.shape,np.uint8)#Enmascarar la parte que no sea la matrícula
    cv2.imshow("dilatación", dilated_image)
    #------------------------------------------------------------------------------------------- detectar objecto
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
img.release()
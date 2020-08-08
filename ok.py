import cv2
import numpy as np
import pytesseract

img = cv2.imread("placa10.jpeg")#cargar imagen
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.imshow("hola",img_gray)
#-----------------------------------------------------------------------------efecto de imagen
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
cv2.imshow("dilatación", dilated_image)
#------------------------------------------------------------------------------------------- detectar objecto
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#Encontrar contornos en la imagen basados en bordes
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None #Ordene los contornos según el área, de modo que la placa de número esté en los 10 contornos principales
for c in contours: #recorrer nuestros contornos
    peri = cv2.arcLength(c, True)# aproximar el contorno
    approx = cv2.approxPolyDP(c, 0.06 * peri, True) #Aproximacion con 6% de error
    #si nuestro contorno aproximado tiene cuatro puntos, entonces #podemos suponer que hemos encontrado nuestra pantalla
    if len(approx) == 4: # Seleccione el contorno con 4 esquinas
        screenCnt = approx
        break
final = cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)# Dibujando el contorno seleccionado en la imagen original
#---------------------------------------------------------------------------------------------- Eliminar lo que este fiera del border
mask = np.zeros(img_gray.shape,np.uint8)#Enmascarar la parte que no sea la matrícula
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)
y,cr,cb = cv2.split(cv2.cvtColor(new_image,cv2.COLOR_RGB2YCrCb))#Histograma igual para mejorar la matrícula para su posterior procesamiento
y = cv2.equalizeHist(y)#Convertir la imagen al modelo YCrCb y dividir los 3 canales
#------------------------------------------------------------
# la imagen final sacarle el texto

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = img_gray[topx:bottomx+1, topy:bottomy+1]
#Read the number plate
text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("Numero de placa:",text)
cv2.imshow('Cropped',Cropped)
cv2.waitKey()
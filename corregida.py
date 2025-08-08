import cv2
import numpy as np

# Cargar los parámetros de calibración guardados
data = np.load('parametros_calibracion.npz')
mtx, dist = data['mtx'], data['dist']

# Cargar la imagen a corregir
img = cv2.imread('corregir3.jpg')
h, w = img.shape[:2]

# Corregir la distorsión
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Mostrar resultado
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Corregida', cv2.WINDOW_NORMAL)
cv2.imshow('Original', img)
cv2.imshow('Corregida', undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()

import os
import cv2
import numpy as np
import glob

# === CONFIGURACIÓN DEL PATRÓN ===
pattern_size = (9, 6)       # Número de esquinas internas (columnas, filas)
square_size = 470          # Tamaño real de un cuadro del tablero (en mm o unidad que uses)

# === GENERAR LOS PUNTOS 3D DEL MUNDO REAL PARA EL TABLERO ===
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# === LISTAS PARA GUARDAR PUNTOS 3D (reales) Y 2D (imagen) ===
objpoints = []
imgpoints = []
image_shape = None  # Guardaremos aquí las dimensiones de la primera imagen válida

# === CREAR CARPETA DE SALIDA SI NO EXISTE ===
os.makedirs('detectadas', exist_ok=True)

# === LEER TODAS LAS IMÁGENES DESDE LA CARPETA ===
images = glob.glob('./Images/*.jpg') # Asegúrate que las imágenes estén en esta carpeta

print("Archivos encontrados:", images)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Buscar esquinas del tablero
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # Guardar forma de imagen (solo la primera vez válida)
        if image_shape is None:
            image_shape = gray.shape[::-1]

        # Refinar ubicación de esquinas
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners)

        # Dibujar y mostrar la imagen
        img_drawn = img.copy()
        cv2.drawChessboardCorners(img_drawn, pattern_size, corners, ret)

        # Guardar imagen con esquinas en carpeta nueva
        nombre_archivo = os.path.basename(fname)
        cv2.imwrite(f'detectadas/{nombre_archivo}', img_drawn)

print("Se guardaron las imagenes con sus puntos en una carpeta llamada detectadas")

# === VERIFICAR QUE SE HAYAN DETECTADO ESQUINAS ===
if not objpoints:
    print("❌ No se detectaron esquinas en ninguna imagen.")
    exit()

# === CALIBRAR LA CÁMARA ===
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

# === GUARDAR PARÁMETROS CALCULADOS ===
np.savez('parametros_calibracion.npz', mtx=mtx, dist=dist)
print("✅ Calibración completada. Parámetros guardados en 'parametros_calibracion.npz'.")

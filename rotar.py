import cv2
import numpy as np
import math

# === Cargar imagen ===
img = cv2.imread('inclinada1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === Detección de bordes ===
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# === Detectar líneas con Hough ===
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

angles = []

if lines is not None:
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90  # Convertimos a grados
        if -45 < angle < 45:  # Filtrar líneas horizontales/inclinadas
            angles.append(angle)

if not angles:
    print("❌ No se encontraron líneas dominantes para calcular el ángulo.")
    exit()

# === Calcular el ángulo promedio
avg_angle = np.mean(angles)
print(f"🔁 Rotando {avg_angle:.2f}° para enderezar...")

# === Rotar imagen
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

# === Mostrar resultado
cv2.namedWindow("Imagen corregida automáticamente", cv2.WINDOW_NORMAL)
cv2.imshow("Imagen corregida automáticamente", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()


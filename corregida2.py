import cv2
import numpy as np

# === Cargar parámetros de calibración ===
data = np.load('parametros_calibracion.npz')
mtx, dist = data['mtx'], data['dist']

# === Cargar y corregir la imagen inclinada ===
img = cv2.imread('inclinada1.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

# === Selección de puntos manual para homografía ===
pts = []

def seleccionar_puntos(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([x, y])
        cv2.circle(undistorted, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Selecciona 4 puntos', undistorted)

cv2.imshow('Selecciona 4 puntos', undistorted)
cv2.setMouseCallback('Selecciona 4 puntos', seleccionar_puntos)
cv2.waitKey(0)
cv2.destroyAllWindows()

# === Aplicar homografía ===
if len(pts) == 4:
    src_pts = np.array(pts, dtype='float32')

    # Puedes ajustar estos valores según el tamaño del objeto "enderezado"
    width, height = 300, 600
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype='float32')

    # Matriz de transformación y warping
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    rectificada = cv2.warpPerspective(undistorted, M, (width, height))

    # Mostrar resultado final
    cv2.imshow('Imagen corregida y rectificada', rectificada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Debes seleccionar exactamente 4 puntos.")

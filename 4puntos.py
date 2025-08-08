import cv2
import numpy as np
import os

# === Cargar la imagen ===
img = cv2.imread('corregir3.jpg')
clone = img.copy()
preview = img.copy()
points = []

# === Mouse callback ===
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(preview, (x, y), 5, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(preview, points[-2], points[-1], (255, 0, 0), 2)
            if len(points) == 4:
                cv2.line(preview, points[3], points[0], (255, 0, 0), 2)
            cv2.imshow("Selecciona 4 puntos", preview)

# === Crear ventana y esperar selección ===
cv2.namedWindow("Selecciona 4 puntos", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Selecciona 4 puntos", click_event)
cv2.imshow("Selecciona 4 puntos", preview)

print("🖱️ Haz clic en 4 puntos (esquinas) en orden horario: arriba izq → arriba der → abajo der → abajo izq.")

while True:
    if len(points) == 4:
        break
    key = cv2.waitKey(1)
    if key == 27:  # ESC para salir
        print("Cancelado por el usuario.")
        exit()

cv2.destroyAllWindows()

# === Calcular dimensiones ===
pts_src = np.array(points, dtype="float32")

widthA = np.linalg.norm(pts_src[0] - pts_src[1])
widthB = np.linalg.norm(pts_src[2] - pts_src[3])
heightA = np.linalg.norm(pts_src[0] - pts_src[3])
heightB = np.linalg.norm(pts_src[1] - pts_src[2])
maxWidth = int(max(widthA, widthB))
maxHeight = int(max(heightA, heightB))

# Validación del área
if maxWidth < 50 or maxHeight < 50:
    print("⚠️ El área seleccionada es muy pequeña. Intenta seleccionar una región más grande.")
    exit()

# === Puntos de destino ===
pts_dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
], dtype="float32")

# === Homografía y transformación ===
M = cv2.getPerspectiveTransform(pts_src, pts_dst)
warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

# === Guardar resultado ===
os.makedirs("enderezadas", exist_ok=True)
output_path = os.path.join("enderezadas", "imagen_enderezada.jpg")
cv2.imwrite(output_path, warped)
print(f"✅ Imagen enderezada guardada en: {output_path}")

# === Mostrar resultado ===
cv2.namedWindow("Imagen enderezada", cv2.WINDOW_NORMAL)
cv2.imshow("Imagen enderezada", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

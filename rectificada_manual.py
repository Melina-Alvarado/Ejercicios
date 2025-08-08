import cv2
import numpy as np

# === Cargar imagen ===
img = cv2.imread("inclinada1.jpg")
if img is None:
    print("❌ No se pudo cargar la imagen.")
    exit()

# === Lista para guardar clics ===
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Selecciona 4 puntos", img)
        if len(points) == 4:
            cv2.destroyAllWindows()

print("🖱️ Haz clic en los 4 vértices del número 1: sup. izq → sup. der → inf. der → inf. izq")
cv2.imshow("Selecciona 4 puntos", img)
cv2.setMouseCallback("Selecciona 4 puntos", click_event)
cv2.waitKey(0)

if len(points) != 4:
    print("❌ Se requieren exactamente 4 puntos.")
    exit()

src_pts = np.float32(points)

# === Calcular dimensiones de destino automáticamente ===
width_top = np.linalg.norm(src_pts[0] - src_pts[1])
width_bottom = np.linalg.norm(src_pts[3] - src_pts[2])
maxWidth = int(max(width_top, width_bottom))

height_left = np.linalg.norm(src_pts[0] - src_pts[3])
height_right = np.linalg.norm(src_pts[1] - src_pts[2])
maxHeight = int(max(height_left, height_right))

dst_pts = np.float32([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
])

# === Aplicar homografía ===
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

# === Mostrar resultado ===
cv2.imshow("Imagen rectificada", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()


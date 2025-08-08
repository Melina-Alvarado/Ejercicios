import cv2
import numpy as np

# === Cargar imagen ===
img = cv2.imread("corregir3.jpg")
if img is None:
    print("❌ Imagen no cargada.")
    exit()

# === Selección de 4 puntos manual ===
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Selecciona 4 puntos", img)
        if len(points) == 4:
            cv2.destroyAllWindows()

cv2.namedWindow("Selecciona 4 puntos", cv2.WINDOW_NORMAL)
cv2.imshow("Selecciona 4 puntos", img)
cv2.setMouseCallback("Selecciona 4 puntos", click_event)
cv2.waitKey(0)

if len(points) != 4:
    print("❌ Se requieren 4 puntos.")
    exit()

# === Homografía ===
src_pts = np.float32(points)
width = int(max(
    np.linalg.norm(src_pts[0] - src_pts[1]),
    np.linalg.norm(src_pts[2] - src_pts[3])
))
height = int(max(
    np.linalg.norm(src_pts[0] - src_pts[3]),
    np.linalg.norm(src_pts[1] - src_pts[2])
))
dst_pts = np.float32([
    [0, 0],
    [width, 0],
    [width, height],
    [0, height]
])
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
rectificada = cv2.warpPerspective(img, M, (width, height))

# === Paso adicional: encontrar orientación del objeto interno ===
gray = cv2.cvtColor(rectificada, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Tomar contorno más grande
if not contours:
    print("❌ No se encontró ningún contorno.")
    exit()

c = max(contours, key=cv2.contourArea)
rect = cv2.minAreaRect(c)  # (center, size, angle)
angle = rect[2]

# Ajustar ángulo para alinear verticalmente
if angle < -45:
    angle += 90

print(f"🔁 Corrigiendo inclinación del objeto: {angle:.2f}°")

# === Rotar para alinear el objeto ===
(h, w) = rectificada.shape[:2]
center = (w // 2, h // 2)
R = cv2.getRotationMatrix2D(center, angle, 1.0)
final = cv2.warpAffine(rectificada, R, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# === Mostrar y guardar resultado final ===
cv2.namedWindow("Resultado Final", cv2.WINDOW_NORMAL)
cv2.imshow("Resultado Final", final)
cv2.imwrite("resultado_final.jpg", final)
cv2.waitKey(0)
cv2.destroyAllWindows()


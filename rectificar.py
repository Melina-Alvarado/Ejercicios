import cv2
import numpy as np

# === Cargar imagen ===
img = cv2.imread("inclinada1.jpg")
if img is None:
    print("❌ No se pudo cargar la imagen.")
    exit()

# === Preprocesamiento ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# === Encontrar contornos ===
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# === Buscar el contorno más grande que parezca un rectángulo ===
max_area = 0
best_approx = None

for cnt in contours:
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) == 4:
        area = cv2.contourArea(approx)
        if area > max_area:
            max_area = area
            best_approx = approx

if best_approx is None:
    print("❌ No se encontró ningún rectángulo.")
    exit()

# === Ordenar puntos (superior izq, sup der, inf der, inf izq) ===
pts = best_approx.reshape(4, 2)
rect = np.zeros((4, 2), dtype="float32")

s = pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]  # top-left
rect[2] = pts[np.argmax(s)]  # bottom-right

diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)]  # top-right
rect[3] = pts[np.argmax(diff)]  # bottom-left

# === Calcular dimensiones del nuevo plano ===
widthA = np.linalg.norm(rect[2] - rect[3])
widthB = np.linalg.norm(rect[1] - rect[0])
maxWidth = int(max(widthA, widthB))

heightA = np.linalg.norm(rect[1] - rect[2])
heightB = np.linalg.norm(rect[0] - rect[3])
maxHeight = int(max(heightA, heightB))

dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
], dtype="float32")

# === Calcular homografía y transformar ===
M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

# === Mostrar resultados ===
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Rectificada', cv2.WINDOW_NORMAL)
cv2.imshow("Original", img)
cv2.imshow("Rectificada", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()


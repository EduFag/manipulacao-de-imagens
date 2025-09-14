import cv2
import numpy as np

# Carregar imagem
img = cv2.imread('fruit1.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel de cruz que aprendemos em aula
kernel = np.array(([0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]), np.uint8)

_, binary = cv2.threshold(img_gray, 190, 255, cv2.THRESH_BINARY_INV)

binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)  # fecha buracos close
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)   # remove ruído open

mascara = cv2.bitwise_and(img, img, mask=binary)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

# Bounding boxes
img_boxes = img.copy()
count = 0
for i in range(1, num_labels):  # ignora o fundo (i = 0)
    count += 1
    x, y, w, h, _ = stats[i]
    cx, cy = centroids[i]

    # Desenha retângulo
    cv2.rectangle(img_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Marca centróide
    cv2.circle(img_boxes, (int(cx), int(cy)), 4, (0, 0, 255), -1)
    # Nome da fruta
    cv2.putText(img_boxes, f"Fruta {count}", (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

print("Frutas detectadas:", count)

# Mostrar resultados
cv2.imshow('Mascara aplicada', mascara)
cv2.imshow('Mascara binaria limpa', binary)
cv2.imshow('Bounding Boxes', img_boxes)

cv2.waitKey(0)
cv2.destroyAllWindows()

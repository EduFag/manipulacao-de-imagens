import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

img = cv2.imread('templo1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = gray.astype(np.int16)

kernel = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2]
], dtype=np.int16)

start_time = time.time()

# convolução
embossed = cv2.filter2D(gray, ddepth=-1, kernel=kernel)

end_time = time.time()
tempo_execucao = end_time - start_time

# adiciona 128 e limita os valores entre 0 e 255
embossed = np.clip(embossed + 128, 0, 255).astype(np.uint8)


print(f"Tempo de execução: {tempo_execucao:.4f} segundos")

plt.imshow(embossed, cmap='gray')
plt.title('Imagem com efeito Emboss')
plt.axis('off')
plt.show()

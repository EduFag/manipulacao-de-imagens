import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

img = cv2.imread('templo1.jpg')

kernel = np.array([[-1, -1, -1], 
                   [-1,  8, -1], 
                   [-1, -1, -1]])

start_time = time.time()

# convolução
resultado = cv2.filter2D(img, -1, kernel)

end_time = time.time()
tempo_execucao = end_time - start_time

print(f"Tempo de execução: {tempo_execucao:.4f} segundos")

plt.imshow(resultado)
plt.title('Imagem com efeito detecção de bordas')
plt.show()
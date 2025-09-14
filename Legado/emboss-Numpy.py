import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
# Carregar imagem em escala de cinza
img_path = 'templo1.jpg'
img = np.array(Image.open(img_path).convert('L'))

# Kernel de Emboss  
kernel = np.array([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2]
], dtype=np.float32)

def calcular_pixel(matriz_imagem, kernel):
    soma = 0
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            soma += int(matriz_imagem[i, j]) * int(kernel[i, j])
    return soma + 128  

start_time = time.time()

def convolucao(imagem, kernel):
    tamanho_kernel = kernel.shape[0]
    img_h, img_w = imagem.shape

    deslocamento_horizontal = img_w - tamanho_kernel + 1
    deslocamento_vertical = img_h - tamanho_kernel + 1

    matriz_resultante = []

    for i in range(deslocamento_vertical):
        linha = []
        for j in range(deslocamento_horizontal):
            parte = imagem[i:i+tamanho_kernel, j:j+tamanho_kernel]
            pixel = calcular_pixel(parte, kernel)
            linha.append(pixel)
        matriz_resultante.append(linha)

    return np.clip(matriz_resultante, 0, 255).astype(np.uint8)

# Aplica convolução
resultado = convolucao(img, kernel)

end_time = time.time()
tempo_execucao = end_time - start_time

# Exibir resultado
plt.imshow(resultado, cmap='gray')
print(f"Tempo de execução: {tempo_execucao:.4f} segundos")
plt.title('Imagem com efeito Emboss')
plt.axis('off')
plt.show()

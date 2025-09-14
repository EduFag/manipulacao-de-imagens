# Manipulação de Imagens
Trabalho focado no entendimento de manipulação de imagens.

## Convolução e Efeitos
### Código feito apenas com Numpy
```
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img_path = 'templo1.jpg'
img = np.array(Image.open(img_path).convert('RGB'))

kernel = np.array([[-1, -1, -1], 
                   [-1,  8, -1], 
                   [-1, -1, -1]])

def calcular_pixel(matriz_imagem, kernel):
    resultado = []
    pixel = 0
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            resultado.append(int(matriz_imagem[i, j]) * int(kernel[i, j]))
    for n in resultado:
        pixel += n
    return pixel

def convolucao(imagem, kernel):
    tamanho_kernel = len(kernel)
    img_w = imagem.shape[1]
    img_h = imagem.shape[0]

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

canais = []
for c in range(3):
    canal = img[:, :, c]
    conv = convolucao(canal, kernel)
    canais.append(conv)

resultado = np.dstack(canais)

plt.imshow(resultado)
plt.title('Imagem RGB com kernel')
plt.axis('off')
plt.show()
```
### Código feito com OpenCV
```

```

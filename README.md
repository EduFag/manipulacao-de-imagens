# Manipulação de Imagens
Trabalho focado no entendimento de manipulação de imagens.

## Convolução e Efeitos
### Código feito apenas com Numpy
📌 Resumo das Funcionalidades

- Utiliza o conceito de convolução para aplicar um kernel sobre uma imagem, calculando os pixels e fazendo o deslocamento do kernel
- Carrega uma imagem RGB.
- Define um kernel de convolução (detecção de bordas).
- Implementa convolução manual (sem usar OpenCV ou funções prontas).
- Aplica a convolução em cada canal RGB separadamente.
- Reconstrói a imagem filtrada.
- Mede e mostra o tempo de execução.
- Exibe o resultado em uma janela gráfica.

```
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

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
start_time = time.time()
for c in range(3):
    canal = img[:, :, c]
    conv = convolucao(canal, kernel)
    canais.append(conv)

resultado = np.dstack(canais)
end_time = time.time()
tempo_execucao = end_time - start_time

print(f"Tempo de execução: {tempo_execucao:.4f} segundos")

plt.imshow(resultado)
plt.title('Imagem RGB com kernel')
plt.axis('off')
plt.show()

```
### Código feito com OpenCV
- O openCV é uma biblioteca que já faz todo o processo anterior de maneira otimizada e prática.
```
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

```

## Detecção de Componentes
📌 Resumo das Funcionalidades:

Ambos os códigos, tanto para de 5 ou 7 frutas, são bem parecidos, eu usei a Matriz de Cruz como foi ensinado em aula, é melhor por que ele pega uma fração menor, para ter melhor precisão.

Foi usado direto o Open e Close, passado as iterações para conseguir ter total precisão para conseguir identificar, ele deixa mais quadrado, mas é a melhor forma de conseguir precisão para ele identificar somente as frutas.

Para fazer a parte da Box, Centroide e Label usei bem o que estava nos slides.

### 5 Frutas:

```
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

```
### 7 Frutas:

Aqui é notável que tem bem mais iterações, mas é a única maneira de conseguir encontrar as frutas com muita diferença de cor.

```
import cv2
import numpy as np

# Carregar imagem
img = cv2.imread('fruit3.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel de cruz que aprendemos em aula
kernel = np.array(([0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]), np.uint8)

_, binary = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY_INV)

binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=9)  # fecha buracos
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=15)   # remove ruído

mascara = cv2.bitwise_and(img, img, mask=binary)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

# Bounding boxes
img_boxes = img.copy()
count = 0
for i in range(1, num_labels):
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
```

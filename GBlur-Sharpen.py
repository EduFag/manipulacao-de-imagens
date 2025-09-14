
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. Carregando a Imagem ---
try:
    img = np.array(Image.open('templo1.jpg').convert('RGB'))
    print("Imagem carregada com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivo 'templo1.jpg' não encontrado.")
    exit()

# --- 2. Definindo os Kernels de Convolução Padrão ---
kernels = {
    "Gaussian Blur": np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ]),
    "Sharpen": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
}

# --- 3. Funções de Convolução Manual ---
def calcular_pixel(fragmento_imagem, kernel):
    soma = 0.0  # Usar float para a soma
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            soma += fragmento_imagem[i, j] * kernel[i, j]
    return soma

def convolucao_manual(imagem_canal, kernel):
    tamanho_kernel = kernel.shape[0]
    altura_img, largura_img = imagem_canal.shape

    altura_resultado = altura_img - tamanho_kernel + 1
    largura_resultado = largura_img - tamanho_kernel + 1
    
    matriz_resultante = [] # Começa como uma lista vazia

    for i in range(altura_resultado):
        linha = [] # Cria uma lista para cada linha
        for j in range(largura_resultado):
            fragmento = imagem_canal[i:i+tamanho_kernel, j:j+tamanho_kernel]
            pixel = calcular_pixel(fragmento, kernel)
            linha.append(pixel)
        matriz_resultante.append(linha) # Adiciona a linha à matriz
    
    # A conversão para array NumPy é feita no final
    return np.array(matriz_resultante)

# --- 4. Aplicando os Kernels Padrão ---
for nome_kernel, kernel in kernels.items():
    print(f"Aplicando o kernel: {nome_kernel}...")
    
    canais_processados = []
    for i in range(3):
        canal_conv = convolucao_manual(img[:, :, i], kernel)
        canais_processados.append(canal_conv)
        
    imagem_final = np.dstack(canais_processados)
    imagem_final = np.clip(imagem_final, 0, 255).astype(np.uint8)

    plt.imshow(imagem_final)
    plt.title(f'Resultado com Kernel: {nome_kernel}')
    plt.axis('off')
    plt.show()
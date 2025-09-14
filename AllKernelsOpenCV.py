import numpy as np
import cv2 # Importando a biblioteca OpenCV
import matplotlib.pyplot as plt
import time

# --- 1. Carregando a Imagem com OpenCV ---
# cv2.imread carrega a imagem diretamente como um array NumPy no formato BGR
try:
    img_bgr = cv2.imread('templo1.jpg')
    if img_bgr is None:
        raise FileNotFoundError
    print("Imagem carregada com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivo 'templo1.jpg' não encontrado ou formato inválido.")
    exit()

# --- 2. Definindo os Kernels de Convolução (sem alterações) ---
kernels = {
    "Lapasciano (Detecção de Bordas)": np.array([
        [-1, -1, -1], 
        [-1,  8, -1], 
        [-1, -1, -1]
    ]),
    "Emboss": np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2]
    ]),
    "Gaussian Blur": np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ]),
    "Sharpen": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ]),
    "Emboss Diagonal": np.array([
        [-1, -1, 0],
        [-1,  0, 1],
        [ 0,  1, 0]
    ]),
}

# --- 3. Aplicando os Kernels com a Função Otimizada do OpenCV ---
for nome_kernel, kernel in kernels.items():
    start_time = time.time()
    print(f"Aplicando o kernel: {nome_kernel}...")
    
    # A função cv2.filter2D aplica o kernel em todos os canais da imagem de uma vez.
    # O argumento -1 indica que a profundidade de bits da imagem de saída será a mesma da imagem de entrada.
    imagem_final_bgr = cv2.filter2D(src=img_bgr, ddepth=-1, kernel=kernel)

    end_time = time.time()
    tempo_execucao = end_time - start_time

    # O Matplotlib espera imagens no formato RGB, enquanto o OpenCV usa BGR.
    # Portanto, precisamos converter o espaço de cores para uma exibição correta.
    imagem_final_rgb = cv2.cvtColor(imagem_final_bgr, cv2.COLOR_BGR2RGB)

    plt.imshow(imagem_final_rgb)
    print(f"Tempo de execução: {tempo_execucao:.4f} segundos")
    plt.title(f'Resultado com Kernel: {nome_kernel}')
    plt.axis('off')
    plt.show()
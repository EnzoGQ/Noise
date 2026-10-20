
from PIL import Image
import numpy as np

def gerar_noise(tamanho=(128, 128)):
    rng = np.random.default_rng()
    noise = rng.integers(
        0, 256,
        size=(tamanho[1], tamanho[0]),
        dtype=np.uint8
    )
    return Image.fromarray(noise, mode="L")


def suavizar_noise_exponencial(img, intensidade=2.0):
    """
    Suaviza o noise usando uma função exponencial.

    intensidade:
        0.0 = mantém o noise original
        valores maiores = aproxima os pixels da média
    """
    if intensidade < 0:
        raise ValueError("A intensidade deve ser >= 0.")

    pixels = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    # Curva exponencial simétrica em torno de 0.5
    k = intensidade
    if k == 0:
        suavizado = pixels
    else:
        x = pixels - 0.5
        suavizado = 0.5 + np.sign(x) * (
            np.expm1(k * np.abs(x)) / np.expm1(k * 0.5)
        ) * 0.5

    resultado = np.clip(suavizado * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(resultado, mode="L")


noise = gerar_noise()
noise_suavizado = suavizar_noise_exponencial(
    noise,
    intensidade=2.0
)

noise.save("noise_128x128.png")
noise_suavizado.save("noise_exponencial_128x128.png")

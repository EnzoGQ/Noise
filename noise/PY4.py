from PIL import Image
import numpy as np


def gerar_noise(tamanho=(128, 128)):
    rng = np.random.default_rng()
    return rng.random((tamanho[1], tamanho[0]), dtype=np.float32)  # 0..1


def curva_exponencial(p, k):
    if k == 0:
        return p
    x = p - 0.5
    return 0.5 + np.sign(x) * (np.expm1(k * np.abs(x)) / np.expm1(k * 0.5)) * 0.5


def curva_linear(p, k):
    return 0.5 + (p - 0.5) / (1.0 + k)


def noise_misturado(tamanho=(128, 128), intensidade=2.0, mistura=0.5):
    """
    Mistura a suavização exponencial com a linear por interpolação linear (lerp):
        resultado = (1 - mistura) * exponencial + mistura * linear

    intensidade: >= 0 (0 = noise original)
    mistura:     0.0 = só exponencial | 1.0 = só linear
    """
    if intensidade < 0:
        raise ValueError("A intensidade deve ser >= 0.")
    if not 0.0 <= mistura <= 1.0:
        raise ValueError("A mistura deve estar entre 0 e 1.")

    p = gerar_noise(tamanho)
    exp = curva_exponencial(p, intensidade)
    lin = curva_linear(p, intensidade)

    resultado = (1.0 - mistura) * exp + mistura * lin
    return Image.fromarray(np.clip(resultado * 255, 0, 255).astype(np.uint8), mode="L")


noise_misturado(intensidade=2.0, mistura=0.5).save("noise_misto_128x128.png")
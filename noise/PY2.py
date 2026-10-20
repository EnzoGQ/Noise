
from PIL import Image, ImageFilter
import numpy as np

def gerar_noise(tamanho=(128, 128)):
    rng = np.random.default_rng()
    noise = rng.integers(
        0, 256,
        size=(tamanho[1], tamanho[0]),
        dtype=np.uint8
    )
    return Image.fromarray(noise, mode="L")


def suavizar_noise(img, intensidade=1.5):
    """
    Suaviza o noise usando um filtro Gaussiano.

    intensidade:
        0.5 = suavização leve
        1.5 = suavização média
        3.0 = suavização forte
    """
    if intensidade < 0:
        raise ValueError("A intensidade deve ser >= 0.")

    return img.filter(
        ImageFilter.GaussianBlur(radius=intensidade)
    )


# Gerar e suavizar
noise = gerar_noise()
noise_suavizado = suavizar_noise(noise, intensidade=1.5)

noise.save("noise_128x128.png")
noise_suavizado.save("noise_suavizado_128x128.png")

print("Noise original e suavizado criados!")

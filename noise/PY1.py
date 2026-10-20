from PIL import Image
import numpy as np

# Gerar noise aleatório em escala de cinza
rng = np.random.default_rng()
noise = rng.integers(
    0, 256,
    size=(128, 128),
    dtype=np.uint8
)

# Criar imagem e salvar
img = Image.fromarray(noise, mode="L")
img.save("noise_128x128.png")

print("Noise criado: 128x128 pixels")

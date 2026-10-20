from PIL import Image
import numpy as np


def gerar_noise(tamanho=(128, 128), seed=None):
    rng = np.random.default_rng(seed)
    return rng.random((tamanho[1], tamanho[0]))  # float64, 0..1


# 1) Suavização exponencial (por pixel)
def curva_exponencial(p, k):
    if k == 0:
        return p
    x = p - 0.5
    return 0.5 + np.sign(x) * (np.expm1(k * np.abs(x)) / np.expm1(k * 0.5)) * 0.5


# 2) Suavização linear (por pixel)
def curva_linear(p, k):
    return 0.5 + (p - 0.5) / (1.0 + k)


# 3) Mistura exponencial + linear por interpolação linear (lerp)
def curva_misturada(p, k, mistura):
    return (1.0 - mistura) * curva_exponencial(p, k) + mistura * curva_linear(p, k)


# 4) Suavização linear espacial (box blur, tileável)
def box_blur(p, raio):
    if raio == 0:
        return p
    acc = np.zeros_like(p)
    for dy in range(-raio, raio + 1):
        for dx in range(-raio, raio + 1):
            acc += np.roll(p, shift=(dy, dx), axis=(0, 1))
    return acc / (2 * raio + 1) ** 2


# 5) Pixel de retorno = função de e^(k/array[3x3])
def suavizar_3x3_exp_inversa(p, k, eps=1.0 / 255.0):
    """
    Para cada pixel, pega a janela 3x3 (ele + 8 vizinhos) e calcula:
        E     = e^(k / janela)          (array 3x3)
        media = mean(E)
        nova_cor = k / ln(media)        (inverte e^(1/x) de volta para a escala de cor)

    Se a janela é uniforme (todos = v), nova_cor = v.
    k = 0 -> box blur 3x3 comum.
    """
    if k == 0:
        return box_blur(p, 1)

    v = np.maximum(p, eps)
    janela = np.stack([
        np.roll(v, shift=(dy, dx), axis=(0, 1))
        for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    ])  # (9, H, W)

    a = k / janela                       # expoentes de e^(k/v)
    m = a.max(axis=0)
    ln_media = m + np.log(np.exp(a - m).sum(axis=0)) - np.log(9.0)  # ln(mean(e^a)), estável

    return np.clip(k / ln_media, 0.0, 1.0)


def gerar_final(
    tamanho=(128, 128),
    seed=None,
    intensidade=2.0,   # etapa por pixel (0 = desligado)
    mistura=0.5,       # 0 = só exponencial | 1 = só linear
    raio=1,            # box blur (0 = desligado)
    k_3x3=1.0,         # grau de suavização e^(k/array[3x3])
    passes=3,          # 0 = desligado
):
    if intensidade < 0 or k_3x3 < 0 or raio < 0 or passes < 0:
        raise ValueError("Parâmetros não podem ser negativos.")
    if not 0.0 <= mistura <= 1.0:
        raise ValueError("A mistura deve estar entre 0 e 1.")

    p = gerar_noise(tamanho, seed)
    p = curva_misturada(p, intensidade, mistura)
    p = box_blur(p, raio)
    for _ in range(passes):
        p = suavizar_3x3_exp_inversa(p, k_3x3)

    return Image.fromarray(np.clip(p * 255, 0, 255).astype(np.uint8), mode="L")


gerar_final().save("noise_final_128x128.png")
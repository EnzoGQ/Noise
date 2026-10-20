from PIL import Image
import numpy as np

# ============================================================
# 1) TERRENO BASE (todas as suavizações anteriores)
# ============================================================
def gerar_noise(tamanho=(128, 128), seed=None):
    return np.random.default_rng(seed).random((tamanho[1], tamanho[0]))

def curva_exponencial(p, k):
    if k == 0:
        return p
    x = p - 0.5
    return 0.5 + np.sign(x) * (np.expm1(k * np.abs(x)) / np.expm1(k * 0.5)) * 0.5

def curva_linear(p, k):
    return 0.5 + (p - 0.5) / (1.0 + k)

def curva_misturada(p, k, mistura):
    return (1 - mistura) * curva_exponencial(p, k) + mistura * curva_linear(p, k)

def box_blur(p, raio):
    if raio == 0:
        return p
    acc = np.zeros_like(p)
    for dy in range(-raio, raio + 1):
        for dx in range(-raio, raio + 1):
            acc += np.roll(p, (dy, dx), axis=(0, 1))
    return acc / (2 * raio + 1) ** 2

def janela_3x3(p):
    return np.stack([np.roll(p, (dy, dx), axis=(0, 1))
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1)])

def suavizar_3x3_exp_inversa(p, k, eps=1 / 255):
    if k == 0:
        return box_blur(p, 1)
    a = k / janela_3x3(np.maximum(p, eps))
    m = a.max(axis=0)
    ln_media = m + np.log(np.exp(a - m).sum(axis=0)) - np.log(9.0)
    return np.clip(k / ln_media, 0, 1)

def estirar(p):
    lo, hi = p.min(), p.max()
    return (p - lo) / (hi - lo) if hi > lo else p

def gerar_base(tamanho=(128, 128), seed=None, intensidade=2.0, mistura=0.5,
               raio=1, k_3x3=1.0, passes=3):
    p = curva_misturada(gerar_noise(tamanho, seed), intensidade, mistura)
    p = box_blur(p, raio)
    for _ in range(passes):
        p = suavizar_3x3_exp_inversa(p, k_3x3)
    return estirar(p)

# ============================================================
# 2) OPERADOR DE TEMPO  t / ln( mean(e^(t/array[3x3])) + 1 )
# ============================================================
def operador_t(p, t, eps=1 / 255):
    a = t / janela_3x3(np.maximum(p, eps))
    m = a.max(axis=0)
    ln_mean = m + np.log(np.exp(a - m).sum(axis=0)) - np.log(9.0)
    return np.clip(t / np.logaddexp(ln_mean, 0.0), 0, 1)

# ============================================================
# 3) PROCESSOS GEOLÓGICOS
# ============================================================
VIZ = [(dy, dx, np.hypot(dy, dx))
       for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]

def mascara_afloramento(shape, seed=None, raio=8, expoente=2.0):
    m = np.random.default_rng(None if seed is None else seed + 1).random(shape)
    for _ in range(3):
        m = box_blur(m, raio)
    return estirar(m) ** expoente

def soerguimento(h, mask, U0, dt):
    return h + dt * U0 * mask

def receptores(h):
    H, W = h.shape
    decl = np.stack([(h - np.roll(h, (-dy, -dx), axis=(0, 1))) / d for dy, dx, d in VIZ])
    idx = decl.argmax(axis=0)
    mx = decl.max(axis=0)
    yy, xx = np.mgrid[0:H, 0:W]
    dys = np.array([v[0] for v in VIZ])[idx]
    dxs = np.array([v[1] for v in VIZ])[idx]
    rec = ((yy + dys) % H) * W + ((xx + dxs) % W)
    rec = np.where(mx > 0, rec, yy * W + xx).ravel()
    return rec, np.maximum(mx, 0)

def area_drenagem(h, rec):
    A = np.ones(h.size)
    for i in np.argsort(h.ravel())[::-1]:
        r = rec[i]
        if r != i:
            A[r] += A[i]
    return (A / h.size).reshape(h.shape)

def erosao_fluvial(h, K, dt, m=0.5, n=1.0):
    rec, S = receptores(h)
    A = area_drenagem(h, rec)
    ero = dt * K * (A ** m) * (S ** n)
    h_rec = h.ravel()[rec].reshape(h.shape)
    ero = np.minimum(ero, np.maximum(h - h_rec, 0) * 0.9)
    return h - ero

def difusao(h, D, dt):
    lap = (np.roll(h, 1, 0) + np.roll(h, -1, 0) +
           np.roll(h, 1, 1) + np.roll(h, -1, 1) - 4 * h)
    return h + dt * D * lap

def erosao_termica(h, talus, taxa):
    novo = h.copy()
    for dy, dx, d in VIZ:
        dif = h - np.roll(h, (-dy, -dx), axis=(0, 1))
        mov = taxa * np.maximum(dif - talus * d, 0) / (2 * len(VIZ))
        novo -= mov
        novo += np.roll(mov, (dy, dx), axis=(0, 1))
    return novo

def intemperismo(h, t, w):
    h0 = estirar(h)
    alvo = 0.5 * (operador_t(h0, t) + 1 - operador_t(1 - h0, t))
    lo, hi = h.min(), h.max()
    return (1 - w) * h + w * (lo + alvo * (hi - lo))

# ============================================================
# 4) RUGOSIDADE MÍNIMA (impede o terreno de ficar liso)
# ============================================================
RAIOS_BANDA = (0, 1, 2, 4)   # bandas de escala: fina -> grossa

def bandas(p):
    """Decompõe em faixas de frequência: p = soma das bandas + residuo grosso."""
    suaves = [box_blur(p, r) for r in RAIOS_BANDA]
    return [suaves[i] - suaves[i + 1] for i in range(len(RAIOS_BANDA) - 1)]

def rugosidade(h):
    """Desvio-padrão do detalhe por banda (fina -> grossa)."""
    return [b.std() for b in bandas(h)]

def garantir_rugosidade(h, rng, minimo=0.006, persistencia=1.6, mask=None):
    """
    Piso de rugosidade por banda (mais forte nas escalas grossas, como terreno fractal):
        piso_i = minimo * persistencia^i
    Se a banda i tem desvio r_i < piso_i, injeta ruído da mesma banda com amplitude
        sqrt(piso_i² - r_i²)
    de modo que o desvio resultante seja exatamente o piso (ruídos independentes
    somam em variância). Nunca remove detalhe, só repõe o que falta.
    """
    r = rugosidade(h)
    ruido_base = rng.standard_normal(h.shape)
    nb = bandas(ruido_base)
    peso = 1.0 if mask is None else (0.6 + 0.8 * mask)   # soerguido = mais acidentado
    for i, (r_i, b) in enumerate(zip(r, nb)):
        piso = minimo * persistencia ** i
        falta = np.sqrt(max(piso ** 2 - r_i ** 2, 0.0))
        if falta > 0:
            h = h + peso * falta * b / (b.std() + 1e-12)
    return h

# ============================================================
# 5) SIMULAÇÃO
# ============================================================
def passo(h, mask, rng, dt=1.0, U0=0.003, K=0.05, D=0.05,
          talus=0.02, taxa_termica=0.5, t=50.0, w_intemp=0.02,
          rug_min=0.006, persistencia=1.6):
    h = soerguimento(h, mask, U0, dt)
    h = erosao_fluvial(h, K * (1 - 0.7 * mask), dt)
    h = difusao(h, D, dt)
    h = erosao_termica(h, talus, taxa_termica)
    h = intemperismo(h, t, w_intemp)
    h = garantir_rugosidade(h, rng, rug_min, persistencia, mask)   # sempre por último
    return h

def para_imagem(h, lo, hi):
    return Image.fromarray(np.clip((h - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8), mode="L")

def simular(arquivo="terreno_erosao.gif", png_final="terreno_final.png",
            frames=60, passos_por_frame=3, fps=15, seed=42, **params):
    rng = np.random.default_rng(seed)
    h = gerar_base(seed=seed)
    mask = mascara_afloramento(h.shape, seed)
    quadros = [h.copy()]
    for _ in range(frames - 1):
        for _ in range(passos_por_frame):
            h = passo(h, mask, rng, **params)
        quadros.append(h.copy())

    lo = min(q.min() for q in quadros)
    hi = max(q.max() for q in quadros)
    imgs = [para_imagem(q, lo, hi) for q in quadros]
    imgs[0].save(arquivo, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    imgs[-1].save(png_final)

    print("rugosidade inicial (fina->grossa):", [round(x, 4) for x in rugosidade(quadros[0])])
    print("rugosidade final   (fina->grossa):", [round(x, 4) for x in rugosidade(quadros[-1])])

simular()
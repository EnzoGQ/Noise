import time
import tkinter as tk
from PIL import Image, ImageTk
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
    ordem = np.argsort(h.ravel())[::-1].tolist()
    rec_l = rec.tolist()
    A = [1.0] * h.size
    for i in ordem:
        r = rec_l[i]
        if r != i:
            A[r] += A[i]
    return (np.array(A) / h.size).reshape(h.shape)

def erosao_fluvial(h, K, dt, m=0.5, n=1.0):
    """Retorna o terreno erodido e a área de drenagem A (usada para desenhar os rios)."""
    rec, S = receptores(h)
    A = area_drenagem(h, rec)
    ero = dt * K * (A ** m) * (S ** n)
    h_rec = h.ravel()[rec].reshape(h.shape)
    ero = np.minimum(ero, np.maximum(h - h_rec, 0) * 0.9)
    return h - ero, A

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

def preencher_depressoes(h, passes=8, eps=1e-4):
    for _ in range(passes):
        viz_min = np.min(np.stack([np.roll(h, (-dy, -dx), axis=(0, 1))
                                   for dy, dx, _ in VIZ]), axis=0)
        h = np.where(h < viz_min, viz_min + eps, h)
    return h

# ============================================================
# 4) RUGOSIDADE MÍNIMA (detalhe coerente no tempo)
# ============================================================
RAIOS_BANDA = (0, 1, 2, 4)

def bandas(p):
    suaves = [box_blur(p, r) for r in RAIOS_BANDA]
    return [suaves[i] - suaves[i + 1] for i in range(len(RAIOS_BANDA) - 1)]

def rugosidade(h):
    return [b.std() for b in bandas(h)]

class Detalhe:
    def __init__(self, shape, seed):
        rng = np.random.default_rng(seed + 13)
        def par():
            return [b / b.std() for b in bandas(rng.standard_normal(shape))]
        self.A, self.B = par(), par()

    def campo(self, fase):
        return [np.cos(fase) * a + np.sin(fase) * b for a, b in zip(self.A, self.B)]

def garantir_rugosidade(h, alvo_bandas, minimo=0.006, persistencia=1.6, mask=None):
    r = rugosidade(h)
    peso = 1.0 if mask is None else (0.6 + 0.8 * mask)
    for i, (r_i, b) in enumerate(zip(r, alvo_bandas)):
        piso = minimo * persistencia ** i
        falta = np.sqrt(max(piso ** 2 - r_i ** 2, 0.0))
        if falta > 0:
            h = h + peso * falta * b / (b.std() + 1e-12)
    return h

# ============================================================
# 5) TRANSFORMAÇÃO CONSTANTE
# ============================================================
class CampoSuave:
    """Campo suave (desvio 1) que evolui sem parar: cos(fase)*A + sin(fase)*B."""
    def __init__(self, shape, seed, raio=8):
        rng = np.random.default_rng(seed)
        def suave():
            m = rng.standard_normal(shape)
            for _ in range(3):
                m = box_blur(m, raio)
            return m / m.std()
        self.A, self.B = suave(), suave()

    def campo(self, fase, deriva=0.0):
        f = np.cos(fase) * self.A + np.sin(fase) * self.B
        if deriva:
            f = np.roll(f, (int(deriva * fase * 6), int(deriva * fase * 10)), axis=(0, 1))
        return f

def clima(n, periodo=150.0, amplitude=0.6):
    return 1.0 + amplitude * np.sin(2 * np.pi * n / periodo)

def regular_relevo(h, media0, std0, forca=0.1):
    """Leva o relevo para o desvio-padrão alvo (std0 * amplitude de altura)."""
    s = h.std()
    h = media0 + (h - h.mean()) * (std0 / max(s, 1e-9)) ** forca
    return h + forca * (media0 - h.mean())

# ============================================================
# 6) SIMULAÇÃO
# ============================================================
def passo(h, f, detalhe, clima_k=1.0, dt=1.0, U0=0.002, K=0.05, D=0.05,
          talus=0.02, taxa_termica=0.5, t=50.0, w_intemp=0.02,
          rug_min=0.006, persistencia=1.6):
    m = np.clip(f, 0, 1)
    h = h + dt * U0 * f
    h, A = erosao_fluvial(h, K * clima_k * (1 - 0.7 * m), dt)
    h = difusao(h, D, dt)
    h = erosao_termica(h, talus, taxa_termica)
    h = intemperismo(h, t, w_intemp)
    h = garantir_rugosidade(h, detalhe, rug_min, persistencia, m)
    h = preencher_depressoes(h)
    return h, A

class Mundo:
    VEL_DETALHE = 0.03
    VEL_BIOMA = 0.004

    def __init__(self, seed, aquecimento=15, **params):
        self.seed, self.params = seed, params
        self.h = gerar_base(seed=seed)
        shape = self.h.shape
        self.tect = CampoSuave(shape, seed + 7)
        self.detalhe = Detalhe(shape, seed)
        self.c_temp = CampoSuave(shape, seed + 21, raio=10)     # variação de temperatura
        self.c_umid = CampoSuave(shape, seed + 33, raio=10)     # variação de umidade
        self.fase, self.fase_d, self.fase_b, self.n = 0.0, 0.0, 0.0, 0
        self.A = np.ones(shape) / self.h.size
        self.media0, self.std0 = self.h.mean(), self.h.std()
        self.h = garantir_rugosidade(self.h, self.detalhe.campo(0.0),
                                     params["rug_min"], params["persistencia"])
        self.h = preencher_depressoes(self.h)
        for _ in range(aquecimento):
            self.avancar(0.01, 0.0, 0.6)

    def avancar(self, vel_tect, deriva, amp_clima,
                amp_altura=1.0, amp_agua=0.35, amp_bioma=0.8, **params):
        self.params.update(params)
        self.fase += vel_tect
        self.fase_d += self.VEL_DETALHE
        self.fase_b += self.VEL_BIOMA
        self.n += 1
        f = np.tanh(self.tect.campo(self.fase, deriva) / 1.2)
        p = dict(self.params)
        p["K"] = p["K"] * (0.5 + amp_agua)          # mais água = rios que escavam mais
        self.h, self.A = passo(self.h, f, self.detalhe.campo(self.fase_d),
                               clima(self.n, amplitude=amp_clima), **p)
        self.h = regular_relevo(self.h, self.media0, self.std0 * max(amp_altura, 0.1))

    @property
    def lo(self):          # escala de cor fixa: a amplitude de altura aparece de verdade
        return self.media0 - 2.2 * self.std0

    @property
    def hi(self):
        return self.media0 + 2.2 * self.std0

# ============================================================
# 7) ÁGUA E BIOMAS
# ============================================================
def agua_mapas(h, A, amp, lo):
    """
    amp (0..1) controla toda a água:
      mar   : nível do mar = percentil 40*amp do terreno
      rios  : células com maior área de drenagem (top 6%*amp)
      lagos : concavidades locais (top 5%*amp) em terreno baixo
    """
    vazio = np.zeros(h.shape, bool)
    if amp <= 0.01:
        return dict(mar=vazio, prof=np.zeros(h.shape), rio=vazio, lago=vazio, nivel=lo)
    nivel = np.percentile(h, 40 * amp)
    mar = h < nivel
    prof = np.clip((nivel - h) / max(nivel - h.min(), 1e-9), 0, 1) * mar
    rio = (A >= np.quantile(A, 1 - 0.06 * amp)) & ~mar
    cav = box_blur(h, 3) - h
    lago = ((cav >= np.quantile(cav, 1 - 0.05 * amp)) & (cav > 0)
            & (h < np.percentile(h, 60)) & ~mar)
    return dict(mar=mar, prof=prof, rio=rio, lago=lago, nivel=nivel)

NOMES_BIOMA = ["Gelo/neve", "Rocha", "Prado alpino", "Tundra", "Deserto frio",
               "Estepe fria", "Taiga", "Floresta temperada", "Pradaria", "Semiárido",
               "Deserto rochoso", "Savana", "Floresta tropical", "Deserto de dunas",
               "Pântano", "Praia"]
PALETA_BIOMA = np.array([
    (240, 245, 250),   # 0  Gelo/neve
    (120, 115, 110),   # 1  Rocha
    (140, 165, 120),   # 2  Prado alpino
    (160, 170, 150),   # 3  Tundra
    (190, 195, 200),   # 4  Deserto frio
    (170, 170, 110),   # 5  Estepe fria
    (50, 100, 80),     # 6  Taiga
    (60, 130, 60),     # 7  Floresta temperada
    (130, 180, 80),    # 8  Pradaria
    (185, 160, 95),    # 9  Semiárido
    (170, 140, 110),   # 10 Deserto rochoso
    (190, 170, 70),    # 11 Savana
    (20, 110, 50),     # 12 Floresta tropical
    (235, 205, 120),   # 13 Deserto de dunas
    (70, 100, 70),     # 14 Pântano
    (238, 225, 180),   # 15 Praia
], float)

def biomas(h, agua, mundo, amp_bio):
    """
    Bioma = f(temperatura, umidade, altitude), diagrama tipo Whittaker.
      temperatura : gradiente de latitude + campo que evolui − altitude
      umidade     : campo que evolui + proximidade de água
    amp_bio escala o desvio de T e U em torno de 0.5:
      0 = bioma único, 1 = variação natural, >1 = extremos.
    """
    H, W = h.shape
    y = np.arange(H)[:, None] / H
    t_lat = 0.5 + 0.5 * np.cos(2 * np.pi * (y - 0.5)) * np.ones((1, W))
    alt = np.clip((h - agua["nivel"]) / max(mundo.hi - agua["nivel"], 1e-9), 0, 1)

    ct = mundo.c_temp.campo(mundo.fase_b)
    cu = mundo.c_umid.campo(mundo.fase_b)
    agua_f = (agua["mar"] | agua["lago"] | agua["rio"]).astype(float)
    prox = np.clip(3.0 * box_blur(agua_f, 5), 0, 1)

    T = 0.65 * t_lat + 0.35 * (0.5 + 0.25 * ct) - 0.55 * alt
    U = 0.40 + 0.22 * cu + 0.60 * prox
    T = np.clip(0.5 + amp_bio * (T - 0.5), 0, 1)
    U = np.clip(0.5 + amp_bio * (U - 0.5), 0, 1)

    costa = (box_blur(agua["mar"].astype(float), 2) > 0.04) & ~agua["mar"] & (alt < 0.08)

    # a ordem importa: vale a primeira condição verdadeira
    regras = [
        ((T < 0.18) | (alt > 0.88),           0),   # Gelo/neve
        (alt > 0.75,                          1),   # Rocha
        (alt > 0.60,                          2),   # Prado alpino
        (costa & (T >= 0.25),                 15),  # Praia
        ((U >= 0.75) & (alt < 0.15) & (T >= 0.40), 14),  # Pântano
        ((T < 0.30) & (U >= 0.22),            3),   # Tundra
        ((T < 0.45) & (U < 0.22),             4),   # Deserto frio
        ((T < 0.45) & (U >= 0.45),            6),   # Taiga
        (T < 0.45,                            5),   # Estepe fria
        ((T < 0.70) & (U >= 0.55),            7),   # Floresta temperada
        ((T < 0.70) & (U >= 0.30),            8),   # Pradaria
        ((T < 0.70) & (U >= 0.22),            9),   # Semiárido
        (T < 0.70,                            10),  # Deserto rochoso
        (U >= 0.65,                           12),  # Floresta tropical
        (U >= 0.35,                           11),  # Savana
        (U >= 0.25,                           9),   # Semiárido
    ]
    idx = np.select([c for c, _ in regras], [i for _, i in regras], default=13)  # Dunas
    return idx, PALETA_BIOMA[idx]

# ============================================================
# 8) VISUALIZAÇÃO
# ============================================================
CORES = np.array([
    (0.00, 38, 70, 120), (0.12, 60, 120, 70), (0.35, 120, 160, 80),
    (0.55, 150, 120, 80), (0.78, 130, 125, 120), (1.00, 250, 250, 250)])

COR_MAR_RASO, COR_MAR_FUNDO = np.array([70, 140, 190.]), np.array([15, 40, 110.])
COR_LAGO, COR_RIO = np.array([50, 115, 180.]), np.array([90, 160, 220.])

def colorir(hn):
    return np.stack([np.interp(hn, CORES[:, 0], CORES[:, c]) for c in (1, 2, 3)], axis=-1)

def renderizar(h, lo, hi, modo, sombreamento=True, agua=None, bioma_rgb=None):
    hn = np.clip((h - lo) / max(hi - lo, 1e-9), 0, 1)
    if modo == "cinza":
        rgb = np.repeat((hn * 255)[..., None], 3, axis=2)
    elif modo == "bioma" and bioma_rgb is not None:
        rgb = bioma_rgb.copy()
    else:
        rgb = colorir(hn)

    if sombreamento:
        gy, gx = np.gradient(hn)
        sombra = np.clip(0.9 + 20.0 * (gx + gy), 0.65, 1.25)
        rgb = rgb * sombra[..., None]

    if agua is not None and modo != "cinza":
        prof = agua["prof"][..., None]
        mar = COR_MAR_RASO * (1 - prof) + COR_MAR_FUNDO * prof
        rgb = np.where(agua["mar"][..., None], mar, rgb)
        rgb = np.where(agua["lago"][..., None], COR_LAGO, rgb)
        rgb = np.where(agua["rio"][..., None], COR_RIO, rgb)
    return np.clip(rgb, 0, 255).astype(np.uint8)

COR_CEU_TOPO = np.array([80, 130, 200.])
COR_CEU_BASE = np.array([200, 218, 235.])

def render_3d(tex, h, lo, hi, agua, az, dist, alt, exagero,
              saida=(320, 320), fov=np.radians(70), passos=160):
    """
    Renderização 3D por 'voxel space' totalmente vetorizada (sem GPU).
      tex     : imagem RGB 2D já pronta (relevo/bioma/cinza + sombra + água) usada como textura
      az      : ângulo da câmera ao redor do mapa
      dist    : distância da câmera ao centro do mapa (em células)
      alt     : altura da câmera
      exagero : multiplicador da altura do relevo
    O mapa se repete nas bordas (é um toro, como na simulação).
    """
    Hm, Wm = h.shape
    Ws, Hs = saida
    escala_h = max(hi - lo, 1e-9)
    hn = np.clip((h - lo) / escala_h, 0, 1)
    if agua is not None and agua["mar"].any():          # o mar é plano
        nivel_n = np.clip((agua["nivel"] - lo) / escala_h, 0, 1)
        hn = np.where(agua["mar"], nivel_n, hn)
    relevo = 22.0 * exagero
    Z = hn * relevo

    cx, cy = Wm / 2, Hm / 2
    fx, fy = np.cos(az), np.sin(az)                      # direção da câmera
    rx, ry = -fy, fx                                     # direita da câmera
    camx, camy = cx - fx * dist, cy - fy * dist
    dmax = dist + 0.9 * max(Wm, Hm)

    d = np.geomspace(2.0, dmax, passos)[:, None]         # profundidades (passo cresce com a distância)
    f = np.linspace(-1, 1, Ws)[None, :]                  # coluna da tela
    meia = np.tan(fov / 2)
    ix = np.floor(camx + fx * d + rx * f * d * meia).astype(int) % Wm
    iy = np.floor(camy + fy * d + ry * f * d * meia).astype(int) % Hm

    foco = Ws / 2 / meia
    horizonte = Hs / 2 - (alt - 0.4 * relevo) * foco / dist   # mantém o centro do mapa no meio da tela
    ytop = horizonte + (alt - Z[iy, ix]) * foco / d           # topo do terreno na tela (D, Ws)

    ymin = np.minimum.accumulate(ytop, axis=0)           # silhueta visível até cada profundidade
    linhas = np.arange(Hs)[None, :, None]
    idx = np.count_nonzero(ymin[:, None, :] > linhas, axis=0)   # primeira profundidade que cobre o pixel
    ceu = idx >= passos
    idx = np.minimum(idx, passos - 1)

    nevoa = np.clip((d / dmax) ** 2, 0, 1)[..., None]
    amostra = tex[iy, ix].astype(float)
    amostra = amostra * (1 - nevoa) + COR_CEU_BASE * nevoa
    img = amostra[idx, np.arange(Ws)[None, :]]

    t = np.linspace(0, 1, Hs)[:, None, None]
    img = np.where(ceu[..., None], COR_CEU_TOPO * (1 - t) + COR_CEU_BASE * t, img)
    return np.clip(img, 0, 255).astype(np.uint8)

# ============================================================
# 9) JANELA EM TEMPO REAL
# ============================================================
class App:
    def __init__(self, root, seed=42, escala=5):
        self.root, self.escala = root, escala
        root.title("Terreno em transformação constante")

        self.params = dict(U0=0.002, K=0.05, D=0.05, talus=0.02,
                           rug_min=0.006, persistencia=1.6)
        self.ctrl = dict(vel_tect=0.01, deriva=0.02, amp_clima=0.6,
                         amp_altura=1.0, amp_agua=0.35, amp_bioma=0.8)
        self.passos_por_frame = tk.IntVar(value=1)
        self.modo = tk.StringVar(value="relevo")
        self.mostrar_sombra = tk.BooleanVar(value=True)
        self.mostrar_agua = tk.BooleanVar(value=True)
        self.modo_3d = tk.BooleanVar(value=False)
        self.girar = tk.BooleanVar(value=False)
        self.cam = dict(az=0.6, dist=110.0, alt=85.0)
        self.vista = dict(exagero=1.0)
        self.pausado = False
        self.t_ant, self.fps = time.time(), 0.0

        self.label = tk.Label(root)
        self.label.pack(side=tk.LEFT)

        # Coluna da direita: processos
        painel2 = tk.Frame(root)
        painel2.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        proc = tk.LabelFrame(painel2, text="Processos")
        proc.pack(fill=tk.X)
        self._slider(proc, "Soerguimento / subsidência U0", self.params, "U0", 0, 0.01, 0.0005)
        self._slider(proc, "Erosão fluvial K", self.params, "K", 0, 0.2, 0.005)
        self._slider(proc, "Difusão D", self.params, "D", 0, 0.2, 0.005)
        self._slider(proc, "Talus (térmica)", self.params, "talus", 0.005, 0.1, 0.005)
        self._slider(proc, "Rugosidade mínima", self.params, "rug_min", 0, 0.02, 0.001)
        self._slider(proc, "Persistência", self.params, "persistencia", 1.0, 3.0, 0.1)
        self._slider(proc, "Velocidade tectônica", self.ctrl, "vel_tect", 0, 0.05, 0.002)
        self._slider(proc, "Deriva das placas", self.ctrl, "deriva", 0, 0.2, 0.01)
        self._slider(proc, "Amplitude do clima", self.ctrl, "amp_clima", 0, 0.95, 0.05)

        # Coluna do meio: info, exibição e amplitudes
        painel = tk.Frame(root)
        painel.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)
        self.info = tk.Label(painel, text="", justify=tk.LEFT, font=("Consolas", 9))
        self.info.pack(anchor="w", pady=(0, 6))

        botoes = tk.Frame(painel)
        botoes.pack(pady=(0, 6))
        self.btn_pausa = tk.Button(botoes, text="Pausar", width=9, command=self.alternar)
        self.btn_pausa.grid(row=0, column=0, padx=2)
        tk.Button(botoes, text="Reiniciar", width=9, command=self.reiniciar).grid(row=0, column=1, padx=2)
        tk.Button(botoes, text="Salvar PNG", width=9, command=self.salvar).grid(row=0, column=2, padx=2)

        exib = tk.LabelFrame(painel, text="Exibição")
        exib.pack(fill=tk.X, pady=(0, 6))
        tk.Radiobutton(exib, text="Relevo colorido", variable=self.modo, value="relevo").pack(anchor="w")
        tk.Radiobutton(exib, text="Biomas", variable=self.modo, value="bioma").pack(anchor="w")
        tk.Radiobutton(exib, text="Escala de cinza", variable=self.modo, value="cinza").pack(anchor="w")
        tk.Checkbutton(exib, text="Água  [A]", variable=self.mostrar_agua).pack(anchor="w")
        tk.Checkbutton(exib, text="Sombreamento  [S]", variable=self.mostrar_sombra).pack(anchor="w")
        tk.Checkbutton(exib, text="Visão 3D  [3]", variable=self.modo_3d).pack(anchor="w")
        tk.Checkbutton(exib, text="Girar câmera automaticamente", variable=self.girar).pack(anchor="w")
        self._slider(exib, "Exagero vertical (3D)", self.vista, "exagero", 0.2, 3.0, 0.1)

        amps = tk.LabelFrame(painel, text="Amplitudes")
        amps.pack(fill=tk.X, pady=(0, 6))
        self._slider(amps, "Altura do terreno", self.ctrl, "amp_altura", 0.2, 2.0, 0.05)
        self._slider(amps, "Água (rios, lagos, mares)", self.ctrl, "amp_agua", 0, 1.0, 0.05)
        self._slider(amps, "Bioma", self.ctrl, "amp_bioma", 0, 1.5, 0.05)

        tk.Label(painel, text="Passos por frame").pack(anchor="w")
        tk.Scale(painel, from_=1, to=10, orient=tk.HORIZONTAL, length=220,
                 variable=self.passos_por_frame).pack()

        root.bind("<space>", lambda e: self.alternar())
        root.bind("r", lambda e: self.reiniciar())
        root.bind("s", lambda e: self.mostrar_sombra.set(not self.mostrar_sombra.get()))
        root.bind("a", lambda e: self.mostrar_agua.set(not self.mostrar_agua.get()))
        root.bind("3", lambda e: self.modo_3d.set(not self.modo_3d.get()))
        # câmera 3D: arrastar = girar / subir e descer, roda = aproximar e afastar
        self.label.bind("<ButtonPress-1>", self._arrastar_ini)
        self.label.bind("<B1-Motion>", self._arrastar)
        self.label.bind("<MouseWheel>", self._roda)
        self.label.bind("<Button-4>", self._roda)
        self.label.bind("<Button-5>", self._roda)

        self.mundo = None
        self.reiniciar(seed)
        self.loop()

    def _slider(self, pai, texto, dic, chave, a, b, res):
        tk.Label(pai, text=texto).pack(anchor="w")
        s = tk.Scale(pai, from_=a, to=b, resolution=res, orient=tk.HORIZONTAL, length=220,
                     command=lambda v, d=dic, c=chave: d.__setitem__(c, float(v)))
        s.set(dic[chave])
        s.pack()

    def reiniciar(self, seed=None):
        seed = seed if seed is not None else int(np.random.randint(0, 10**6))
        self.mundo = Mundo(seed, **self.params)

    def alternar(self):
        self.pausado = not self.pausado
        self.btn_pausa.config(text="Continuar" if self.pausado else "Pausar")

    def atualizar_camadas(self):
        m = self.mundo
        self.agua = agua_mapas(m.h, m.A, self.ctrl["amp_agua"], m.lo)
        self.bioma_idx, bioma_rgb = biomas(m.h, self.agua, m, self.ctrl["amp_bioma"])
        self.bioma_rgb = bioma_rgb

    def _arrastar_ini(self, e):
        self._ult = (e.x, e.y)

    def _arrastar(self, e):
        dx, dy = e.x - self._ult[0], e.y - self._ult[1]
        self._ult = (e.x, e.y)
        self.cam["az"] += dx * 0.01
        self.cam["alt"] = float(np.clip(self.cam["alt"] + dy * 0.5, 10, 200))

    def _roda(self, e):
        if getattr(e, "num", 0) == 4:
            sentido = 1
        elif getattr(e, "num", 0) == 5:
            sentido = -1
        else:
            sentido = 1 if e.delta > 0 else -1
        fator = 0.93 if sentido > 0 else 1 / 0.93
        self.cam["dist"] = float(np.clip(self.cam["dist"] * fator, 30, 220))

    def imagem(self):
        m = self.mundo
        agua = self.agua if self.mostrar_agua.get() else None
        rgb = renderizar(m.h, m.lo, m.hi, self.modo.get(), self.mostrar_sombra.get(),
                         agua, self.bioma_rgb)
        tam = (m.h.shape[1] * self.escala, m.h.shape[0] * self.escala)
        if self.modo_3d.get():
            rgb3 = render_3d(rgb, m.h, m.lo, m.hi, agua, self.cam["az"],
                             self.cam["dist"], self.cam["alt"], self.vista["exagero"])
            return Image.fromarray(rgb3).resize(tam, Image.BILINEAR)
        return Image.fromarray(rgb).resize(tam, Image.NEAREST)

    def salvar(self):
        self.imagem().save(f"terreno_{self.mundo.seed}_{self.mundo.n}.png")

    def loop(self):
        m = self.mundo
        if not self.pausado:
            for _ in range(self.passos_por_frame.get()):
                m.avancar(**self.ctrl, **self.params)

        if self.girar.get():
            self.cam["az"] += 0.01

        self.atualizar_camadas()
        self.foto = ImageTk.PhotoImage(self.imagem())
        self.label.config(image=self.foto)

        agora = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(agora - self.t_ant, 1e-6)
        self.t_ant = agora
        r = rugosidade(m.h)
        total = m.h.size
        dom = NOMES_BIOMA[int(np.bincount(self.bioma_idx.ravel(), minlength=len(NOMES_BIOMA)).argmax())]
        self.info.config(text=(
            f"seed {m.seed}   passo {m.n}   {self.fps:4.1f} fps\n"
            f"mar {100 * self.agua['mar'].sum() / total:4.1f}%  "
            f"rios {100 * self.agua['rio'].sum() / total:4.1f}%  "
            f"lagos {100 * self.agua['lago'].sum() / total:4.1f}%\n"
            f"bioma dominante: {dom}\n"
            f"altura {m.h.min():.2f} .. {m.h.max():.2f}\n"
            f"rugosidade fina {r[0]:.4f}  média {r[1]:.4f}  grossa {r[2]:.4f}"))
        self.root.after(1, self.loop)


if __name__ == "__main__":
    raiz = tk.Tk()
    App(raiz)
    raiz.mainloop()
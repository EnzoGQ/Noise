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

def preencher_depressoes(h, passes=8, eps=1e-4):
    """Sedimentação: células mais baixas que todos os vizinhos sobem, mantendo a drenagem."""
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
    """Campo fractal de detalhe que evolui devagar: banda_i = cos(fase)*A_i + sin(fase)*B_i."""
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
class Tectonica:
    """f(fase) = cos(fase)*A + sin(fase)*B: soerguimento (f>0) e subsidência (f<0) que migram."""
    def __init__(self, shape, seed, raio=8):
        rng = np.random.default_rng(seed + 7)
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
        return np.tanh(f / 1.2)

def clima(n, periodo=150.0, amplitude=0.6):
    return 1.0 + amplitude * np.sin(2 * np.pi * n / periodo)

def regular_relevo(h, media0, std0, forca=0.05):
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
    h = erosao_fluvial(h, K * clima_k * (1 - 0.7 * m), dt)
    h = difusao(h, D, dt)
    h = erosao_termica(h, talus, taxa_termica)
    h = intemperismo(h, t, w_intemp)
    h = garantir_rugosidade(h, detalhe, rug_min, persistencia, m)
    h = preencher_depressoes(h)
    return h

class Mundo:
    VEL_DETALHE = 0.03

    def __init__(self, seed, aquecimento=15, **params):
        self.seed, self.params = seed, params
        self.h = gerar_base(seed=seed)
        self.tect = Tectonica(self.h.shape, seed)
        self.detalhe = Detalhe(self.h.shape, seed)
        self.fase, self.fase_d, self.n = 0.0, 0.0, 0
        self.media0, self.std0 = self.h.mean(), self.h.std()
        self.h = garantir_rugosidade(self.h, self.detalhe.campo(0.0),
                                     params["rug_min"], params["persistencia"])
        self.h = preencher_depressoes(self.h)
        for _ in range(aquecimento):
            self.avancar(0.01, 0.0, 0.6)

    def avancar(self, vel_tect, deriva, amp_clima, **params):
        self.params.update(params)
        self.fase += vel_tect
        self.fase_d += self.VEL_DETALHE
        self.n += 1
        self.f = self.tect.campo(self.fase, deriva)
        self.h = passo(self.h, self.f, self.detalhe.campo(self.fase_d),
                       clima(self.n, amplitude=amp_clima), **self.params)
        self.h = regular_relevo(self.h, self.media0, self.std0)

# ============================================================
# 7) VISUALIZAÇÃO
# ============================================================
CORES = np.array([
    (0.00, 38, 70, 120), (0.12, 60, 120, 70), (0.35, 120, 160, 80),
    (0.55, 150, 120, 80), (0.78, 130, 125, 120), (1.00, 250, 250, 250)])

def colorir(hn):
    return np.stack([np.interp(hn, CORES[:, 0], CORES[:, c]) for c in (1, 2, 3)], axis=-1)

def renderizar(h, lo, hi, modo, sombreamento=True):
    """sombreamento=False remove as manchas escuras (rugosidade/sombra), deixando só a cor por altura."""
    hn = np.clip((h - lo) / max(hi - lo, 1e-9), 0, 1)
    if modo == "cinza":
        return np.repeat((hn * 255).astype(np.uint8)[..., None], 3, axis=2)
    rgb = colorir(hn)
    if sombreamento:
        gy, gx = np.gradient(hn)
        sombra = np.clip(0.9 + 20.0 * (gx + gy), 0.65, 1.25)   # luz de noroeste, piso 0.65
        rgb = rgb * sombra[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)

# ============================================================
# 8) JANELA EM TEMPO REAL
# ============================================================
class App:
    def __init__(self, root, seed=42, escala=5):
        self.root, self.escala = root, escala
        root.title("Terreno em transformação constante")

        self.params = dict(U0=0.002, K=0.05, D=0.05, talus=0.02,
                           rug_min=0.006, persistencia=1.6)
        self.ctrl = dict(vel_tect=0.01, deriva=0.02, amp_clima=0.6)
        self.passos_por_frame = tk.IntVar(value=1)
        self.modo = tk.StringVar(value="relevo")
        self.mostrar_rug = tk.BooleanVar(value=True)     # liga/desliga as manchas escuras
        self.pausado = False
        self.t_ant, self.fps = time.time(), 0.0

        self.label = tk.Label(root)
        self.label.pack(side=tk.LEFT)

        painel = tk.Frame(root)
        painel.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)
        self.info = tk.Label(painel, text="", justify=tk.LEFT, font=("Consolas", 9))
        self.info.pack(anchor="w", pady=(0, 6))

        self._slider(painel, "Soerguimento / subsidência U0", self.params, "U0", 0, 0.01, 0.0005)
        self._slider(painel, "Erosão fluvial K", self.params, "K", 0, 0.2, 0.005)
        self._slider(painel, "Difusão D", self.params, "D", 0, 0.2, 0.005)
        self._slider(painel, "Talus (térmica)", self.params, "talus", 0.005, 0.1, 0.005)
        self._slider(painel, "Rugosidade mínima", self.params, "rug_min", 0, 0.02, 0.001)
        self._slider(painel, "Persistência", self.params, "persistencia", 1.0, 3.0, 0.1)
        self._slider(painel, "Velocidade tectônica", self.ctrl, "vel_tect", 0, 0.05, 0.002)
        self._slider(painel, "Deriva das placas", self.ctrl, "deriva", 0, 0.2, 0.01)
        self._slider(painel, "Amplitude do clima", self.ctrl, "amp_clima", 0, 0.95, 0.05)

        tk.Label(painel, text="Passos por frame").pack(anchor="w")
        tk.Scale(painel, from_=1, to=10, orient=tk.HORIZONTAL, length=220,
                 variable=self.passos_por_frame).pack()
        tk.Radiobutton(painel, text="Relevo colorido", variable=self.modo, value="relevo").pack(anchor="w")
        tk.Radiobutton(painel, text="Escala de cinza", variable=self.modo, value="cinza").pack(anchor="w")
        tk.Checkbutton(painel, text="Rugosidade / sombreamento  [S]",
                       variable=self.mostrar_rug).pack(anchor="w", pady=(4, 0))

        botoes = tk.Frame(painel)
        botoes.pack(pady=8)
        self.btn_pausa = tk.Button(botoes, text="Pausar", width=9, command=self.alternar)
        self.btn_pausa.grid(row=0, column=0, padx=2)
        tk.Button(botoes, text="Reiniciar", width=9, command=self.reiniciar).grid(row=0, column=1, padx=2)
        tk.Button(botoes, text="Salvar PNG", width=9, command=self.salvar).grid(row=0, column=2, padx=2)
        root.bind("<space>", lambda e: self.alternar())
        root.bind("r", lambda e: self.reiniciar())
        root.bind("s", lambda e: self.mostrar_rug.set(not self.mostrar_rug.get()))

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
        self.lo, self.hi = np.percentile(self.mundo.h, (1, 99))

    def alternar(self):
        self.pausado = not self.pausado
        self.btn_pausa.config(text="Continuar" if self.pausado else "Pausar")

    def imagem(self):
        h = self.mundo.h
        rgb = renderizar(h, self.lo, self.hi, self.modo.get(), self.mostrar_rug.get())
        return Image.fromarray(rgb).resize(
            (h.shape[1] * self.escala, h.shape[0] * self.escala), Image.NEAREST)

    def salvar(self):
        self.imagem().save(f"terreno_{self.mundo.seed}_{self.mundo.n}.png")

    def loop(self):
        m = self.mundo
        if not self.pausado:
            for _ in range(self.passos_por_frame.get()):
                m.avancar(**self.ctrl, **self.params)

        p1, p99 = np.percentile(m.h, (1, 99))
        self.lo += 0.1 * (p1 - self.lo)
        self.hi += 0.1 * (p99 - self.hi)
        self.foto = ImageTk.PhotoImage(self.imagem())
        self.label.config(image=self.foto)

        agora = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(agora - self.t_ant, 1e-6)
        self.t_ant = agora
        r = rugosidade(m.h)
        self.info.config(text=(f"seed {m.seed}   passo {m.n}   {self.fps:4.1f} fps\n"
                               f"fase tectônica {m.fase % (2 * np.pi):.2f} rad   "
                               f"clima x{clima(m.n, amplitude=self.ctrl['amp_clima']):.2f}\n"
                               f"altura {m.h.min():.2f} .. {m.h.max():.2f}\n"
                               f"rugosidade fina {r[0]:.4f}  média {r[1]:.4f}  grossa {r[2]:.4f}"))
        self.root.after(1, self.loop)


if __name__ == "__main__":
    raiz = tk.Tk()
    App(raiz)
    raiz.mainloop()
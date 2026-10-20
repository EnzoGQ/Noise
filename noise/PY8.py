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

# ============================================================
# 4) RUGOSIDADE MÍNIMA
# ============================================================
RAIOS_BANDA = (0, 1, 2, 4)

def bandas(p):
    suaves = [box_blur(p, r) for r in RAIOS_BANDA]
    return [suaves[i] - suaves[i + 1] for i in range(len(RAIOS_BANDA) - 1)]

def rugosidade(h):
    return [b.std() for b in bandas(h)]

def garantir_rugosidade(h, rng, minimo=0.006, persistencia=1.6, mask=None):
    r = rugosidade(h)
    nb = bandas(rng.standard_normal(h.shape))
    peso = 1.0 if mask is None else (0.6 + 0.8 * mask)
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
    h = garantir_rugosidade(h, rng, rug_min, persistencia, mask)
    return h

def terreno_inicial(rng, seed, rug_min, persistencia, aquecimento, **params):
    h = gerar_base(seed=seed)
    mask = mascara_afloramento(h.shape, seed)
    h = garantir_rugosidade(h, rng, rug_min, persistencia, mask)
    for _ in range(aquecimento):
        h = passo(h, mask, rng, rug_min=rug_min, persistencia=persistencia, **params)
    return h, mask

# ============================================================
# 6) VISUALIZAÇÃO
# ============================================================
CORES = np.array([  # altura 0..1 -> RGB
    (0.00, 38, 70, 120), (0.12, 60, 120, 70), (0.35, 120, 160, 80),
    (0.55, 150, 120, 80), (0.78, 130, 125, 120), (1.00, 250, 250, 250)])

def colorir(hn):
    return np.stack([np.interp(hn, CORES[:, 0], CORES[:, c]) for c in (1, 2, 3)], axis=-1)

def renderizar(h, lo, hi, modo):
    hn = np.clip((h - lo) / max(hi - lo, 1e-9), 0, 1)
    if modo == "cinza":
        return np.repeat((hn * 255).astype(np.uint8)[..., None], 3, axis=2)
    gy, gx = np.gradient(hn)
    sombra = np.clip(0.75 + 40.0 * (gx + gy), 0.35, 1.25)       # luz vinda de noroeste
    rgb = colorir(hn) * sombra[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)

# ============================================================
# 7) JANELA EM TEMPO REAL
# ============================================================
class App:
    def __init__(self, root, seed=42, escala=5):
        self.root, self.seed, self.escala = root, seed, escala
        root.title("Terreno: erosão em tempo real")

        self.params = dict(U0=0.003, K=0.05, D=0.05, talus=0.02,
                           rug_min=0.006, persistencia=1.6)
        self.passos_por_frame = tk.IntVar(value=1)
        self.modo = tk.StringVar(value="relevo")
        self.pausado = False
        self.n_passos = 0
        self.t_ant = time.time()
        self.fps = 0.0

        self.label = tk.Label(root)
        self.label.pack(side=tk.LEFT)

        painel = tk.Frame(root)
        painel.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        self.info = tk.Label(painel, text="", justify=tk.LEFT, font=("Consolas", 9))
        self.info.pack(anchor="w", pady=(0, 6))

        self._slider(painel, "Soerguimento U0", "U0", 0, 0.01, 0.0005)
        self._slider(painel, "Erosão fluvial K", "K", 0, 0.2, 0.005)
        self._slider(painel, "Difusão D", "D", 0, 0.2, 0.005)
        self._slider(painel, "Talus (térmica)", "talus", 0.005, 0.1, 0.005)
        self._slider(painel, "Rugosidade mínima", "rug_min", 0, 0.02, 0.001)
        self._slider(painel, "Persistência", "persistencia", 1.0, 3.0, 0.1)

        tk.Label(painel, text="Passos por frame").pack(anchor="w")
        tk.Scale(painel, from_=1, to=10, orient=tk.HORIZONTAL, length=220,
                 variable=self.passos_por_frame).pack()

        tk.Radiobutton(painel, text="Relevo colorido", variable=self.modo,
                       value="relevo").pack(anchor="w")
        tk.Radiobutton(painel, text="Escala de cinza", variable=self.modo,
                       value="cinza").pack(anchor="w")

        botoes = tk.Frame(painel)
        botoes.pack(pady=8)
        self.btn_pausa = tk.Button(botoes, text="Pausar", width=9, command=self.alternar)
        self.btn_pausa.grid(row=0, column=0, padx=2)
        tk.Button(botoes, text="Reiniciar", width=9, command=self.reiniciar).grid(row=0, column=1, padx=2)
        tk.Button(botoes, text="Salvar PNG", width=9, command=self.salvar).grid(row=0, column=2, padx=2)

        root.bind("<space>", lambda e: self.alternar())
        root.bind("r", lambda e: self.reiniciar())

        self.reiniciar()
        self.loop()

    def _slider(self, pai, texto, chave, a, b, res):
        tk.Label(pai, text=texto).pack(anchor="w")
        s = tk.Scale(pai, from_=a, to=b, resolution=res, orient=tk.HORIZONTAL, length=220,
                     command=lambda v, c=chave: self.params.__setitem__(c, float(v)))
        s.set(self.params[chave])
        s.pack()

    def reiniciar(self):
        self.seed = np.random.randint(0, 10**6) if getattr(self, "h", None) is not None else self.seed
        self.rng = np.random.default_rng(self.seed)
        extra = {k: v for k, v in self.params.items() if k not in ("rug_min", "persistencia")}
        self.h, self.mask = terreno_inicial(self.rng, self.seed, self.params["rug_min"],
                                            self.params["persistencia"], 15, **extra)
        self.lo, self.hi = self.h.min(), self.h.max()
        self.n_passos = 0

    def alternar(self):
        self.pausado = not self.pausado
        self.btn_pausa.config(text="Continuar" if self.pausado else "Pausar")

    def salvar(self):
        Image.fromarray(renderizar(self.h, self.lo, self.hi, self.modo.get())).resize(
            (self.h.shape[1] * self.escala,) * 2 if False else
            (self.h.shape[1] * self.escala, self.h.shape[0] * self.escala), Image.NEAREST
        ).save(f"terreno_{self.seed}_{self.n_passos}.png")

    def loop(self):
        if not self.pausado:
            for _ in range(self.passos_por_frame.get()):
                self.h = passo(self.h, self.mask, self.rng, **self.params)
                self.n_passos += 1

        # escala de cor suavizada (evita piscar) e contínua com o crescimento do relevo
        self.lo += 0.1 * (self.h.min() - self.lo)
        self.hi += 0.1 * (self.h.max() - self.hi)

        rgb = renderizar(self.h, self.lo, self.hi, self.modo.get())
        img = Image.fromarray(rgb).resize(
            (self.h.shape[1] * self.escala, self.h.shape[0] * self.escala), Image.NEAREST)
        self.foto = ImageTk.PhotoImage(img)
        self.label.config(image=self.foto)

        agora = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(agora - self.t_ant, 1e-6)
        self.t_ant = agora
        r = rugosidade(self.h)
        self.info.config(text=(f"seed {self.seed}   passo {self.n_passos}   {self.fps:4.1f} fps\n"
                               f"altura {self.h.min():.2f} .. {self.h.max():.2f}\n"
                               f"rugosidade fina {r[0]:.4f}  média {r[1]:.4f}  grossa {r[2]:.4f}"))
        self.root.after(1, self.loop)


if __name__ == "__main__":
    raiz = tk.Tk()
    App(raiz)
    raiz.mainloop()
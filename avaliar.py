"""Avaliação no conjunto de TESTE: inferência com métricas, melhores/piores casos, teste de esteganografia e ablação."""
import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from analyze import escolher_epocas, fonte, ler_csv, listar_imgs
from models import load_generators, load_image, to_pil

COLS_METRICAS = ["dominio", "arquivo", "cyc_l1", "psnr", "ssim"]
COLS_ESTEGO = ["dominio", "sigma", "cyc_sem_ruido", "cyc_com_ruido", "aumento", "ruido_l1", "razao", "sensibilidade"]
SENTIDO = {"A": "A->B->A", "B": "B->A->B"}


# ---------- métricas (tensores em [-1, 1], formato (N, 3, H, W); devolvem um valor por imagem) ----------

def cyc_l1(rec, real):
    return (rec - real).abs().mean((1, 2, 3))


def psnr(rec, real):
    mse = ((rec - real) ** 2).mean((1, 2, 3)).clamp_min(1e-10)
    return 10 * torch.log10(4.0 / mse)  # amplitude de [-1, 1] = 2


def ssim(rec, real, janela=11, sigma=1.5):
    """SSIM com janela gaussiana, média dos 3 canais; imagens passadas para [0, 1]."""
    n = min(janela, min(rec.shape[-2:]))
    n -= 1 - n % 2  # ímpar
    x = torch.arange(n, device=rec.device, dtype=torch.float32) - n // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    w = (g[:, None] * g[None, :]).expand(3, 1, n, n).contiguous()
    f = lambda t: F.conv2d(t, w, groups=3)  # noqa: E731
    a, b = (rec.float() + 1) / 2, (real.float() + 1) / 2
    ma, mb = f(a), f(b)
    saa, sbb, sab = f(a * a) - ma ** 2, f(b * b) - mb ** 2, f(a * b) - ma * mb
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * ma * mb + c1) * (2 * sab + c2)) / ((ma ** 2 + mb ** 2 + c1) * (saa + sbb + c2))
    return s.mean((1, 2, 3))


# ---------- utilidades ----------

def achar_ckpt(run, ckpt):
    if ckpt:
        p = Path(ckpt)
        if not p.exists() and (run / ckpt).exists():
            p = run / ckpt
        if not p.exists():
            raise SystemExit(f"Checkpoint não encontrado: {ckpt}")
        return p
    cands = sorted(run.glob("G_epoca_*.pt"))
    if not cands:
        raise SystemExit(f"Sem G_epoca_*.pt em {run}: passe --ckpt.")
    return cands[-1]


def preparar(args):
    run = Path(args.runs_dir) / args.run
    if not run.is_dir():
        raise SystemExit(f"Run não encontrada: {run}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = achar_ckpt(run, args.ckpt)
    G_AB, G_BA, ep = load_generators(ck, dev)
    cfg = run / "config.json"
    size = json.loads(cfg.read_text(encoding="utf-8"))["args"]["size"] if cfg.exists() else 256
    print(f"Dispositivo: {dev.type} | checkpoint: {ck} (época {ep}) | tamanho {size}")
    return run, dev, {"A": (G_AB, G_BA), "B": (G_BA, G_AB)}, size


def passo(g, x, fp16):
    with torch.no_grad(), torch.autocast("cuda", enabled=fp16 and x.is_cuda):
        return g(x).float()


def lotes(arqs, size, bs, dev):
    for i in range(0, len(arqs), bs):
        parte = arqs[i:i + bs]
        yield parte, torch.stack([load_image(p, size) for p in parte]).to(dev)


def arquivos_teste(data, dom):
    arqs = listar_imgs(Path(data) / f"test{dom}")
    if not arqs:
        raise SystemExit(f"Sem imagens em {Path(data) / f'test{dom}'}")
    return arqs


def escrever_csv(path, cols, linhas):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows([f"{v:.6g}" if isinstance(v, float) else v for v in l] for l in linhas)


def achar_img(pasta, stem, sufixo):
    for ext in (".jpg", ".png"):
        p = pasta / f"{stem}_{sufixo}{ext}"
        if p.exists():
            return p
    raise SystemExit(f"Falta {pasta / f'{stem}_{sufixo}.*'}: rode `inferir` antes.")


# ---------- inferir ----------

def cmd_inferir(args):
    run, dev, G, size = preparar(args)
    linhas = []
    for dom, (g1, g2) in G.items():
        outro = "B" if dom == "A" else "A"
        pasta = run / "teste" / f"{dom}2{outro}"
        pasta.mkdir(parents=True, exist_ok=True)
        for nomes, x in lotes(arquivos_teste(args.data, dom), size, args.batch, dev):
            fake = passo(g1, x, args.fp16)
            rec = passo(g2, fake, args.fp16)
            c, p, s = cyc_l1(rec, x).tolist(), psnr(rec, x).tolist(), ssim(rec, x).tolist()
            for i, q in enumerate(nomes):
                for t, suf in ((fake[i], "traduzida"), (rec[i], "reconstruida")):
                    to_pil(t).save(pasta / f"{q.stem}_{suf}.{args.formato}", quality=95)
                linhas.append([dom, q.name, c[i], p[i], s[i]])
        m = [np.mean([l[k] for l in linhas if l[0] == dom]) for k in (2, 3, 4)]
        print(f"{SENTIDO[dom]} ({sum(l[0] == dom for l in linhas)} imagens): cíclica L1 {m[0]:.4f} | PSNR {m[1]:.2f} dB | SSIM {m[2]:.4f}")
    escrever_csv(run / "metricas_teste.csv", COLS_METRICAS, linhas)
    print(f"Salvo: {run / 'metricas_teste.csv'} e imagens em {run / 'teste'}")


# ---------- casos ----------

def cmd_casos(args):
    run = Path(args.runs_dir) / args.run
    csv_path = run / "metricas_teste.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} não existe: rode `inferir` antes.")
    with open(csv_path, newline="", encoding="utf-8") as f:
        todas = list(csv.DictReader(f))
    s, marg = args.lado, 3 * args.lado // 4
    topo, f = max(24, s // 6), fonte(max(12, s // 12))
    for dom in "AB":
        outro = "B" if dom == "A" else "A"
        rs = sorted((r for r in todas if r["dominio"] == dom), key=lambda r: float(r["cyc_l1"]))
        if not rs:
            continue
        n = min(args.n, max(1, len(rs) // 2))
        sel = [("melhor", i + 1, r) for i, r in enumerate(rs[:n])] + [("pior", i + 1, r) for i, r in enumerate(reversed(rs[-n:]))]
        pasta = run / "teste" / f"{dom}2{outro}"
        tela = Image.new("RGB", (marg + 3 * s, topo + len(sel) * s), "white")
        d = ImageDraw.Draw(tela)
        for j, tit in enumerate((f"real ({dom})", f"traduzida ({outro})", f"reconstruída ({dom})")):
            d.text((marg + (j + 0.5) * s, topo // 2), tit, fill="black", font=f, anchor="mm")
        for i, (tipo, k, r) in enumerate(sel):
            stem, y = Path(r["arquivo"]).stem, topo + i * s
            d.multiline_text((6, y + s // 2), f"{tipo} {k}\ncíclica L1 {float(r['cyc_l1']):.3f}\n{stem[:16]}",
                             fill="black", font=f, anchor="lm")
            for j, p in enumerate((Path(args.data) / f"test{dom}" / r["arquivo"], achar_img(pasta, stem, "traduzida"),
                                   achar_img(pasta, stem, "reconstruida"))):
                tela.paste(Image.open(p).convert("RGB").resize((s, s)), (marg + j * s, y))
            if i == n:  # separa melhores de piores
                d.line([(0, y), (marg + 3 * s, y)], fill="red", width=3)
        destino = run / f"casos_{dom}2{outro}.jpg"
        tela.save(destino, quality=90)
        print(f"Salvo: {destino} ({n} melhores + {n} piores por cíclica L1, só teste)")


# ---------- esteganografia ----------

def cmd_esteganografia(args):
    run, dev, G, size = preparar(args)
    sigmas = sorted(set(args.sigma))
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    linhas, exemplos = [], {}
    for dom, (g1, g2) in G.items():
        arqs = arquivos_teste(args.data, dom)
        arqs = sorted(random.Random(args.seed).sample(arqs, min(args.k, len(arqs))))
        ac = {s: {"base": [], "com": [], "ruido": [], "dif": []} for s in sigmas}
        for nomes, x in lotes(arqs, size, args.batch, dev):
            fake = passo(g1, x, args.fp16)
            rec0 = passo(g2, fake, args.fp16)
            z = torch.randn(fake.shape, generator=gen, device=dev)
            recs = []
            for s in sigmas:
                ruidosa = (fake + s * z).clamp(-1, 1)
                rec = passo(g2, ruidosa, args.fp16)
                recs.append(rec)
                a = ac[s]
                a["base"] += cyc_l1(rec0, x).tolist()
                a["com"] += cyc_l1(rec, x).tolist()
                a["ruido"] += cyc_l1(ruidosa, fake).tolist()
                a["dif"] += cyc_l1(rec, rec0).tolist()
            if dom not in exemplos:
                exemplos[dom] = [x[0], fake[0], rec0[0]] + [r[0] for r in recs]
        base = np.mean(ac[sigmas[0]]["base"])
        linhas.append([dom, 0.0, base, base, 0.0, 0.0, "", ""])
        for s in sigmas:
            a = {k: float(np.mean(v)) for k, v in ac[s].items()}
            aumento = a["com"] - a["base"]
            linhas.append([dom, s, a["base"], a["com"], aumento, a["ruido"], aumento / a["ruido"], a["dif"] / a["ruido"]])
    escrever_csv(run / "esteganografia.csv", COLS_ESTEGO, linhas)

    print(f"\nTeste de esteganografia (Chu et al., 2017): {args.k} imagens de teste por domínio, ruído gaussiano em fake")
    print(f"{'sentido':<8} {'sigma':>6} {'cyc s/ruído':>12} {'cyc c/ruído':>12} {'aumento':>9} {'|ruído|':>9} {'razão':>7} {'sensib.':>8}")
    for dom, s, base, com, aum, ru, raz, sen in linhas:
        extra = "" if raz == "" else f" {raz:7.2f} {sen:8.2f}" + ("  <- suspeito" if raz > args.limiar else "")
        print(f"{SENTIDO[dom]:<8} {s:6.3f} {base:12.4f} {com:12.4f} {aum:9.4f} {ru:9.4f}{extra}")
    print(f"razão = aumento da cíclica L1 / L1 do ruído; sensib. = L1(rec com ruído, rec sem ruído) / L1 do ruído.\n"
          f"Razão ~1 ou menos: G_BA não amplifica o ruído. Razão >> 1 (limiar {args.limiar}): informação escondida em alta frequência.")

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
    ax = fig.add_subplot(gs[0])
    larg = 0.8 / (len(sigmas) + 1)
    for j, s in enumerate([0.0] + sigmas):
        ys = [next(l[3] for l in linhas if l[0] == dom and l[1] == s) for dom in "AB"]
        ax.bar(np.arange(2) + (j - len(sigmas) / 2) * larg, ys, larg, label="sem ruído" if s == 0 else f"σ = {s:g}")
    ax.set_xticks(range(2), [SENTIDO[d] for d in "AB"], fontsize=12)
    ax.set_ylabel("Cíclica L1", fontsize=13)
    ax.set_title(f"Cíclica L1 com ruído gaussiano na imagem traduzida ({args.k} imagens de teste)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11)
    rot = "real | traduzida | rec. sem ruído | " + " | ".join(f"rec. σ={s:g}" for s in sigmas)
    for i, dom in enumerate("AB"):
        a = fig.add_subplot(gs[1 + i])
        a.imshow(np.concatenate([np.asarray(to_pil(t)) for t in exemplos[dom]], axis=1))
        a.set_title(f"{SENTIDO[dom]}: {rot}", fontsize=9)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(run / "esteganografia.png", dpi=150)
    plt.close(fig)
    print(f"Salvo: {run / 'esteganografia.csv'} e {run / 'esteganografia.png'}")


# ---------- ablação ----------

def serie_epocas(run):
    """{época: {cyc_A, cyc_B, G_adv, D}} a partir de epocas.csv (ou, se faltar, da média de perdas.csv por época)."""
    if (run / "epocas.csv").exists():
        linhas = ler_csv(run / "epocas.csv")
    elif (run / "perdas.csv").exists():
        por = {}
        for l in ler_csv(run / "perdas.csv"):
            por.setdefault(int(l["epoca"]), []).append(l)
        linhas = [{k: float(np.mean([l[k] for l in ls])) for k in ls[0]} for ls in por.values()]
    else:
        raise SystemExit(f"Sem epocas.csv/perdas.csv em {run}")
    return {int(l["epoca"]): {"cyc_A": l["cyc_A"], "cyc_B": l["cyc_B"], "G_adv": (l["G_adv_AB"] + l["G_adv_BA"]) / 2,
                              "D": (l["D_A"] + l["D_B"]) / 2} for l in linhas}


def cmd_ablacao(args):
    pa, pb = Path(args.runs_dir) / args.runA, Path(args.runs_dir) / args.runB
    sa, sb = serie_epocas(pa), serie_epocas(pb)
    comuns = sorted(set(sa) & set(sb))
    if not comuns:
        raise SystemExit("As duas runs não têm épocas em comum.")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (k, tit) in zip(axs.flat, [("cyc_A", "Cíclica A (A→B→A)"), ("cyc_B", "Cíclica B (B→A→B)"),
                                       ("G_adv", "Adversarial do gerador (média AB/BA)"), ("D", "Discriminador (média A/B)")]):
        for nome, s, cor in ((args.runA, sa, "C0"), (args.runB, sb, "C1")):
            ax.plot(comuns, [s[e][k] for e in comuns], color=cor, linewidth=2, label=nome)
        ax.set_title(tit, fontsize=13)
        ax.grid(alpha=0.3)
    for ax in axs[1]:
        ax.set_xlabel("Época", fontsize=12)
    axs[0, 0].legend(fontsize=11)
    fig.suptitle(f"Ablação: {args.runB} vs {args.runA} (épocas 1–{comuns[-1]})", fontsize=15)
    fig.tight_layout()
    destino = pb / f"ablacao_vs_{args.runA}.png"
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    print(f"Salvo: {destino}")

    def com_grade(run):
        return {int(p.stem.split("_")[1]) for p in (run / "amostras").glob("epoca_*.jpg")}

    disp = sorted(com_grade(pa) & com_grade(pb))
    pedidas = [int(e) for e in args.epocas.split(",")] if args.epocas else None
    epocas = escolher_epocas(disp, pedidas, maximo=3)
    if not epocas:
        print("Sem grades de amostras em comum: grade lado a lado não gerada.")
        return
    fig, axs = plt.subplots(1, 2 * len(epocas), figsize=(2.6 * 2 * len(epocas), 7.2), squeeze=False)
    for j, e in enumerate(epocas):
        for k, (nome, run) in enumerate(((args.runA, pa), (args.runB, pb))):
            ax = axs[0, 2 * j + k]
            ax.imshow(Image.open(run / "amostras" / f"epoca_{e:03d}.jpg").convert("RGB"))
            ax.set_title(f"{nome}\nép. {e}", fontsize=10)
            ax.axis("off")
    fig.suptitle("Linhas: real | traduzida | reconstruída (amostras fixas de teste; A em cima, B embaixo)", fontsize=10)
    fig.tight_layout()
    destino = pb / f"ablacao_vs_{args.runA}.jpg"
    fig.savefig(destino, dpi=130, pil_kwargs={"quality": 90})
    plt.close(fig)
    print(f"Salvo: {destino}")


# ---------- main ----------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def comum(p, ckpt=True):
        p.add_argument("--runs-dir", default="runs")
        if ckpt:
            p.add_argument("run")
            p.add_argument("--ckpt", default=None, help="arquivo .pt (ou nome dentro da run); padrão: última G_epoca_XXX.pt")
            p.add_argument("--data", default="data")
            p.add_argument("--batch", type=int, default=8)
            p.add_argument("--fp16", action="store_true", help="inferência em fp16 (só CUDA)")

    p = sub.add_parser("inferir", help="traduz/reconstrói testA e testB e grava métricas por imagem em metricas_teste.csv")
    comum(p)
    p.add_argument("--formato", choices=["jpg", "png"], default="jpg")
    p.set_defaults(fn=cmd_inferir)

    p = sub.add_parser("casos", help="figuras casos_A2B.jpg/casos_B2A.jpg com melhores e piores casos de TESTE",
                       description="Escolhe as N melhores e N piores imagens de teste pela cíclica L1 (menor = melhor), "
                                   "o mesmo critério e o mesmo N nos dois sentidos. Critério objetivo e fixo: evita cherry-picking.")
    p.add_argument("run")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--data", default="data")
    p.add_argument("-n", "--n", type=int, default=4, help="N melhores e N piores por sentido (por cíclica L1)")
    p.add_argument("--lado", type=int, default=256, help="lado de cada célula, em pixels")
    p.set_defaults(fn=cmd_casos)

    p = sub.add_parser("esteganografia", help="teste de Chu et al. (2017): ruído em fake antes do 2º gerador",
                       description="Compara a cíclica L1 com e sem ruído gaussiano na imagem traduzida. Se o erro sobe muito mais "
                                   "do que o ruído justificaria (razão >> 1), o 1º gerador escondeu informação em alta frequência.")
    comum(p)
    p.add_argument("-k", "--k", type=int, default=50, help="imagens de teste por domínio")
    p.add_argument("--sigma", type=float, nargs="+", default=[0.02, 0.05], help="desvio do ruído, em escala [-1, 1]")
    p.add_argument("--limiar", type=float, default=3.0, help="razão acima da qual a linha é marcada como suspeita")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_esteganografia)

    p = sub.add_parser("ablacao", help="curvas e amostras de duas runs lado a lado, em runs/<runB>/ablacao_vs_<runA>.*")
    p.add_argument("runA", help="referência (ex.: principal)")
    p.add_argument("runB", help="variante (ex.: ablacao_id); as saídas vão para a pasta dela")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--epocas", default=None, help="épocas das amostras, separadas por vírgula (padrão: 3 espalhadas)")
    p.set_defaults(fn=cmd_ablacao)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

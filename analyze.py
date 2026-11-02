"""Figuras e análises a partir de data/ e runs/<run>/. Roda na CPU (e no Windows), sem precisar dos .pt."""
import argparse
import csv
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

PASTAS = ["trainA", "trainB", "testA", "testB"]


def listar_imgs(pasta):
    return sorted(p for p in Path(pasta).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def ler_csv(caminho):
    with open(caminho, newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in linha.items()} for linha in csv.DictReader(f)]


def fonte(tamanho):
    return ImageFont.load_default(size=tamanho)


# ---------- dataset ----------

def cmd_dataset(args):
    data, out = Path(args.data), Path(args.out)
    imgs = {}
    for nome in PASTAS:
        imgs[nome] = listar_imgs(data / nome)
        formatos = set()
        for p in imgs[nome]:
            with Image.open(p) as im:  # abre só o cabeçalho
                formatos.add((im.size, im.mode))
        print(f"{nome}: {len(imgs[nome])} imagens; tamanho/modo: {sorted(formatos) or '-'}")
    rng = random.Random(args.seed)
    n, lado = 6, 256
    tela = Image.new("RGB", (n * lado, 2 * lado), "white")
    for linha, nome in enumerate(["trainA", "trainB"]):
        for col, p in enumerate(rng.sample(imgs[nome], min(n, len(imgs[nome])))):
            tela.paste(Image.open(p).convert("RGB").resize((lado, lado)), (col * lado, linha * lado))
    out.mkdir(parents=True, exist_ok=True)
    tela.save(out / "dataset_exemplos.jpg", quality=90)
    print(f"Salvo: {out / 'dataset_exemplos.jpg'}")


# ---------- curvas ----------

def media_movel(ys, k):
    k = max(1, min(k, len(ys)))
    return [sum(ys[max(0, i - k + 1): i + 1]) / (i + 1 - max(0, i - k + 1)) for i in range(len(ys))]


def plot_curvas(linhas, xs, colunas, titulo, ylabel, k, destino):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (col, rotulo) in enumerate(colunas):
        ys = [l[col] for l in linhas]
        cor = f"C{i}"
        ax.plot(xs, ys, color=cor, alpha=0.2, linewidth=1)
        ax.plot(xs, media_movel(ys, k), color=cor, linewidth=2.2, label=rotulo)
    ax.set_title(titulo, fontsize=16)
    ax.set_xlabel("Época", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    print(f"Salvo: {destino}")


def cmd_curvas(args):
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    linhas = ler_csv(run / "perdas.csv")
    if not linhas:
        raise SystemExit("perdas.csv vazio")
    xs = [l["iter_global"] / cfg["iters_por_epoca"] for l in linhas]
    k = args.suavizar
    plot_curvas(linhas, xs, [("G_adv_AB", "G_adv_AB (engana D_B)"), ("G_adv_BA", "G_adv_BA (engana D_A)"),
                             ("D_A", "D_A"), ("D_B", "D_B")],
                "Perdas adversariais (LSGAN)", "Perda", k, run / "curvas_adversarial.png")
    plot_curvas(linhas, xs, [("cyc_A", "cíclica A (A→B→A)"), ("cyc_B", "cíclica B (B→A→B)")],
                "Perda de consistência cíclica (L1)", "L1", k, run / "curvas_ciclica.png")
    if any(l["id_A"] or l["id_B"] for l in linhas):
        plot_curvas(linhas, xs, [("id_A", "identidade A"), ("id_B", "identidade B")],
                    "Perda de identidade (L1)", "L1", k, run / "curvas_identidade.png")
    ep = ler_csv(run / "epocas.csv")
    if ep:
        seg = [e["segundos"] for e in ep]
        print(f"Épocas: {len(ep)} | média {sum(seg) / len(seg):.1f} s/época | total {sum(seg) / 3600:.2f} h | "
              f"{sum(e['its_por_s'] for e in ep) / len(ep):.2f} it/s | VRAM máx {max(e['vram_max_gb'] for e in ep):.2f} GB")


# ---------- evolução ----------

def escolher_epocas(disp, pedidas, maximo=8):
    if pedidas:
        ok = [e for e in pedidas if e in disp]
        for e in set(pedidas) - set(ok):
            print(f"Aviso: sem grade para a época {e}, ignorada.")
        return ok
    if len(disp) <= maximo:
        return disp
    return sorted({disp[round(i * (len(disp) - 1) / (maximo - 1))] for i in range(maximo)})


def cmd_evolucao(args):
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    s, n = cfg["args"]["size"], cfg["args"]["n_amostras"]
    disp = sorted(int(m.group(1)) for p in (run / "amostras").glob("epoca_*.jpg") if (m := re.fullmatch(r"epoca_(\d+)", p.stem)))
    pedidas = [int(e) for e in args.epocas.split(",")] if args.epocas else None
    epocas = escolher_epocas(disp, pedidas)
    if not epocas:
        raise SystemExit("nenhuma grade em amostras/")
    grades = {e: Image.open(run / "amostras" / f"epoca_{e:03d}.jpg").convert("RGB") for e in epocas}
    topo, f = max(24, s // 6), fonte(max(12, s // 8))
    for dom, deslocamento in (("A", 0), ("B", n)):
        tela = Image.new("RGB", ((1 + len(epocas)) * s, topo + n * s), "white")
        d = ImageDraw.Draw(tela)
        d.text((s // 2, topo // 2), "real", fill="black", font=f, anchor="mm")
        for j, e in enumerate(epocas):
            d.text(((j + 1.5) * s, topo // 2), f"trad. ép. {e}", fill="black", font=f, anchor="mm")
        for i in range(n):
            y = (deslocamento + i) * s
            tela.paste(grades[epocas[0]].crop((0, y, s, y + s)), (0, topo + i * s))
            for j, e in enumerate(epocas):
                tela.paste(grades[e].crop((s, y, 2 * s, y + s)), ((j + 1) * s, topo + i * s))
        destino = run / f"evolucao_{dom}.jpg"
        tela.save(destino, quality=90)
        print(f"Salvo: {destino}")


# ---------- traduzir ----------

def cmd_traduzir(args):
    import time

    import torch  # preguiçoso: dataset/curvas/evolucao não carregam torch

    import models

    s = args.size
    if s % 4:
        raise SystemExit("--size precisa ser múltiplo de 4")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G_AB, G_BA, epoca = models.load_generators(args.pesos, dev)
    G_ida, G_volta = (G_AB, G_BA) if args.dominio == "A" else (G_BA, G_AB)
    print(f"Pesos: {args.pesos} (época {epoca}) | entrada: domínio {args.dominio} | dispositivo: {dev}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    legendas = ("original", "traduzida", "reconstruída")
    topo, f = max(24, s // 8), fonte(max(12, s // 10))
    for caminho in args.imagens:
        try:
            with Image.open(caminho) as im:
                img = im.convert("RGB")
        except (OSError, ValueError) as e:
            print(f"Aviso: não consegui ler {caminho} ({e}), ignorada.")
            continue
        t0 = time.time()
        if args.ladrilhar:  # lados múltiplos de size, proporção aproximada
            w, h = (max(1, round(l / s)) * s for l in img.size)
        else:
            w = h = s
        img = img.resize((w, h), Image.LANCZOS)
        # mesma normalização de models.load_image
        x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(h, w, 3).permute(2, 0, 1).float() / 127.5 - 1.0
        trad, rec = torch.zeros_like(x), torch.zeros_like(x)
        with torch.no_grad():
            for y0 in range(0, h, s):
                for x0 in range(0, w, s):
                    bloco = x[None, :, y0:y0 + s, x0:x0 + s].to(dev)
                    b = G_ida(bloco)
                    trad[:, y0:y0 + s, x0:x0 + s] = b[0].cpu()
                    rec[:, y0:y0 + s, x0:x0 + s] = G_volta(b)[0].cpu()
        imagens = (img, models.to_pil(trad), models.to_pil(rec))
        nome = Path(caminho).stem
        for sufixo, im in zip(("original", "traduzida", "reconstruida"), imagens):
            im.save(out / f"{nome}_{sufixo}.jpg", quality=92)
        tela = Image.new("RGB", (3 * w, topo + h), "white")
        d = ImageDraw.Draw(tela)
        for j, (im, leg) in enumerate(zip(imagens, legendas)):
            tela.paste(im, (j * w, topo))
            d.text(((j + 0.5) * w, topo // 2), leg, fill="black", font=f, anchor="mm")
        tela.save(out / f"{nome}_lado_a_lado.jpg", quality=92)
        print(f"Salvo: {out / (nome + '_lado_a_lado.jpg')} ({w}x{h}, {time.time() - t0:.1f} s)")


# ---------- main ----------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dataset", help="contagens e figura de exemplos do dataset")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="figuras")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_dataset)
    p = sub.add_parser("curvas", help="curvas de perda e resumo de tempo")
    p.add_argument("run")
    p.add_argument("--suavizar", type=int, default=20, help="janela da média móvel (em pontos de perdas.csv)")
    p.set_defaults(fn=cmd_curvas)
    p = sub.add_parser("evolucao", help="evolução por época a partir das grades")
    p.add_argument("run")
    p.add_argument("--epocas", default=None, help="lista separada por vírgula, ex.: 1,5,10")
    p.set_defaults(fn=cmd_evolucao)
    p = sub.add_parser("traduzir", help="traduz imagens com os pesos treinados (demo)")
    p.add_argument("--pesos", required=True, help="G_epoca_XXX.pt ou last.pt")
    p.add_argument("--imagens", nargs="+", required=True)
    p.add_argument("--dominio", choices=["A", "B"], default="A", help="domínio de entrada: A usa G_AB (mapa→satélite), B usa G_BA")
    p.add_argument("--out", default="traducoes")
    p.add_argument("--size", type=int, default=256, help="lado do bloco (múltiplo de 4)")
    p.add_argument("--ladrilhar", action="store_true",
                   help="traduz a imagem inteira em blocos size×size e costura o mosaico (sem blending: podem aparecer emendas)")
    p.set_defaults(fn=cmd_traduzir)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

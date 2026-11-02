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
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

"""Monitor do treino: manda o status a cada N min para o celular (ntfy.sh) e avisa na hora se travar.

Roda no PC de treino, em um segundo terminal, só com a stdlib. Sem --topic, só imprime (serve de teste).
"""
import argparse
import csv
import json
import math
import subprocess
import time
import urllib.request
from pathlib import Path


def ler(caminho):
    try:
        with open(caminho, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:  # arquivo ainda não existe ou está momentaneamente travado
        return []


def gpu():
    try:
        t, usada, total = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.split(",")
        return f"GPU {t.strip()} °C, {int(usada) / 1024:.1f}/{int(total) / 1024:.1f} GB"
    except (OSError, ValueError, subprocess.SubprocessError):
        return "GPU: nvidia-smi indisponível"


def resumo(run, parado_min):
    """Devolve (texto, problema ou None, terminou)."""
    try:
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "sem config.json: o treino ainda não começou", None, False
    total, ipe = cfg["args"]["epochs"], cfg["iters_por_epoca"]
    eps, per = ler(run / "epocas.csv"), ler(run / "perdas.csv")
    feitas = len(eps)
    linhas, problema = [f"Épocas concluídas: {feitas}/{total}"], None
    if per:
        u = per[-1]
        it = int(float(u["iter_global"])) - (int(float(u["epoca"])) - 1) * ipe
        linhas.append(f"Agora: época {u['epoca']}, it {it}/{ipe}")
        linhas.append(f"G_adv {float(u['G_adv_AB']):.3f}/{float(u['G_adv_BA']):.3f}  cyc {float(u['cyc_A']):.3f}/{float(u['cyc_B']):.3f}  "
                      f"D {float(u['D_A']):.3f}/{float(u['D_B']):.3f}")
        if not all(math.isfinite(float(v)) for v in u.values()):
            problema = "perda NaN/inf"
        idade = (time.time() - (run / "perdas.csv").stat().st_mtime) / 60
        if idade > parado_min:  # processo morto, travado ou PC suspenso
            problema = f"sem progresso há {idade:.0f} min"
    if eps:
        seg = sum(float(e["segundos"]) for e in eps[-3:]) / len(eps[-3:])
        linhas.append(f"{seg / 60:.1f} min/época (últimas 3) | faltam {(total - feitas) * seg / 3600:.1f} h até a época {total}")
    linhas.append(gpu())
    return "\n".join(linhas), problema, feitas >= total


def enviar(topico, titulo, msg, prioridade="default", tags="chart_with_upwards_trend"):
    print(f"[{time.strftime('%H:%M')}] {titulo}\n{msg}\n", flush=True)
    if not topico:
        return
    req = urllib.request.Request(f"https://ntfy.sh/{topico}", data=msg.encode("utf-8"),
                                 headers={"Title": titulo, "Priority": prioridade, "Tags": tags})  # cabeçalhos só ASCII
    try:
        urllib.request.urlopen(req, timeout=15).close()
    except OSError as e:
        print(f"falha ao enviar para o ntfy: {e}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--topic", help="tópico do ntfy.sh (invente um nome longo e aleatório); sem ele, só imprime")
    p.add_argument("--every", type=float, default=30, help="minutos entre os relatórios")
    p.add_argument("--parado", type=float, default=10, help="minutos sem escrita em perdas.csv para dar alerta de travamento")
    p.add_argument("--once", action="store_true", help="manda um relatório e sai (teste)")
    args = p.parse_args(argv)
    run = Path(args.runs_dir) / args.run
    ultimo, alertado = 0.0, None
    while True:
        texto, problema, terminou = resumo(run, args.parado)
        if problema and problema != alertado:  # alerta na hora, uma vez por problema
            enviar(args.topic, f"ALERTA {args.run}: {problema}", texto, "urgent", "rotating_light")
        alertado = problema
        if terminou:
            enviar(args.topic, f"Treino {args.run} concluido", texto, "high", "white_check_mark")
            return 0
        if time.time() - ultimo >= args.every * 60:
            enviar(args.topic, f"Treino {args.run}", texto)
            ultimo = time.time()
        if args.once:
            return 0
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())

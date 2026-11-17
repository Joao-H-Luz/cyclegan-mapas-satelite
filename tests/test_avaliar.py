"""Testa avaliar.py na CPU: dados sintéticos 64 px, treino mínimo e todos os subcomandos."""
import csv
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import avaliar
import train


def test_metricas():
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    assert torch.allclose(avaliar.cyc_l1(x, x), torch.zeros(2))
    assert torch.allclose(avaliar.ssim(x, x), torch.ones(2), atol=1e-5)
    assert (avaliar.psnr(x, x) > 90).all()
    y = (x + 0.2 * torch.randn_like(x)).clamp(-1, 1)
    assert (avaliar.ssim(y, x) < 0.99).all() and (avaliar.psnr(y, x) < 40).all() and (avaliar.cyc_l1(y, x) > 0).all()


def test_avaliar(tmp_path):
    data = tmp_path / "data"
    rng = random.Random(0)
    for pasta in ["trainA", "trainB", "testA", "testB"]:
        (data / pasta).mkdir(parents=True)
        for i in range(6):
            Image.frombytes("RGB", (64, 64), rng.randbytes(64 * 64 * 3)).save(data / pasta / f"{i}.jpg")

    def treinar(nome, epocas):
        train.main(["--data", str(data), "--run", nome, "--runs-dir", str(tmp_path), "--size", "64", "--max-iters", "2",
                    "--workers", "0", "--snapshot-every", "1", "--log-every", "1", "--epochs", str(epocas)])

    treinar("a", 2)
    treinar("b", 1)
    base = ["--runs-dir", str(tmp_path), "--data", str(data)]
    run = tmp_path / "a"

    avaliar.main(["inferir", "a"] + base)
    with open(run / "metricas_teste.csv", newline="") as f:
        linhas = list(csv.DictReader(f))
    assert len(linhas) == 12 and {l["dominio"] for l in linhas} == {"A", "B"}
    assert len(list((run / "teste" / "A2B").glob("*_traduzida.jpg"))) == 6
    assert len(list((run / "teste" / "B2A").glob("*_reconstruida.jpg"))) == 6

    avaliar.main(["casos", "a", "-n", "2", "--lado", "64"] + base)
    for nome in ["casos_A2B.jpg", "casos_B2A.jpg"]:
        im = Image.open(run / nome)
        assert im.size[0] > 3 * 64 and im.size[1] > 4 * 64

    avaliar.main(["esteganografia", "a", "-k", "4"] + base)
    with open(run / "esteganografia.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 6  # 2 domínios x (sem ruído + 2 sigmas)
    assert (run / "esteganografia.png").exists()

    avaliar.main(["ablacao", "a", "b", "--runs-dir", str(tmp_path)])
    for nome in ["ablacao_vs_a.png", "ablacao_vs_a.jpg"]:
        assert (tmp_path / "b" / nome).exists(), nome

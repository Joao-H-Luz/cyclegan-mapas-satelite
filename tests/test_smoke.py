"""Smoke test na CPU: dados sintéticos 64 px, treino de 2 iterações, resume e análises."""
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analyze
import models
import train


def test_smoke(tmp_path):
    data = tmp_path / "data"
    rng = random.Random(0)
    for pasta in ["trainA", "trainB", "testA", "testB"]:
        (data / pasta).mkdir(parents=True)
        for i in range(6):
            Image.frombytes("RGB", (64, 64), rng.randbytes(64 * 64 * 3)).save(data / pasta / f"{i}.jpg")

    G = models.ResNetGenerator()
    y = G(torch.randn(1, 3, 64, 64))
    assert y.shape == (1, 3, 64, 64) and y.min() >= -1 and y.max() <= 1
    assert models.PatchDiscriminator()(y).ndim == 4

    base = ["--data", str(data), "--run", "t", "--runs-dir", str(tmp_path), "--size", "64", "--max-iters", "2",
            "--workers", "0", "--snapshot-every", "1", "--log-every", "1"]
    train.main(base + ["--epochs", "2"])
    run = tmp_path / "t"
    for nome in ["config.json", "perdas.csv", "epocas.csv", "amostras/epoca_002.jpg", "last.pt", "G_epoca_002.pt"]:
        assert (run / nome).exists(), nome

    train.main(base + ["--epochs", "3", "--resume"])
    assert len((run / "epocas.csv").read_text().strip().splitlines()) == 4  # cabeçalho + 3 épocas
    assert (run / "amostras" / "epoca_003.jpg").exists()

    analyze.main(["curvas", str(run)])
    analyze.main(["evolucao", str(run)])
    analyze.main(["dataset", "--data", str(data), "--out", str(tmp_path / "figuras")])
    for nome in ["curvas_adversarial.png", "curvas_ciclica.png", "evolucao_A.jpg", "evolucao_B.jpg"]:
        assert (run / nome).exists(), nome
    assert (tmp_path / "figuras" / "dataset_exemplos.jpg").exists()
    assert json.loads((run / "config.json").read_text())["args"]["size"] == 64

    G_AB, G_BA, epoca = models.load_generators(run / "G_epoca_003.pt")
    assert epoca == 3 and not G_AB.training and not G_BA.training

"""Smoke test na CPU: dados sintéticos 64 px, treino de 2 iterações, resume e análises."""
import json
import random
import sys
from pathlib import Path

import pytest
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


def test_amostrador_cobre_A_uniformemente():
    nA, nB = 10, 25  # |A| < |B|: A repete, mas de forma uniforme
    gen = torch.Generator().manual_seed(0)
    amostrador = train.AmostradorPares(nA, nB, gen)
    assert len(amostrador) == 25
    for _ in range(3):  # várias épocas
        pares = list(amostrador)
        ia, ib = [p[0] for p in pares], [p[1] for p in pares]
        contagem = [ia.count(i) for i in range(nA)]
        assert set(contagem) <= {2, 3} and sum(contagem) == 25  # floor/ceil, sem favorecer os primeiros
        assert sorted(ib) == list(range(nB))  # domínio maior: permutação completa
    # determinístico por seed
    a = list(train.AmostradorPares(nA, nB, torch.Generator().manual_seed(7)))
    b = list(train.AmostradorPares(nA, nB, torch.Generator().manual_seed(7)))
    assert a == b


def test_resume_de_no_amp_com_scaler_vazio(tmp_path):
    # lógica: scaler ligado + state dict vazio (checkpoint de --no-amp) não pode levantar
    class ScalerLigado(torch.amp.GradScaler):
        def load_state_dict(self, estado):
            if not estado:
                raise RuntimeError("state dict vazio")

    train.carregar(ScalerLigado("cpu", enabled=False), {})  # não levanta
    with pytest.raises(RuntimeError):
        ScalerLigado("cpu", enabled=False).load_state_dict({})  # a fake realmente levantaria

    # ponta a ponta: 1 época com --no-amp, depois --resume sem --no-amp
    data = tmp_path / "data"
    rng = random.Random(1)
    for pasta in ["trainA", "trainB", "testA", "testB"]:
        (data / pasta).mkdir(parents=True)
        for i in range(6):
            Image.frombytes("RGB", (64, 64), rng.randbytes(64 * 64 * 3)).save(data / pasta / f"{i}.jpg")
    base = ["--data", str(data), "--run", "r", "--runs-dir", str(tmp_path), "--size", "64", "--max-iters", "1",
            "--workers", "0", "--log-every", "1"]
    train.main(base + ["--epochs", "1", "--no-amp"])
    assert train.main(base + ["--epochs", "2", "--resume"]) == 0
    assert (tmp_path / "r" / "amostras" / "epoca_002.jpg").exists()

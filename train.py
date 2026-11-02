"""Treino da CycleGAN (mapas históricos <-> satélite) com checkpoint/resume."""
import argparse
import csv
import itertools
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from models import PatchDiscriminator, ResNetGenerator, init_weights, load_image, to_pil

CHAVES = ["G_adv_AB", "G_adv_BA", "cyc_A", "cyc_B", "id_A", "id_B", "D_A", "D_B"]
COLS_PERDAS = ["epoca", "iter_global"] + CHAVES
COLS_EPOCAS = ["epoca", "segundos", "its_por_s", "lr", "vram_max_gb"] + CHAVES
EXTS = {".jpg", ".jpeg", ".png"}


def listar(pasta):
    if not pasta.is_dir():
        raise SystemExit(f"Pasta não encontrada: {pasta}")
    return sorted(p for p in pasta.iterdir() if p.suffix.lower() in EXTS)


class Pares(torch.utils.data.Dataset):
    """A segue a ordem embaralhada do sampler; B é sorteado (não pareado)."""

    def __init__(self, A, B, size):
        self.A, self.B, self.size = A, B, size

    def __len__(self):
        return max(len(self.A), len(self.B))

    def __getitem__(self, i):
        a = self.A[i % len(self.A)]
        b = self.B[random.randrange(len(self.B))]
        return load_image(a, self.size), load_image(b, self.size)


class Pool:
    """Image pool de 50 (Shrivastava et al.) para as falsas que vão ao D."""

    def __init__(self, n=50):
        self.n, self.imgs = n, []

    def query(self, x):
        if len(self.imgs) < self.n:
            self.imgs.append(x)
            return x
        if random.random() < 0.5:
            i = random.randrange(self.n)
            x, self.imgs[i] = self.imgs[i], x
        return x


def mse(pred, alvo):
    return F.mse_loss(pred.float(), torch.full_like(pred, alvo, dtype=torch.float32))


def anexar(path, cols, linha):
    novo = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(cols)
        w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in linha])


def podar_csv(path, cols, epoca):
    """Mantém só as linhas com epoca <= epoca do checkpoint (órfãs de uma interrupção)."""
    if not path.exists():
        return
    with path.open(newline="") as f:
        linhas = list(csv.reader(f))
    manter = [linhas[0]] + [r for r in linhas[1:] if r and int(r[0]) <= epoca]
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(manter)


def salvar_atomico(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main(argv=None):
    p = argparse.ArgumentParser(description="Treina uma CycleGAN (A = mapas históricos, B = satélite).")
    p.add_argument("--data", default="data")
    p.add_argument("--run", required=True)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--decay-start", type=int, default=50)
    p.add_argument("--lambda-cyc", type=float, default=10.0)
    p.add_argument("--lambda-id", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--snapshot-every", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--n-amostras", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args(argv)

    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")
    nome_dev = torch.cuda.get_device_name(0) if cuda else "cpu"
    amp = cuda and not args.no_amp
    torch.backends.cudnn.benchmark = cuda
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Dispositivo: {dev.type} ({nome_dev}) | AMP fp16: {'sim' if amp else 'não'}")

    data = Path(args.data)
    treino = {d: listar(data / f"train{d}") for d in "AB"}
    teste = {d: listar(data / f"test{d}") for d in "AB"}
    iters = max(len(treino["A"]), len(treino["B"]))
    if args.max_iters:
        iters = min(iters, args.max_iters)

    run = Path(args.runs_dir) / args.run
    cfg_path, last = run / "config.json", run / "last.pt"
    if last.exists() and not args.resume:
        raise SystemExit(f"{last} já existe: use --resume para continuar ou outro nome em --run.")
    if args.resume and not (last.exists() and cfg_path.exists()):
        raise SystemExit(f"--resume sem {last} / {cfg_path}: nada para continuar.")
    (run / "amostras").mkdir(parents=True, exist_ok=True)

    # amostras fixas: sorteadas na 1ª execução e reaproveitadas no resume
    if args.resume:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        cfg = {
            "args": vars(args), "torch": torch.__version__, "dispositivo": nome_dev,
            "inicio": datetime.now().isoformat(timespec="seconds"),
            "n_train": {d: len(treino[d]) for d in "AB"}, "iters_por_epoca": iters,
            "amostras_fixas": {d: [q.name for q in random.Random(args.seed).sample(sorted(teste[d]), args.n_amostras)] for d in "AB"},
        }
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    fixas = {d: torch.stack([load_image(data / f"test{d}" / n, args.size) for n in cfg["amostras_fixas"][d]]).to(dev) for d in "AB"}

    G_AB, G_BA = ResNetGenerator().to(dev), ResNetGenerator().to(dev)
    D_A, D_B = PatchDiscriminator().to(dev), PatchDiscriminator().to(dev)
    for net in (G_AB, G_BA, D_A, D_B):
        init_weights(net)
    opt_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), args.lr, betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ini = 1
    if args.resume:
        ck = torch.load(last, map_location=dev, weights_only=True)
        for k, net in (("G_AB", G_AB), ("G_BA", G_BA), ("D_A", D_A), ("D_B", D_B), ("opt_G", opt_G), ("opt_D", opt_D), ("scaler", scaler)):
            net.load_state_dict(ck[k])
        ini = ck["epoca"] + 1
        podar_csv(run / "perdas.csv", COLS_PERDAS, ck["epoca"])
        podar_csv(run / "epocas.csv", COLS_EPOCAS, ck["epoca"])
        print(f"Retomando da época {ck['epoca']} (próxima: {ini}).")
        if ini > args.epochs:
            print("Nada a fazer: run já concluída.")
            return 0
    else:
        for f in ("perdas.csv", "epocas.csv"):  # restos de uma tentativa que não chegou a salvar last.pt
            (run / f).unlink(missing_ok=True)

    ds = Pares(treino["A"], treino["B"], args.size)
    gen = torch.Generator().manual_seed(args.seed + ini)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=True, num_workers=args.workers, generator=gen,
        pin_memory=cuda, persistent_workers=args.workers > 0)
    pool_A, pool_B = Pool(), Pool()
    ac = lambda: torch.autocast(dev.type, enabled=amp)  # noqa: E731
    lam_id = args.lambda_cyc * args.lambda_id
    ultima = ini - 1

    def congelar(nets, req):
        for n in nets:
            for q in n.parameters():
                q.requires_grad_(req)

    @torch.no_grad()
    def grade():
        linhas = []
        for dom, g1, g2 in (("A", G_AB, G_BA), ("B", G_BA, G_AB)):
            with ac():
                real = fixas[dom]
                fake = g1(real)
                rec = g2(fake)
            for r, f, c in zip(real, fake.float(), rec.float()):
                linhas.append(torch.cat([r, f, c], dim=2))
        return to_pil(torch.cat(linhas, dim=1))

    nao_finitas = 0  # janelas seguidas com NaN/inf: aborta antes de gravar um last.pt estragado
    try:
        for ep in range(ini, args.epochs + 1):
            lr = args.lr * (1 - max(0, ep - args.decay_start) / (args.epochs - args.decay_start + 1))
            for o in (opt_G, opt_D):
                for g in o.param_groups:
                    g["lr"] = lr
            soma_ep = torch.zeros(len(CHAVES), device=dev)
            janela = torch.zeros(len(CHAVES), device=dev)
            n_jan, t0 = 0, time.time()
            t_log = t0
            for it, (a, b) in enumerate(loader, 1):
                if it > iters:
                    break
                a, b = a.to(dev, non_blocking=True), b.to(dev, non_blocking=True)

                # --- passo de G (D congelados) ---
                congelar((D_A, D_B), False)
                opt_G.zero_grad(set_to_none=True)
                with ac():
                    fake_B, fake_A = G_AB(a), G_BA(b)
                    rec_A, rec_B = G_BA(fake_B), G_AB(fake_A)
                    adv_AB, adv_BA = mse(D_B(fake_B), 1.0), mse(D_A(fake_A), 1.0)
                    cyc_A, cyc_B = F.l1_loss(rec_A.float(), a), F.l1_loss(rec_B.float(), b)
                    loss_G = adv_AB + adv_BA + args.lambda_cyc * (cyc_A + cyc_B)
                    if args.lambda_id > 0:
                        id_A, id_B = F.l1_loss(G_BA(a).float(), a), F.l1_loss(G_AB(b).float(), b)
                        loss_G = loss_G + lam_id * (id_A + id_B)
                    else:
                        id_A = id_B = torch.zeros((), device=dev)
                scaler.scale(loss_G).backward()
                scaler.step(opt_G)

                # --- passo de D (fakes do pool, detach) ---
                congelar((D_A, D_B), True)
                opt_D.zero_grad(set_to_none=True)
                with ac():
                    fa, fb = pool_A.query(fake_A.detach()), pool_B.query(fake_B.detach())
                    loss_DA = 0.5 * (mse(D_A(a), 1.0) + mse(D_A(fa), 0.0))
                    loss_DB = 0.5 * (mse(D_B(b), 1.0) + mse(D_B(fb), 0.0))
                    loss_D = loss_DA + loss_DB
                scaler.scale(loss_D).backward()
                scaler.step(opt_D)
                scaler.update()

                v = torch.stack([x.detach().float() for x in (adv_AB, adv_BA, cyc_A, cyc_B, id_A, id_B, loss_DA, loss_DB)])
                soma_ep += v
                janela += v
                n_jan += 1
                if it % args.log_every == 0 or it == iters:
                    med = (janela / n_jan).tolist()  # sincroniza com a GPU
                    ig = (ep - 1) * iters + it
                    anexar(run / "perdas.csv", COLS_PERDAS, [ep, ig] + med)
                    nao_finitas = 0 if all(math.isfinite(x) for x in med) else nao_finitas + 1
                    if nao_finitas >= 3:
                        raise SystemExit("Perdas não finitas por 3 janelas seguidas: abortando. Retome com --resume --no-amp.")
                    if it % args.log_every == 0:
                        agora = time.time()
                        ips = n_jan / (agora - t_log)
                        vram = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
                        if nao_finitas:
                            print("AVISO: perda não finita (NaN/inf). Tente --no-amp ou um lr menor.")
                        print(f"ép {ep}/{args.epochs} it {it}/{iters} | {ips:.2f} it/s | ETA ép {(iters - it) / ips / 60:.1f} min | "
                              f"VRAM {vram:.2f} GB | G_adv {med[0]:.3f}/{med[1]:.3f} cyc {med[2]:.3f}/{med[3]:.3f} D {med[6]:.3f}/{med[7]:.3f}", flush=True)
                        t_log = agora
                    janela.zero_()
                    n_jan = 0

            # --- fim da época: tudo só é "salvo" quando last.pt é gravado ---
            n_it = min(it, iters)
            seg = time.time() - t0
            med = (soma_ep / n_it).tolist()
            vram = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
            anexar(run / "epocas.csv", COLS_EPOCAS, [ep, seg, n_it / seg, lr, vram] + med)
            grade().save(run / "amostras" / f"epoca_{ep:03d}.jpg", quality=90)
            if ep % args.snapshot_every == 0 or ep == args.epochs:
                snap = {"epoca": ep, **{k: {n: t.half() if t.is_floating_point() else t for n, t in g.state_dict().items()}
                                        for k, g in (("G_AB", G_AB), ("G_BA", G_BA))}}
                salvar_atomico(snap, run / f"G_epoca_{ep:03d}.pt")
            salvar_atomico({"epoca": ep, "G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(), "D_A": D_A.state_dict(),
                            "D_B": D_B.state_dict(), "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
                            "scaler": scaler.state_dict()}, last)
            ultima = ep
            print(f"== época {ep} concluída em {seg / 60:.1f} min (lr {lr:.2e}); last.pt salvo.", flush=True)
    except KeyboardInterrupt:
        print(f"\nInterrompido. Última época salva: {ultima or 'nenhuma'}. Continue com --resume.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())

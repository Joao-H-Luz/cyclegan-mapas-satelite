"""
Limpeza do Domínio A: remove recortes que não são "pedaços de mapa" (ilustrações, vistas,
rosas dos ventos, cartuchos de título, legendas, texto, papel em branco...).

Usa o CLIP (classificação zero-shot): cada imagem é comparada com descrições textuais de
classes "boas" (mapa) e "ruins" (ruído). A probabilidade somada das classes de mapa vira um
score; recortes abaixo do limiar são MOVIDOS (não apagados) para <out>/descartadosA/,
de modo que dá para revisar e devolver qualquer um manualmente.

Uso:
  python limpar_mapas.py --dry-run          # só calcula scores e mostra quantos sairiam
  python limpar_mapas.py                    # move os descartados

Regras de descarte:
  - recorte com score de mapa < --threshold;
  - todos os recortes de um mapa em que mais de --map-bad-frac dos recortes foram reprovados
    (ex.: mapas pictóricos/ilustrados, que não têm correspondência com imagens de satélite).
"""
import argparse
import csv
import glob
import os
import shutil
from collections import defaultdict

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL = "openai/clip-vit-base-patch16"

MAP_PROMPTS = [
    "a close-up crop of an old printed map showing streets and city blocks",
    "a crop of a historical topographic map with contour lines, hills and rivers",
    "a crop of an old map with roads, rivers, railways and small place names",
    "a crop of a colored real estate atlas map with building lots and street names",
    "a crop of an old nautical chart showing a coastline and water depth numbers",
    "a crop of an old hand-colored map with regions and borders",
]
NOISE_PROMPTS = [
    "an engraved illustration of people",
    "an engraving of a building, a church or a landscape view",
    "a decorative compass rose",
    "an ornate decorative title cartouche",
    "large decorative lettering",
    "a page of printed text",
    "a map legend or color key with text labels",
    "a scale bar with numbers",
    "blank old paper with a straight border line",
    "a pictorial illustrated poster with drawings of figures",
    "a portrait or a statue",
    "a coat of arms",
    "a colorful cartoon pictorial map with illustrated people, animals and buildings",
    "a drawing of a factory, a ship or a scene",
    "a statistical chart or graph",
    "an empty margin of an old map with only a few straight lines",
    "a table of names and numbers",
]


def score_images(paths, batch_size, device):
    model = CLIPModel.from_pretrained(MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    prompts = MAP_PROMPTS + NOISE_PROMPTS
    with torch.no_grad():
        t = proc(text=prompts, return_tensors="pt", padding=True).to(device)
        text_emb = model.text_projection(model.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).pooler_output)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    results = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        with torch.no_grad():
            x = proc(images=imgs, return_tensors="pt").to(device)
            emb = model.visual_projection(model.vision_model(pixel_values=x["pixel_values"]).pooler_output)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            probs = (100.0 * emb @ text_emb.T).softmax(dim=-1).cpu()
        for p, pr in zip(batch, probs):
            p_map = pr[:len(MAP_PROMPTS)].sum().item()
            top = prompts[pr.argmax().item()]
            results.append((p, p_map, top))
        if (i // batch_size) % 20 == 0:
            print(f"  {min(i + batch_size, len(paths))}/{len(paths)}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./dataset_rumsey_satelite")
    ap.add_argument("--threshold", type=float, default=0.6, help="score mínimo de 'mapa' para manter")
    ap.add_argument("--map-bad-frac", type=float, default=0.25,
                    help="descarta o mapa inteiro se esta fração dos seus recortes for reprovada")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    scores_path = os.path.join(args.out, "scores_clip_A.csv")

    paths = sorted(glob.glob(os.path.join(args.out, "trainA", "*.jpg")) +
                   glob.glob(os.path.join(args.out, "testA", "*.jpg")))
    cached = {}
    if os.path.exists(scores_path):
        with open(scores_path, newline="", encoding="utf-8") as f:
            cached = {r["file"]: (float(r["p_map"]), r["top_prompt"]) for r in csv.DictReader(f)}
    todo = [p for p in paths if os.path.basename(p) not in cached]
    print(f"{len(paths)} imagens em trainA/testA; {len(todo)} sem score (device={device})")
    if todo:
        for p, s, top in score_images(todo, args.batch_size, device):
            cached[os.path.basename(p)] = (s, top)
        with open(scores_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "p_map", "top_prompt"])
            for name, (s, top) in sorted(cached.items()):
                w.writerow([name, f"{s:.4f}", top])

    # Nome: rumsey_<id do mapa>_<x>_<y>.jpg
    def map_of(p):
        return os.path.basename(p).rsplit("_", 2)[0]

    by_map = defaultdict(list)
    for p in paths:
        by_map[map_of(p)].append(p)
    low = {p for p in paths if cached[os.path.basename(p)][0] < args.threshold}
    bad_maps = {m for m, ps in by_map.items() if sum(p in low for p in ps) / len(ps) > args.map_bad_frac}
    bad = [p for p in paths if p in low or map_of(p) in bad_maps]
    print(f"Limiar {args.threshold}: {len(low)} recortes reprovados; {len(bad_maps)} mapas inteiros descartados")
    print(f"Total descartado: {len(bad)} de {len(paths)} ({100 * len(bad) / max(len(paths), 1):.1f}%)")
    if args.dry_run:
        return

    for p in bad:
        split = os.path.basename(os.path.dirname(p))  # trainA ou testA
        dest = os.path.join(args.out, "descartadosA", split)
        os.makedirs(dest, exist_ok=True)
        shutil.move(p, os.path.join(dest, os.path.basename(p)))
    for d in ("trainA", "testA"):
        print(f"{d}: {len(os.listdir(os.path.join(args.out, d)))} imagens restantes")


if __name__ == "__main__":
    main()

"""
Coleta do Domínio A (mapas históricos) a partir da David Rumsey Map Collection.

Fluxo:
  1. Busca mapas na API de busca do LUNA (davidrumsey.com) por palavras-chave.
  2. Para cada mapa, consulta o serviço IIIF (info.json) para saber a resolução real.
  3. Baixa recortes aleatórios em alta resolução via IIIF (região -> 256x256), de modo
     que a escala fique parecida com a de um tile de satélite (quarteirões/ruas),
     em vez de reduzir o mapa inteiro.
  4. Filtra recortes vazios (margens, papel liso, texto) e separa treino/teste POR MAPA,
     evitando que recortes do mesmo mapa apareçam nos dois conjuntos.

A Library of Congress (loc.gov) está atrás de um desafio anti-bot do Cloudflare e
devolve HTTP 403 para qualquer acesso via script, por isso não é usada aqui.

O script é retomável: mapas já processados (registrados em metadata.csv) são pulados.
"""
import argparse
import csv
import io
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from PIL import Image

SEARCH_URL = "https://www.davidrumsey.com/luna/servlet/as/search"
IIIF_BASE = "https://www.davidrumsey.com/luna/servlet/iiif"
COLLECTION = "RUMSEY~8~1"
HEADERS = {"User-Agent": "PUCRS-T4-CycleGAN-dataset/1.0 (trabalho academico)"}

# Mapas com detalhe de terreno/ruas, que conversam melhor com imagens de satélite
DEFAULT_QUERIES = [
    "city plan",
    "town",
    "street map",
    "real estate atlas",
    "ward",
    "plat",
    "topographic",
    "survey",
    "harbor",
    "environs",
    "county atlas",
    "Sanborn",
]

# Só itens catalogados como mapa e sem nenhum tipo que indique outra coisa
BAD_TYPES = ("view", "diagram", "text", "cover", "title", "contents", "index", "legend", "binding")
# Títulos de vistas em perspectiva e de mapas de escala muito pequena (continentes, mundo)
BAD_TITLE = re.compile(
    r"\b(view|views|vol d.oiseau|vogelschau|veduta|panoram|pittorico|black rock|world|monde|welt|mundi|"
    r"hemisphere|globe|america|amerika|europe|europa|asia|africa|oceania|continent)\b",
    re.IGNORECASE,
)
MIN_YEAR, MAX_YEAR = 1700, 1950

session = requests.Session()
session.headers.update(HEADERS)


def get(url, params=None, timeout=60, retries=3):
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(5 * (attempt + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(3 * (attempt + 1))
    return None


def field(item, name):
    for fv in item.get("fieldValues", []):
        if name in fv:
            return "; ".join(fv[name])
    return ""


def year_of(date):
    m = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", date)
    return int(m.group(1)) if m else None


def is_wanted(item, seen):
    name = item.get("displayName", "").strip()
    if item.get("mediaType") != "Image" or item["id"] in seen:
        return False
    types = field(item, "Type").lower()
    if "map" not in types or any(b in types for b in BAD_TYPES):
        return False
    if name.lower().startswith(("covers", "title", "text", "index", "(text")) or BAD_TITLE.search(name):
        return False
    year = year_of(field(item, "Date"))
    return year is not None and MIN_YEAR <= year <= MAX_YEAR


def search_maps(queries, max_per_query, page_size=100, max_offset=3000):
    """Retorna uma lista de dicts {id, title, date, query} sem duplicatas."""
    maps, seen = [], set()
    for q in queries:
        found = 0
        offset = 0
        while found < max_per_query and offset < max_offset:
            r = get(SEARCH_URL, params={"q": q, "lc": COLLECTION, "os": offset, "bs": page_size})
            if r is None:
                print(f"  Falha na busca '{q}' (offset {offset})")
                break
            results = r.json().get("results", [])
            if not results:
                break
            for it in results:
                if not is_wanted(it, seen):
                    continue
                name = it.get("displayName", "").strip()
                seen.add(it["id"])
                maps.append({"id": it["id"], "title": name[:200], "date": field(it, "Date"), "query": q})
                found += 1
                if found >= max_per_query:
                    break
            offset += page_size
            time.sleep(0.5)
        print(f"Busca '{q}': {found} mapas")
    print(f"Total: {len(maps)} mapas únicos\n")
    return maps


def is_informative(arr, min_std, min_edges, max_flat):
    """Rejeita papel liso, margens/molduras e áreas quase uniformes."""
    gray = arr.mean(axis=2)
    # Fração de blocos 32x32 praticamente lisos (molduras, margens com uma ou duas linhas)
    h, w = gray.shape
    blocks = gray[: h - h % 32, : w - w % 32].reshape(h // 32, 32, w // 32, 32).std(axis=(1, 3))
    if (blocks < 6).mean() > max_flat:
        return False
    if gray.std() < min_std:
        return False
    mean = gray.mean()
    if mean < 25 or mean > 245:
        return False
    # Fração de pixels com gradiente relevante (linhas, texturas, hachuras)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edges = ((gx[:-1, :] > 40) | (gy[:, :-1] > 40)).mean()
    return edges >= min_edges


def process_map(m, args):
    """Baixa recortes aleatórios de um mapa. Retorna (map_id, lista de (PIL.Image, nome), largura, altura)."""
    svc = f"{IIIF_BASE}/{m['id']}"
    info = get(f"{svc}/info.json", timeout=30)
    if info is None:
        return m, [], 0, 0
    info = info.json()
    W, H = info["width"], info["height"]
    region = args.region
    if W < region * 2 or H < region * 2:
        return m, [], W, H

    rng = random.Random(m["id"])
    # Ignora uma margem de 7% para fugir de bordas/molduras do mapa
    mx, my = int(W * 0.07), int(H * 0.07)
    patches, attempts = [], 0
    while len(patches) < args.patches_per_map and attempts < args.patches_per_map * 3:
        attempts += 1
        x = rng.randint(mx, W - mx - region)
        y = rng.randint(my, H - my - region)
        url = f"{svc}/{x},{y},{region},{region}/{args.size},{args.size}/0/default.jpg"
        r = get(url, timeout=60)
        if r is None:
            continue
        try:
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            continue
        if is_informative(np.asarray(img, dtype=np.float32), args.min_std, args.min_edges, args.max_flat):
            safe_id = m["id"].replace("~", "_")
            patches.append((img, f"rumsey_{safe_id}_{x}_{y}.jpg"))
        time.sleep(args.delay)
    return m, patches, W, H


def main():
    p = argparse.ArgumentParser(description="Coleta mapas históricos (David Rumsey) para CycleGAN")
    p.add_argument("--out", default="./dataset_rumsey_satelite")
    p.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    p.add_argument("--max-per-query", type=int, default=60, help="mapas por palavra-chave")
    p.add_argument("--patches-per-map", type=int, default=12)
    p.add_argument("--region", type=int, default=1024,
                   help="lado do recorte em pixels do mapa original (maior = mais 'zoom out')")
    p.add_argument("--size", type=int, default=256, help="lado da imagem final")
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--delay", type=float, default=0.2, help="pausa entre requisições por worker")
    p.add_argument("--min-std", type=float, default=18.0)
    p.add_argument("--min-edges", type=float, default=0.08)
    p.add_argument("--max-flat", type=float, default=0.30)
    args = p.parse_args()

    train_dir = os.path.join(args.out, "trainA")
    test_dir = os.path.join(args.out, "testA")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    meta_path = os.path.join(args.out, "metadata.csv")

    done = set()
    if os.path.exists(meta_path):
        with open(meta_path, newline="", encoding="utf-8") as f:
            done = {row["map_id"] for row in csv.DictReader(f)}

    maps = [m for m in search_maps(args.queries, args.max_per_query) if m["id"] not in done]
    print(f"{len(done)} mapas já processados; {len(maps)} novos para baixar.\n")

    new_file = not os.path.exists(meta_path)
    n_train = n_test = 0
    with open(meta_path, "a", newline="", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["map_id", "title", "date", "query", "width", "height", "split", "n_patches"])
        futures = [ex.submit(process_map, m, args) for m in maps]
        for i, fut in enumerate(as_completed(futures), 1):
            m, patches, W, H = fut.result()
            # Divisão treino/teste por mapa (determinística pelo id)
            split = "test" if random.Random("split" + m["id"]).random() < args.test_ratio else "train"
            target = test_dir if split == "test" else train_dir
            for img, name in patches:
                img.save(os.path.join(target, name), quality=95)
            if split == "test":
                n_test += len(patches)
            else:
                n_train += len(patches)
            writer.writerow([m["id"], m["title"], m["date"], m["query"], W, H, split, len(patches)])
            f.flush()
            print(f"[{i}/{len(maps)}] {len(patches):2d} patches ({split}) {W}x{H}  {m['title'][:60]}")

    print(f"\nNovos patches: trainA +{n_train}, testA +{n_test}")
    print(f"Total em disco: trainA {len(os.listdir(train_dir))}, testA {len(os.listdir(test_dir))}")


if __name__ == "__main__":
    main()

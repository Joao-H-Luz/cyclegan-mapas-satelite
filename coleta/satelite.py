"""
Coleta do Domínio B (imagens de satélite/aéreas) a partir do USGS The National Map.

Fonte: serviço de tiles "USGSImageryOnly" (ortoimagens NAIP / High Resolution Orthoimagery),
domínio público (dados do governo dos EUA), sem necessidade de chave de API.
  https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer

Fluxo:
  1. Para cada localidade (cidades com mapas históricos no acervo Rumsey + áreas rurais,
     costeiras e montanhosas, que correspondem aos mapas topográficos), sorteia tiles
     aleatórios dentro de um raio em torno do centro.
  2. Mistura níveis de zoom (14-16) para cobrir a variação de escala dos mapas históricos.
  3. Filtra tiles sem informação (áreas sem cobertura, só água, nuvens/uniformes).
  4. Separa treino/teste por bloco espacial (tile-pai de zoom 11), para que tiles vizinhos
     não fiquem divididos entre treino e teste.

O script é retomável: tiles já baixados (registrados em metadata.csv) são pulados.
"""
import argparse
import csv
import io
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from PIL import Image

TILE_URL = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
HEADERS = {"User-Agent": "PUCRS-T4-CycleGAN-dataset/1.0 (trabalho academico)"}

# (nome, latitude, longitude, raio em km)
URBAN = [
    ("Boston", 42.355, -71.060, 8), ("New York", 40.735, -73.990, 10), ("Brooklyn", 40.680, -73.950, 7),
    ("Philadelphia", 39.955, -75.165, 8), ("Washington", 38.900, -77.030, 8), ("Baltimore", 39.290, -76.610, 6),
    ("Pittsburgh", 40.445, -79.990, 6), ("Chicago", 41.880, -87.640, 10), ("Detroit", 42.335, -83.050, 8),
    ("Cleveland", 41.495, -81.690, 6), ("St. Louis", 38.630, -90.200, 7), ("New Orleans", 29.955, -90.075, 6),
    ("San Francisco", 37.775, -122.430, 6), ("Oakland", 37.805, -122.270, 6), ("Los Angeles", 34.050, -118.250, 12),
    ("Seattle", 47.610, -122.335, 7), ("Portland", 45.520, -122.675, 6), ("Denver", 39.740, -104.990, 7),
    ("Salt Lake City", 40.760, -111.890, 6), ("Charleston", 32.785, -79.935, 4), ("Savannah", 32.075, -81.095, 4),
    ("Richmond", 37.540, -77.435, 5), ("Providence", 41.825, -71.410, 5), ("Newport", 41.490, -71.315, 3),
    ("Hartford", 41.765, -72.680, 5), ("Buffalo", 42.885, -78.875, 6), ("Cincinnati", 39.105, -84.510, 6),
    ("Milwaukee", 43.040, -87.910, 6), ("Minneapolis", 44.975, -93.265, 7), ("Kansas City", 39.100, -94.580, 6),
    ("San Antonio", 29.425, -98.495, 7), ("Sacramento", 38.580, -121.495, 5), ("Fresno", 36.740, -119.785, 5),
    ("San Diego", 32.715, -117.160, 6), ("Galveston", 29.300, -94.800, 4), ("Albany", 42.650, -73.755, 4),
]
NATURAL = [
    ("Rocky Mountains CO", 39.600, -105.700, 25), ("Appalachians WV", 38.600, -79.800, 30),
    ("White Mountains NH", 44.200, -71.400, 20), ("Sierra Nevada CA", 37.900, -119.500, 25),
    ("Cascades WA", 47.400, -121.400, 25), ("Hudson Valley NY", 41.700, -73.950, 20),
    ("Pennsylvania farmland", 40.050, -76.200, 20), ("Iowa farmland", 41.900, -93.600, 30),
    ("Shenandoah VA", 38.700, -78.300, 20), ("Cape Cod MA", 41.700, -70.100, 15),
    ("Chesapeake Bay MD", 38.700, -76.300, 20), ("Puget Sound WA", 47.300, -122.600, 15),
    ("Columbia River OR", 45.650, -121.900, 20), ("Mississippi Delta LA", 29.700, -90.000, 20),
    ("Lake Champlain VT", 44.500, -73.250, 15), ("Ozarks MO", 36.900, -92.600, 25),
]

session = requests.Session()
session.headers.update(HEADERS)


def latlon_to_tile(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def sample_tiles(loc, n, zooms, weights, rng):
    """Sorteia n tiles (z, x, y) distintos dentro do raio da localidade."""
    name, lat, lon, radius_km = loc
    tiles = set()
    attempts = 0
    while len(tiles) < n and attempts < n * 20:
        attempts += 1
        # Ponto uniforme dentro do círculo
        r = radius_km * math.sqrt(rng.random())
        t = rng.random() * 2 * math.pi
        dlat = (r * math.cos(t)) / 111.0
        dlon = (r * math.sin(t)) / (111.0 * math.cos(math.radians(lat)))
        z = rng.choices(zooms, weights=weights)[0]
        x, y = latlon_to_tile(lat + dlat, lon + dlon, z)
        tiles.add((z, x, y))
    return sorted(tiles)


def is_informative(arr, min_std, max_flat):
    """Rejeita tiles sem cobertura (preto/branco), só água ou muito uniformes."""
    gray = arr.mean(axis=2)
    mean = gray.mean()
    if mean < 15 or mean > 240 or gray.std() < min_std:
        return False
    blocks = gray.reshape(8, 32, 8, 32).std(axis=(1, 3))
    return (blocks < 4).mean() <= max_flat


def fetch(task, args):
    loc_name, z, x, y = task
    url = TILE_URL.format(z=z, x=x, y=y)
    for attempt in range(3):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                if img.size != (256, 256):
                    img = img.resize((256, 256), Image.BICUBIC)
                ok = is_informative(np.asarray(img, dtype=np.float32), args.min_std, args.max_flat)
                time.sleep(args.delay)
                return task, (img if ok else None)
            if r.status_code in (404, 400):
                return task, None
        except (requests.RequestException, OSError):
            pass
        time.sleep(3 * (attempt + 1))
    return task, None


def main():
    p = argparse.ArgumentParser(description="Coleta imagens de satélite (USGS National Map) para CycleGAN")
    p.add_argument("--out", default="./dataset_rumsey_satelite")
    p.add_argument("--tiles-per-city", type=int, default=170)
    p.add_argument("--tiles-per-natural", type=int, default=140)
    p.add_argument("--zooms", type=int, nargs="+", default=[14, 15, 16])
    p.add_argument("--zoom-weights", type=float, nargs="+", default=[0.2, 0.45, 0.35])
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--delay", type=float, default=0.1)
    p.add_argument("--min-std", type=float, default=12.0)
    p.add_argument("--max-flat", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_dir = os.path.join(args.out, "trainB")
    test_dir = os.path.join(args.out, "testB")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    meta_path = os.path.join(args.out, "metadata_satelite.csv")

    done = set()
    if os.path.exists(meta_path):
        with open(meta_path, newline="", encoding="utf-8") as f:
            done = {(int(r["z"]), int(r["x"]), int(r["y"])) for r in csv.DictReader(f)}

    rng = random.Random(args.seed)
    tasks = []
    for loc in URBAN:
        tasks += [(loc[0], *t) for t in sample_tiles(loc, args.tiles_per_city, args.zooms, args.zoom_weights, rng)]
    for loc in NATURAL:
        tasks += [(loc[0], *t) for t in sample_tiles(loc, args.tiles_per_natural, args.zooms, args.zoom_weights, rng)]
    tasks = [t for t in tasks if (t[1], t[2], t[3]) not in done]
    print(f"{len(done)} tiles já processados; {len(tasks)} novos para baixar.\n")

    new_file = not os.path.exists(meta_path)
    n_train = n_test = n_rej = 0
    with open(meta_path, "a", newline="", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["location", "z", "x", "y", "split", "kept"])
        futures = [ex.submit(fetch, t, args) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            (loc_name, z, x, y), img = fut.result()
            # Bloco espacial = tile-pai no zoom 11; todos os tiles do bloco caem no mesmo split
            shift = z - 11
            block = f"{x >> shift}_{y >> shift}"
            split = "test" if random.Random("split" + block).random() < args.test_ratio else "train"
            if img is not None:
                slug = loc_name.lower().replace(" ", "-").replace(".", "")
                name = f"usgs_{slug}_z{z}_{x}_{y}.jpg"
                img.save(os.path.join(test_dir if split == "test" else train_dir, name), quality=95)
                if split == "test":
                    n_test += 1
                else:
                    n_train += 1
            else:
                n_rej += 1
            writer.writerow([loc_name, z, x, y, split, int(img is not None)])
            if i % 500 == 0:
                f.flush()
                print(f"[{i}/{len(tasks)}] train {n_train}  test {n_test}  rejeitados {n_rej}")

    print(f"\nNovos tiles: trainB +{n_train}, testB +{n_test}, rejeitados {n_rej}")
    print(f"Total em disco: trainB {len(os.listdir(train_dir))}, testB {len(os.listdir(test_dir))}")


if __name__ == "__main__":
    main()

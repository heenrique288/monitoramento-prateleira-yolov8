# -*- coding: utf-8 -*-
"""
Prepara um subconjunto do dataset SKU-110K (prateleiras reais de supermercado,
classe única "produto") no formato YOLO.

Fonte: espelho Parquet do SKU-110K no Hugging Face
       https://huggingface.co/datasets/benjamintli/sku110k
       (original: https://github.com/eg4000/SKU110K_CVPR19 - Goldman et al., CVPR 2019)

Cada linha do Parquet tem: image{bytes}, objects{bbox[[x, y, w, h] em pixels], category}.
O script:
  1. baixa os shards escolhidos (se ainda não existirem);
  2. redimensiona cada imagem para no máx. 1024 px no maior lado (treino mais rápido);
  3. converte as caixas para o formato YOLO: "classe xc yc w h" normalizado (0-1);
  4. gera o arquivo sku110k.yaml usado no treino.

Uso:
  python preparar_dataset.py --destino C:/datasets/sku110k --train-shards 4
"""
import argparse
import io
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

BASE_URL = "https://huggingface.co/datasets/benjamintli/sku110k/resolve/main/data/"
SHARDS = {
    "train": [f"train-{i:05d}-of-00019.parquet" for i in range(19)],
    "val": [f"validation-{i:05d}-of-00002.parquet" for i in range(2)],
    "test": [f"test-{i:05d}-of-00007.parquet" for i in range(7)],
}
LADO_MAX = 1024


def baixar(nome: str, pasta: Path) -> Path:
    destino = pasta / nome
    if not destino.exists():
        print(f"Baixando {nome} ...")
        urllib.request.urlretrieve(BASE_URL + nome, destino)
    return destino


def converter_shard(parquet: Path, split: str, raiz: Path, prefixo: str) -> tuple[int, int]:
    pasta_img = raiz / "images" / split
    pasta_lbl = raiz / "labels" / split
    pasta_img.mkdir(parents=True, exist_ok=True)
    pasta_lbl.mkdir(parents=True, exist_ok=True)
    n_img = n_cx = 0
    for lote in pq.ParquetFile(parquet).iter_batches(batch_size=16, columns=["image", "objects"]):
        for linha in lote.to_pylist():
            img = Image.open(io.BytesIO(linha["image"]["bytes"])).convert("RGB")
            w, h = img.size
            esc = min(1.0, LADO_MAX / max(w, h))
            if esc < 1.0:
                img = img.resize((round(w * esc), round(h * esc)), Image.BILINEAR)
            nome = f"{prefixo}_{n_img:05d}"
            img.save(pasta_img / f"{nome}.jpg", quality=90)

            linhas = []
            for x, y, bw, bh in linha["objects"]["bbox"]:
                # recorta à imagem e normaliza (coordenadas originais em pixels)
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(w), x + bw), min(float(h), y + bh)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                linhas.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                              f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (pasta_lbl / f"{nome}.txt").write_text("\n".join(linhas))
            n_img += 1
            n_cx += len(linhas)
    return n_img, n_cx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", required=True, help="pasta do dataset YOLO")
    ap.add_argument("--parquet", default=None, help="pasta dos .parquet (padrão: <destino>/parquet)")
    ap.add_argument("--train-shards", type=int, default=4, help="shards de treino (~430 imagens cada)")
    ap.add_argument("--val-shards", type=int, default=1)
    ap.add_argument("--test-shards", type=int, default=1)
    args = ap.parse_args()

    raiz = Path(args.destino)
    pasta_pq = Path(args.parquet) if args.parquet else raiz / "parquet"
    pasta_pq.mkdir(parents=True, exist_ok=True)

    plano = {"train": args.train_shards, "val": args.val_shards, "test": args.test_shards}
    for split, qtd in plano.items():
        tot_i = tot_c = 0
        for k, nome in enumerate(SHARDS[split][:qtd]):
            pq_path = baixar(nome, pasta_pq)
            ni, nc = converter_shard(pq_path, split, raiz, f"{split}{k}")
            tot_i += ni
            tot_c += nc
            print(f"  {split}: {nome} -> {ni} imagens, {nc} caixas")
        print(f"[{split}] {tot_i} imagens | {tot_c} produtos anotados")

    yaml = raiz / "sku110k.yaml"
    yaml.write_text(
        f"path: {raiz.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        "names:\n  0: produto\n", encoding="utf-8")
    print(f"Dataset pronto: {yaml}")


if __name__ == "__main__":
    main()

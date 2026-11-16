# -*- coding: utf-8 -*-
"""
Gera um VÍDEO DE SIMULAÇÃO a partir de uma foto real de prateleira, para testar
a regra de negócio temporal do main.py (persistência de alertas):

  1. prateleira abastecida                       -> tudo OK
  2. um "cliente" passa na frente (oclusão curta) -> NÃO deve gerar alerta
  3. os produtos de uma zona são retirados um a um (vendas)
                                                  -> ESTOQUE BAIXO -> RUPTURA
  4. o repositor reabastece a zona                -> NORMALIZADO

Os produtos retirados são apagados da foto com inpainting (OpenCV).
Uso:
  python ferramentas/gerar_video_simulacao.py --imagem amostras/x.jpg --zona B2
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from video_h264 import converter_h264

BASE = Path(__file__).resolve().parent.parent


def ler(caminho):
    return cv2.imdecode(np.fromfile(str(caminho), np.uint8), cv2.IMREAD_COLOR)


def desenhar_cliente(img, cx, alt_ini):
    """Silhueta simples de uma pessoa passando em frente à gôndola."""
    out = img.copy()
    h, w = img.shape[:2]
    larg = int(w * 0.16)
    topo = int(h * alt_ini)
    cor_corpo, cor_cabeca = (95, 60, 35), (120, 150, 190)
    cv2.ellipse(out, (cx, topo + larg // 3), (larg // 4, larg // 3), 0, 0, 360, cor_cabeca, -1)
    cv2.rectangle(out, (cx - larg // 2, topo + int(larg * 0.6)), (cx + larg // 2, h), cor_corpo, -1)
    return cv2.addWeighted(out, 0.92, img, 0.08, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagem", required=True)
    ap.add_argument("--pesos", default=str(BASE / "modelo" / "best.pt"))
    ap.add_argument("--linhas", type=int, default=3)
    ap.add_argument("--colunas", type=int, default=4)
    ap.add_argument("--zona", default="B2", help="zona que será esvaziada (ex.: B2)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--largura", type=int, default=1280)
    ap.add_argument("--recorte", default=None, help="enquadramento da gôndola 'x1,y1,x2,y2' (0-1)")
    ap.add_argument("--saida", default=str(BASE / "amostras" / "simulacao_ruptura.mp4"))
    args = ap.parse_args()

    img = ler(args.imagem)
    if args.recorte:  # simula a câmera fixa enquadrando só a gôndola
        x1, y1, x2, y2 = (float(v) for v in args.recorte.split(","))
        H0, W0 = img.shape[:2]
        img = img[int(y1 * H0):int(y2 * H0), int(x1 * W0):int(x2 * W0)]
    esc = args.largura / img.shape[1]
    img = cv2.resize(img, (args.largura, round(img.shape[0] * esc)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    # zona alvo na grade
    lin, col = ord(args.zona[0].upper()) - ord("A"), int(args.zona[1:]) - 1
    zx1, zx2 = col * w / args.colunas, (col + 1) * w / args.colunas
    zy1, zy2 = lin * h / args.linhas, (lin + 1) * h / args.linhas

    # produtos da zona (detectados pelo próprio modelo)
    caixas = YOLO(args.pesos).predict(img, imgsz=800, conf=0.3, max_det=1000,
                                      verbose=False)[0].boxes.xyxy.cpu().numpy()
    cx, cy = (caixas[:, 0] + caixas[:, 2]) / 2, (caixas[:, 1] + caixas[:, 3]) / 2
    alvo = caixas[(cx >= zx1) & (cx < zx2) & (cy >= zy1) & (cy < zy2)]
    alvo = alvo[np.lexsort((alvo[:, 0], -alvo[:, 1]))]  # de baixo p/ cima, esquerda p/ direita
    print(f"{len(alvo)} produtos na zona {args.zona} serão retirados")

    # sequência de imagens com produtos removidos (inpainting cumulativo)
    estados, atual = [img.copy()], img.copy()
    mascara = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in alvo.astype(int):
        mascara[max(0, y1 - 3):y2 + 3, max(0, x1 - 3):x2 + 3] = 255
        atual = cv2.inpaint(img, mascara, 9, cv2.INPAINT_TELEA)
        escuro = (atual * 0.55).astype(np.uint8)  # fundo vazio da prateleira fica sombreado
        atual = np.where(mascara[..., None] > 0, escuro, atual)
        estados.append(atual.copy())

    fps = args.fps
    quadros = []
    quadros += [img] * (3 * fps)                                   # 1) abastecida
    n_cli = int(1.2 * fps)                                         # 2) cliente passa (1,2 s)
    for k in range(n_cli):
        quadros.append(desenhar_cliente(img, int(-0.1 * w + 1.2 * w * k / (n_cli - 1)), 0.25))
    quadros += [img] * int(0.8 * fps)
    passo = max(1, int(0.8 * fps))                                 # 3) retirada (1 produto/0,8 s)
    for est in estados[1:]:
        quadros += [est] * passo
    quadros += [estados[-1]] * (4 * fps)
    for est in estados[::-1][::3]:                                 # 4) reposição rápida
        quadros += [est] * 2
    quadros += [img] * (4 * fps)

    tmp = Path(tempfile.gettempdir()) / f"sim_{os.getpid()}.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for q in quadros:
        vw.write(q)
    vw.release()
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), args.saida)
    print(f"Vídeo de simulação: {args.saida} ({len(quadros)} quadros, {len(quadros) / fps:.1f} s)")
    converter_h264(Path(args.saida))  # opcional: arquivo menor (requer imageio-ffmpeg)


if __name__ == "__main__":
    main()

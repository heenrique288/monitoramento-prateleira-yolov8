#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 ATIVIDADE AVALIATIVA N3 - Tópicos Especiais
 Monitoramento de Prateleira de Supermercado e Alerta de Ruptura com YOLOv8
=============================================================================

Fluxo do script
---------------
 1. Carrega o modelo YOLOv8 treinado para detectar produtos em prateleiras
    (modelo/best.pt, treinado com o dataset SKU-110K).
 2. Lê a fonte: imagem estática, pasta de imagens, vídeo gravado ou webcam.
 3. Detecta todos os produtos de cada imagem/quadro.
 4. Aplica a REGRA DE NEGÓCIO de ruptura:
      - a gôndola é dividida em ZONAS (grade linhas x colunas ou um
        "planograma" em JSON);
      - para cada zona calcula-se a quantidade de produtos e a TAXA DE
        OCUPAÇÃO (fração da área da zona coberta por produtos detectados);
      - cada zona é classificada em:
            OK             nível >= limiar_baixo
            ESTOQUE BAIXO  limiar_ruptura <= nível < limiar_baixo
            RUPTURA        nível <  limiar_ruptura  (ou nenhum produto)
      - em vídeo/webcam a mudança de status só é confirmada se persistir
        por N segundos (evita falso alarme quando um cliente passa na frente
        da prateleira).
 5. Emite alertas no terminal, grava um log CSV de eventos, um resumo JSON e
    a imagem/vídeo anotado com as zonas coloridas por status.

Exemplos de uso
---------------
  python main.py --source amostras/prateleira_cheia.jpg
  python main.py --source amostras/                       (pasta inteira)
  python main.py --source amostras/simulacao_ruptura.mp4
  python main.py --source 0 --show                        (webcam)
  python main.py --source amostras/x.jpg --zonas config_zonas.json
  python main.py --source amostras/x.jpg --linhas 4 --colunas 5 --limiar-baixo 0.6
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PESOS_PADRAO = BASE_DIR / "modelo" / "best.pt"
PESOS_RESERVA = "yolov8n.pt"  # modelo genérico (COCO), usado só se best.pt faltar

EXT_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXT_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Classes do COCO que podem representar produtos (apenas p/ o modelo genérico)
# 39 bottle, 40 wine glass, 41 cup, 45 bowl, 46 banana, 47 apple, 49 orange,
# 50 broccoli, 51 carrot
CLASSES_PRODUTO_COCO = [39, 40, 41, 45, 46, 47, 49, 50, 51]

OK, BAIXO, RUPTURA = "OK", "ESTOQUE BAIXO", "RUPTURA"
GRAVIDADE = {OK: 0, BAIXO: 1, RUPTURA: 2}
ACAO = {
    OK: "Nenhuma - zona abastecida",
    BAIXO: "Programar reposição (estoque baixo)",
    RUPTURA: "Repor IMEDIATAMENTE (produto em falta na gôndola)",
}
COR = {  # RGB
    OK: (46, 160, 67),
    BAIXO: (255, 176, 0),
    RUPTURA: (229, 57, 53),
}
COR_CAIXA_PRODUTO = (0, 229, 255)
LADO_MAX_SAIDA = 1600  # imagens anotadas são reduzidas para no máx. 1600 px


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------
@dataclass
class Zona:
    """Região da gôndola monitorada (coordenadas normalizadas 0..1)."""
    nome: str
    retangulo: tuple[float, float, float, float]  # x1, y1, x2, y2
    capacidade: int | None = None  # nº de frentes esperado (planograma), opcional
    descricao: str = ""            # produto/categoria da zona (planograma), opcional


@dataclass
class Regra:
    """Parâmetros da regra de negócio."""
    # Calibrados no conjunto de teste do SKU-110K: a ocupação mediana de uma
    # zona real é ~54% (bordas, etiquetas e vãos entre produtos nunca ficam
    # cobertos). < 40% ocorre em ~15% das zonas e < 20% em ~2% (vão evidente).
    limiar_baixo: float = 0.40     # abaixo disso -> ESTOQUE BAIXO
    limiar_ruptura: float = 0.20   # abaixo disso -> RUPTURA
    persistencia_seg: float = 0.0  # tempo mínimo para confirmar mudança (vídeo)


@dataclass
class ResultadoZona:
    zona: Zona
    caixa_px: tuple[int, int, int, int]
    qtd_produtos: int
    ocupacao: float          # fração da área coberta por produtos (0..1)
    nivel: float             # métrica usada na decisão (0..1)
    status: str              # status instantâneo (deste quadro)
    status_confirmado: str | None = None
    progresso: float = 0.0   # 0..1 do tempo de confirmação de um novo status

    @property
    def status_exibido(self) -> str:
        return self.status_confirmado or self.status


@dataclass
class Evento:
    data_hora: str
    fonte: str
    quadro: int
    tempo_video_s: float
    zona: str
    produto_zona: str
    evento: str
    status_anterior: str
    status: str
    qtd_produtos: int
    ocupacao_pct: float
    nivel_pct: float
    acao_recomendada: str


# ---------------------------------------------------------------------------
# Zonas (grade automática ou planograma em JSON)
# ---------------------------------------------------------------------------
def criar_grade(linhas: int, colunas: int, roi=(0.0, 0.0, 1.0, 1.0)) -> list[Zona]:
    """Divide a área da gôndola (ROI) em linhas (prateleiras A, B, C...)
    e colunas (seções 1, 2, 3...)."""
    rx1, ry1, rx2, ry2 = roi
    lw, lh = (rx2 - rx1) / colunas, (ry2 - ry1) / linhas
    zonas = []
    for i in range(linhas):
        for j in range(colunas):
            nome = f"{chr(ord('A') + i)}{j + 1}"
            zonas.append(Zona(nome, (rx1 + j * lw, ry1 + i * lh,
                                     rx1 + (j + 1) * lw, ry1 + (i + 1) * lh)))
    return zonas


def carregar_planograma(caminho: str) -> tuple[list[Zona], dict]:
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    zonas = [Zona(z["nome"], tuple(z["retangulo"]), z.get("capacidade"), z.get("descricao", ""))
             for z in dados["zonas"]]
    return zonas, dados


# ---------------------------------------------------------------------------
# REGRA DE NEGÓCIO
# ---------------------------------------------------------------------------
def classificar(nivel: float, qtd: int, regra: Regra) -> str:
    """Classifica o abastecimento de uma zona."""
    if qtd == 0 or nivel < regra.limiar_ruptura:
        return RUPTURA
    if nivel < regra.limiar_baixo:
        return BAIXO
    return OK


def avaliar_prateleira(caixas: np.ndarray, zonas: list[Zona], largura: int,
                       altura: int, regra: Regra) -> list[ResultadoZona]:
    """Para cada zona: conta os produtos (pelo centro da caixa) e mede a taxa
    de ocupação = área da zona coberta pela UNIÃO das caixas detectadas.

    Se a zona tiver 'capacidade' (planograma), o nível de abastecimento é
    qtd/capacidade; caso contrário, o nível é a própria taxa de ocupação."""
    # Máscara binária de ocupação em resolução reduzida (mais rápido)
    escala = min(1.0, 800 / max(largura, altura))
    mw, mh = max(1, round(largura * escala)), max(1, round(altura * escala))
    mascara = np.zeros((mh, mw), dtype=np.uint8)
    for x1, y1, x2, y2 in caixas:
        mascara[int(y1 * escala):int(np.ceil(y2 * escala)),
                int(x1 * escala):int(np.ceil(x2 * escala))] = 1

    if len(caixas):
        cx = (caixas[:, 0] + caixas[:, 2]) / 2
        cy = (caixas[:, 1] + caixas[:, 3]) / 2
    else:
        cx = cy = np.empty(0)

    resultados = []
    for z in zonas:
        x1n, y1n, x2n, y2n = z.retangulo
        px = (int(x1n * largura), int(y1n * altura),
              int(x2n * largura), int(y2n * altura))
        dentro = (cx >= px[0]) & (cx < px[2]) & (cy >= px[1]) & (cy < px[3])
        qtd = int(dentro.sum())
        recorte = mascara[int(y1n * mh):int(y2n * mh), int(x1n * mw):int(x2n * mw)]
        ocupacao = float(recorte.mean()) if recorte.size else 0.0
        nivel = min(1.0, qtd / z.capacidade) if z.capacidade else ocupacao
        resultados.append(ResultadoZona(z, px, qtd, ocupacao, nivel,
                                        classificar(nivel, qtd, regra)))
    return resultados


def resumir(resultados: list[ResultadoZona], n_produtos: int) -> dict:
    """Indicadores da gôndola inteira."""
    status = [r.status_exibido for r in resultados]
    n = len(resultados)
    n_rupt, n_baixo = status.count(RUPTURA), status.count(BAIXO)
    if n_rupt:
        situacao = "CRÍTICA"
    elif n_baixo:
        situacao = "ATENÇÃO"
    else:
        situacao = "NORMAL"
    prioridade = sorted((r for r in resultados if r.status_exibido != OK),
                        key=lambda r: (-GRAVIDADE[r.status_exibido], r.nivel))
    return {
        "situacao": situacao,
        "produtos": n_produtos,
        "zonas": n,
        "zonas_ok": status.count(OK),
        "zonas_baixo": n_baixo,
        "zonas_ruptura": n_rupt,
        "indice_ruptura_pct": round(100 * n_rupt / n, 1) if n else 0.0,
        "ocupacao_media_pct": round(100 * float(np.mean([r.ocupacao for r in resultados])), 1) if n else 0.0,
        "prioridade_reposicao": [r.zona.nome for r in prioridade],
    }


class GerenciadorAlertas:
    """Confirma mudanças de status por zona somente quando persistem por
    'persistencia_seg' segundos (debounce) e registra cada evento no CSV."""

    CAMPOS = list(Evento.__dataclass_fields__.keys())

    def __init__(self, caminho_csv: Path, persistencia_seg: float):
        self.persistencia = max(0.0, persistencia_seg)
        self.confirmado: dict[str, str] = {}
        self.candidato: dict[str, tuple[str, float]] = {}
        self.ignoradas: list[str] = []  # oscilações curtas descartadas (ex.: oclusão)
        novo = not caminho_csv.exists()
        self._arq = open(caminho_csv, "a", newline="", encoding="utf-8-sig")
        self._csv = csv.DictWriter(self._arq, fieldnames=self.CAMPOS, delimiter=";")
        if novo:
            self._csv.writeheader()

    def reiniciar(self, persistencia_seg: float | None = None):
        self.confirmado.clear()
        self.candidato.clear()
        self.ignoradas.clear()
        if persistencia_seg is not None:
            self.persistencia = max(0.0, persistencia_seg)

    def atualizar(self, resultados: list[ResultadoZona], fonte: str,
                  quadro: int, t: float) -> list[Evento]:
        eventos = []
        for r in resultados:
            nome = r.zona.nome
            cand, t0 = self.candidato.get(nome, (None, t))
            anterior = self.confirmado.get(nome)
            if r.status != cand:
                # o status pendente sumiu antes de completar a persistência -> descartado
                if cand is not None and anterior is not None and cand != anterior:
                    msg = (f" [t={t:.1f}s] (ignorado) zona {nome}: {cand} durou só "
                           f"{t - t0:.1f}s < {self.persistencia:.1f}s -> oscilação/oclusão, sem alerta")
                    self.ignoradas.append(msg)
                    print(msg)
                cand, t0 = r.status, t
                self.candidato[nome] = (cand, t0)
            decorrido = t - t0
            r.progresso = 1.0 if self.persistencia == 0 else min(1.0, decorrido / self.persistencia)

            if r.status != anterior and decorrido >= self.persistencia:
                self.confirmado[nome] = r.status
                if not (anterior is None and r.status == OK):  # início normal: sem evento
                    if GRAVIDADE[r.status] > GRAVIDADE.get(anterior or OK, 0):
                        tipo = "ALERTA"
                    elif r.status == OK:
                        tipo = "NORMALIZADO"
                    else:
                        tipo = "REPOSICAO PARCIAL"
                    ev = Evento(
                        data_hora=datetime.now().isoformat(sep=" ", timespec="seconds"),
                        fonte=fonte, quadro=quadro, tempo_video_s=round(t, 2),
                        zona=nome, produto_zona=r.zona.descricao, evento=tipo,
                        status_anterior=anterior or "-",
                        status=r.status, qtd_produtos=r.qtd_produtos,
                        ocupacao_pct=round(100 * r.ocupacao, 1),
                        nivel_pct=round(100 * r.nivel, 1),
                        acao_recomendada=ACAO[r.status],
                    )
                    self._csv.writerow(ev.__dict__)
                    eventos.append(ev)
            r.status_confirmado = self.confirmado.get(nome)
        self._arq.flush()
        return eventos

    def fechar(self):
        self._arq.close()


# ---------------------------------------------------------------------------
# Detecção (YOLOv8)
# ---------------------------------------------------------------------------
def carregar_modelo(caminho: str) -> tuple[YOLO, list[int] | None]:
    if not Path(caminho).exists():
        print(f"[AVISO] Pesos '{caminho}' não encontrados. Usando o modelo "
              f"genérico {PESOS_RESERVA} (COCO) apenas como contingência.")
        caminho = PESOS_RESERVA
    modelo = YOLO(caminho)
    nomes = modelo.names
    eh_coco = len(nomes) == 80 and nomes.get(0) == "person"
    classes = CLASSES_PRODUTO_COCO if eh_coco else None
    print(f"[MODELO] {caminho} | classes: {list(nomes.values())[:6]}"
          f"{' ...' if len(nomes) > 6 else ''}")
    return modelo, classes


def detectar(modelo: YOLO, frame: np.ndarray, args, classes) -> tuple[np.ndarray, float]:
    res = modelo.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                         max_det=args.max_det, classes=classes, verbose=False)[0]
    caixas = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
    return caixas, float(res.speed.get("inference", 0.0))


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def fonte(tamanho: int, negrito: bool = False):
    nomes = (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if negrito
             else ["arial.ttf", "DejaVuSans.ttf"])
    pastas = ["", "C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/",
              "/Library/Fonts/"]
    for nome in nomes:
        for pasta in pastas:
            try:
                return ImageFont.truetype(pasta + nome, tamanho)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=tamanho)
    except TypeError:
        return ImageFont.load_default()


def desenhar(frame_bgr: np.ndarray, caixas: np.ndarray, resultados: list[ResultadoZona],
             resumo: dict, info: str, persistencia: float) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    esc = min(1.0, LADO_MAX_SAIDA / max(w, h))
    if esc < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (round(w * esc), round(h * esc)),
                               interpolation=cv2.INTER_AREA)
    W, H = frame_bgr.shape[1], frame_bgr.shape[0]
    base = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    esp = max(2, round(W / 500))

    # 1) preenchimento translúcido das zonas com problema
    fundo = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(fundo)
    transp = {OK: 0, BAIXO: 38, RUPTURA: 62}
    for r in resultados:
        if transp[r.status_exibido]:
            d.rectangle([v * esc for v in r.caixa_px],
                        fill=COR[r.status_exibido] + (transp[r.status_exibido],))
    img = Image.alpha_composite(base, fundo)

    # 2) caixas dos produtos detectados + 3) contorno das zonas
    linhas = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(linhas)
    for x1, y1, x2, y2 in caixas * esc:
        d.rectangle([x1, y1, x2, y2], outline=COR_CAIXA_PRODUTO + (210,),
                    width=max(1, esp // 2))
    for r in resultados:
        d.rectangle([v * esc for v in r.caixa_px], outline=COR[r.status_exibido] + (255,), width=esp)
    img = Image.alpha_composite(img, linhas)
    d = ImageDraw.Draw(img)

    # 4) rótulo de cada zona
    fs = max(12, round(W * 0.0115))
    f_txt, f_neg = fonte(fs), fonte(fs, True)
    for r in resultados:
        x1, y1 = r.caixa_px[0] * esc + esp, r.caixa_px[1] * esc + esp
        l1 = f"{r.zona.nome}  {r.qtd_produtos} un  {r.ocupacao:.0%}"
        if r.zona.capacidade:
            l1 += f"  ({r.qtd_produtos}/{r.zona.capacidade})"
        l2 = r.status_exibido + (f" · {r.zona.descricao}" if r.zona.descricao else "")
        if r.status_confirmado and r.status != r.status_confirmado and persistencia > 0:
            l2 += f"  > {r.status}? {r.progresso * persistencia:.1f}/{persistencia:.0f}s"
        larg = max(d.textlength(l1, font=f_txt), d.textlength(l2, font=f_neg)) + 12
        d.rectangle([x1, y1, x1 + larg, y1 + 2 * fs + 14], fill=(20, 20, 20, 255))
        d.rectangle([x1, y1, x1 + 6, y1 + 2 * fs + 14], fill=COR[r.status_exibido])
        d.text((x1 + 10, y1 + 3), l1, font=f_txt, fill=(255, 255, 255))
        d.text((x1 + 10, y1 + fs + 8), l2, font=f_neg, fill=COR[r.status_exibido])

    # 5) painel de cabeçalho (fora da imagem, para não cobrir produtos)
    fh = max(16, round(W * 0.016))
    f_tit, f_lin, f_rep, f_info = fonte(fh, True), fonte(round(fh * 0.72)), \
        fonte(round(fh * 0.8), True), fonte(round(fh * 0.65))
    linha = (f"Produtos: {resumo['produtos']}   |   Ocupação média: {resumo['ocupacao_media_pct']:.0f}%"
             f"   |   Zonas: {resumo['zonas_ok']} OK · {resumo['zonas_baixo']} baixo · "
             f"{resumo['zonas_ruptura']} ruptura   |   Índice de ruptura: {resumo['indice_ruptura_pct']:.0f}%")
    medir = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lw, iw = medir.textlength(linha, font=f_lin), medir.textlength(info, font=f_info)
    info_embaixo = lw + iw + 40 > W  # sem espaço: info vai para uma 3ª linha
    alt_cab = round(fh * (4.6 if info_embaixo else 3.6))
    tela = Image.new("RGB", (W, H + alt_cab), (24, 26, 31))
    tela.paste(img.convert("RGB"), (0, alt_cab))
    d = ImageDraw.Draw(tela)
    cor_sit = {"NORMAL": COR[OK], "ATENÇÃO": COR[BAIXO], "CRÍTICA": COR[RUPTURA]}[resumo["situacao"]]
    titulo = f"GÔNDOLA: {resumo['situacao']}"
    tw = d.textlength(titulo, font=f_tit)
    d.rectangle([10, 8, 30 + tw, 14 + fh + 8], fill=cor_sit)
    d.text((20, 11), titulo, font=f_tit, fill=(255, 255, 255))
    d.text((10, 14 + fh + 14), linha, font=f_lin, fill=(230, 230, 230))
    if resumo["prioridade_reposicao"]:
        rep = "Repor: " + ", ".join(resumo["prioridade_reposicao"][:8])
        d.text((44 + tw, 13), rep, font=f_rep, fill=cor_sit)
    y_info = 14 + fh + 16 + (round(fh * 1.0) if info_embaixo else 0)
    x_info = 10 if info_embaixo else W - iw - 10
    d.text((x_info, y_info), info, font=f_info, fill=(150, 150, 150))
    return cv2.cvtColor(np.asarray(tela), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Saída no terminal
# ---------------------------------------------------------------------------
def imprimir_relatorio(nome: str, resultados: list[ResultadoZona], resumo: dict, ms: float):
    print("\n" + "=" * 70)
    print(f" MONITORAMENTO DE PRATELEIRA  |  {nome}")
    print("=" * 70)
    print(f" Produtos detectados: {resumo['produtos']:<6} Inferência YOLOv8: {ms:.1f} ms")
    print(f" {'Zona':<8}{'Produtos':>9}{'Ocupação':>10}{'Nível':>8}   Status")
    print(" " + "-" * 50)
    for r in resultados:
        marca = {OK: "  ", BAIXO: " !", RUPTURA: "!!"}[r.status_exibido]
        desc = f"  ({r.zona.descricao})" if r.zona.descricao else ""
        print(f" {r.zona.nome:<8}{r.qtd_produtos:>9}{r.ocupacao:>10.0%}{r.nivel:>8.0%}   "
              f"{marca} {r.status_exibido}{desc}")
    print(" " + "-" * 50)
    print(f" Situação da gôndola: {resumo['situacao']} | {resumo['zonas_ruptura']} zona(s) em "
          f"ruptura, {resumo['zonas_baixo']} com estoque baixo | índice de ruptura "
          f"{resumo['indice_ruptura_pct']:.0f}%")
    if resumo["prioridade_reposicao"]:
        print(f" Prioridade de reposição: {' > '.join(resumo['prioridade_reposicao'])}")


def imprimir_evento(ev: Evento):
    marca = {"ALERTA": "!! ALERTA", "NORMALIZADO": "OK NORMALIZADO",
             "REPOSICAO PARCIAL": "-> REPOSIÇÃO PARCIAL"}[ev.evento]
    tempo = f" t={ev.tempo_video_s:.1f}s" if ev.quadro > 0 else ""
    zona = f"{ev.zona} ({ev.produto_zona})" if ev.produto_zona else ev.zona
    print(f" [{ev.data_hora}{tempo}] {marca} | zona {zona}: {ev.status_anterior} -> "
          f"{ev.status} | {ev.qtd_produtos} produto(s), ocupação {ev.ocupacao_pct:.0f}% "
          f"| Ação: {ev.acao_recomendada}")


# ---------------------------------------------------------------------------
# Utilitários de arquivo (OpenCV não aceita caminhos com acento no Windows)
# ---------------------------------------------------------------------------
def ler_imagem(caminho: Path) -> np.ndarray | None:
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_COLOR) if dados.size else None


def salvar_imagem(caminho: Path, img: np.ndarray):
    ok, buf = cv2.imencode(caminho.suffix or ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(str(caminho))


def caminho_ascii(caminho: Path) -> tuple[str, Path | None]:
    """Devolve um caminho que o OpenCV consiga abrir (copia p/ pasta temporária se preciso)."""
    s = str(caminho)
    if s.isascii():
        return s, None
    rel = os.path.relpath(s)
    if rel.isascii():
        return rel, None
    tmp = Path(tempfile.gettempdir()) / f"prateleira_{os.getpid()}{caminho.suffix}"
    shutil.copy(caminho, tmp)
    return str(tmp), tmp


# ---------------------------------------------------------------------------
# Processamento das fontes
# ---------------------------------------------------------------------------
def processar_imagem(caminho: Path, modelo, classes, zonas, regra, alertas, args, saida: Path) -> dict:
    frame = ler_imagem(caminho)
    if frame is None:
        print(f"[ERRO] Não foi possível ler {caminho}")
        return {}
    h, w = frame.shape[:2]
    caixas, ms = detectar(modelo, frame, args, classes)
    resultados = avaliar_prateleira(caixas, zonas, w, h, regra)
    alertas.reiniciar(0.0)  # imagem estática: confirmação imediata
    eventos = alertas.atualizar(resultados, caminho.name, 0, 0.0)
    resumo = resumir(resultados, len(caixas))
    imprimir_relatorio(caminho.name, resultados, resumo, ms)
    for ev in eventos:
        imprimir_evento(ev)

    info = f"{caminho.name}  ·  {datetime.now():%d/%m/%Y %H:%M:%S}  ·  YOLOv8 {ms:.0f} ms"
    anotada = desenhar(frame, caixas, resultados, resumo, info, 0.0)
    destino = saida / f"{caminho.stem}_resultado.jpg"
    salvar_imagem(destino, anotada)
    print(f" Imagem anotada salva em: {destino}")
    return {"fonte": caminho.name, "tipo": "imagem", "inferencia_ms": round(ms, 1), **resumo,
            "detalhe_zonas": [_zona_json(r) for r in resultados]}


def processar_video(fonte_v, modelo, classes, zonas, regra, alertas, args, saida: Path) -> dict:
    webcam = isinstance(fonte_v, int)
    nome = f"webcam_{fonte_v}" if webcam else Path(fonte_v).name
    entrada, tmp_in = (fonte_v, None) if webcam else caminho_ascii(Path(fonte_v))
    cap = cv2.VideoCapture(entrada)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a fonte de vídeo: {fonte_v}")
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 120:
        fps = 30.0
    alertas.reiniciar(regra.persistencia_seg)
    print(f"\n[VÍDEO] {nome} | {fps:.0f} FPS | persistência p/ confirmar alerta: "
          f"{regra.persistencia_seg:.1f} s")

    tmp_out = Path(tempfile.gettempdir()) / f"prateleira_saida_{os.getpid()}.mp4"
    destino = saida / f"{Path(nome).stem}_resultado.mp4"
    writer, quadro, t_ini = None, 0, time.perf_counter()
    todos_eventos, tempos, ultimo = [], [], None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            quadro += 1
            t = (time.perf_counter() - t_ini) if webcam else quadro / fps
            h, w = frame.shape[:2]
            caixas, ms = detectar(modelo, frame, args, classes)
            tempos.append(ms)
            resultados = avaliar_prateleira(caixas, zonas, w, h, regra)
            eventos = alertas.atualizar(resultados, nome, quadro, t)
            for ev in eventos:
                imprimir_evento(ev)
            todos_eventos += eventos
            resumo = resumir(resultados, len(caixas))
            info = f"{nome}  ·  quadro {quadro}  ·  t={t:.1f}s  ·  YOLOv8 {ms:.0f} ms"
            anotado = desenhar(frame, caixas, resultados, resumo, info, regra.persistencia_seg)
            if writer is None:
                writer = cv2.VideoWriter(str(tmp_out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (anotado.shape[1], anotado.shape[0]))
            writer.write(anotado)
            ultimo = (resultados, resumo, anotado)
            if args.show:
                cv2.imshow("Monitoramento de Prateleira - YOLOv8 (q para sair)", anotado)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            shutil.move(str(tmp_out), destino)
        if tmp_in is not None:
            tmp_in.unlink(missing_ok=True)
        if args.show:
            cv2.destroyAllWindows()

    if ultimo is None:
        return {}
    resultados, resumo, anotado = ultimo
    imprimir_relatorio(f"{nome} (último quadro, {quadro} quadros)", resultados, resumo,
                       float(np.mean(tempos)))
    salvar_imagem(saida / f"{Path(nome).stem}_ultimo_quadro.jpg", anotado)
    print(f" Vídeo anotado salvo em: {destino}")
    print(f" Eventos registrados: {len(todos_eventos)} "
          f"({sum(e.evento == 'ALERTA' for e in todos_eventos)} alerta(s)) | "
          f"oscilações ignoradas pela persistência: {len(alertas.ignoradas)}")
    return {"fonte": nome, "tipo": "webcam" if webcam else "video", "quadros": quadro,
            "fps": round(fps, 1), "inferencia_media_ms": round(float(np.mean(tempos)), 1),
            "eventos": [e.__dict__ for e in todos_eventos],
            "oscilacoes_ignoradas": [m.strip() for m in alertas.ignoradas], "estado_final": resumo,
            "detalhe_zonas_final": [_zona_json(r) for r in resultados]}


def _zona_json(r: ResultadoZona) -> dict:
    return {"zona": r.zona.nome, "descricao": r.zona.descricao, "produtos": r.qtd_produtos,
            "ocupacao_pct": round(100 * r.ocupacao, 1), "nivel_pct": round(100 * r.nivel, 1),
            "capacidade": r.zona.capacidade, "status": r.status_exibido}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def argumentos():
    p = argparse.ArgumentParser(
        description="Monitoramento de prateleira e alerta de ruptura com YOLOv8.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", required=True,
                   help="imagem, pasta de imagens, vídeo ou índice da webcam (ex.: 0)")
    p.add_argument("--weights", default=str(PESOS_PADRAO), help="pesos YOLOv8 (.pt)")
    p.add_argument("--conf", type=float, default=0.35, help="confiança mínima")
    p.add_argument("--iou", type=float, default=0.6, help="IoU do NMS")
    p.add_argument("--imgsz", type=int, default=800, help="tamanho de entrada da rede")
    p.add_argument("--max-det", type=int, default=1000, help="máx. de detecções por imagem")
    p.add_argument("--linhas", type=int, default=3, help="nº de prateleiras (linhas da grade)")
    p.add_argument("--colunas", type=int, default=4, help="nº de seções (colunas da grade)")
    p.add_argument("--roi", default=None, help="área da gôndola 'x1,y1,x2,y2' normalizada (0-1)")
    p.add_argument("--zonas", default=None, help="planograma em JSON (substitui a grade)")
    p.add_argument("--limiar-baixo", type=float, default=None, help="nível p/ ESTOQUE BAIXO (padrão 0.40)")
    p.add_argument("--limiar-ruptura", type=float, default=None, help="nível p/ RUPTURA (padrão 0.20)")
    p.add_argument("--persistencia", type=float, default=2.0,
                   help="segundos que a condição deve persistir p/ confirmar alerta (vídeo/webcam)")
    p.add_argument("--saida", default="resultados", help="pasta de saída")
    p.add_argument("--show", action="store_true", help="exibir janela em tempo real (vídeo/webcam)")
    return p.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = argumentos()
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    # Zonas e limiares: CLI > JSON do planograma > padrão
    regra = Regra(persistencia_seg=args.persistencia)
    if args.zonas:
        zonas, cfg = carregar_planograma(args.zonas)
        regra.limiar_baixo = cfg.get("limiar_baixo", regra.limiar_baixo)
        regra.limiar_ruptura = cfg.get("limiar_ruptura", regra.limiar_ruptura)
        print(f"[ZONAS] Planograma '{cfg.get('descricao', args.zonas)}' com {len(zonas)} zonas")
    else:
        roi = tuple(float(v) for v in args.roi.split(",")) if args.roi else (0.0, 0.0, 1.0, 1.0)
        zonas = criar_grade(args.linhas, args.colunas, roi)
        print(f"[ZONAS] Grade {args.linhas} prateleiras x {args.colunas} seções = {len(zonas)} zonas")
    if args.limiar_baixo is not None:
        regra.limiar_baixo = args.limiar_baixo
    if args.limiar_ruptura is not None:
        regra.limiar_ruptura = args.limiar_ruptura
    print(f"[REGRA] OK >= {regra.limiar_baixo:.0%} | ESTOQUE BAIXO >= {regra.limiar_ruptura:.0%}"
          f" | RUPTURA < {regra.limiar_ruptura:.0%} (ou zona sem produtos)")

    modelo, classes = carregar_modelo(args.weights)
    alertas = GerenciadorAlertas(saida / "log_alertas.csv", regra.persistencia_seg)

    relatorio = {"execucao": datetime.now().isoformat(timespec="seconds"),
                 "pesos": args.weights, "regra": regra.__dict__, "resultados": []}
    try:
        if args.source.isdigit():
            relatorio["resultados"].append(
                processar_video(int(args.source), modelo, classes, zonas, regra, alertas, args, saida))
        else:
            fonte_p = Path(args.source)
            if fonte_p.is_dir():
                arquivos = sorted(f for f in fonte_p.iterdir() if f.suffix.lower() in EXT_IMAGEM | EXT_VIDEO)
            elif fonte_p.exists():
                arquivos = [fonte_p]
            else:
                sys.exit(f"[ERRO] Fonte não encontrada: {fonte_p}")
            for arq in arquivos:
                if arq.suffix.lower() in EXT_VIDEO:
                    r = processar_video(arq, modelo, classes, zonas, regra, alertas, args, saida)
                else:
                    r = processar_imagem(arq, modelo, classes, zonas, regra, alertas, args, saida)
                relatorio["resultados"].append(r)
    finally:
        alertas.fechar()

    (saida / "resumo_execucao.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[FIM] Log de alertas: {saida / 'log_alertas.csv'} | Resumo: {saida / 'resumo_execucao.json'}")


if __name__ == "__main__":
    main()

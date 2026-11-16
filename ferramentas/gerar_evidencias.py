# -*- coding: utf-8 -*-
"""
Gera TODAS as evidências do relatório de forma reproduzível:
  - executa o main.py em cada cenário e salva a saída do terminal (.txt + print .png);
  - gera o vídeo de simulação e o processa com o main.py;
  - monta comparativos (modelo genérico COCO x treinado; sem ROI x com ROI);
  - extrai quadros-chave do vídeo anotado.

Uso (na pasta do projeto):  python ferramentas/gerar_evidencias.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ferramentas"))
from print_terminal import fonte, renderizar  # noqa: E402
from video_h264 import converter_h264  # noqa: E402

EVID = BASE / "evidencias"
AMOS = BASE / "amostras"
PY = sys.executable
TMP = Path(tempfile.gettempdir()) / "evidencias_prateleira"

CENARIOS = [
    ("01_abastecida", ["--source", "amostras/prateleira_01_abastecida.jpg", "--roi", "0,0.04,1,0.92"]),
    ("02_com_rupturas", ["--source", "amostras/prateleira_02_com_rupturas.jpg"]),
    ("03_enquadramento_roi", ["--source", "amostras/prateleira_03_enquadramento.jpg", "--roi", "0,0.16,0.83,0.78"]),
    ("04_planograma", ["--source", "amostras/prateleira_04_planograma.jpg", "--zonas", "config_zonas.json"]),
    ("05_video_simulacao", ["--source", "amostras/simulacao_ruptura.mp4"]),
]


def rodar(args: list[str], saida_dir: Path, cwd: Path = BASE) -> str:
    cmd = [PY, str(BASE / "main.py"), *args, "--saida", str(saida_dir)]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)
    return r.stdout


def ler(p: Path):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def salvar(p: Path, img):
    cv2.imencode(p.suffix, img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(p))


def lado_a_lado(imgs, titulos, destino: Path, altura=1000):
    """Junta imagens lado a lado com um título sobre cada uma."""
    partes = []
    for img, tit in zip(imgs, titulos):
        h, w = img.shape[:2]
        img = cv2.resize(img, (round(w * altura / h), altura), interpolation=cv2.INTER_AREA)
        pil = Image.new("RGB", (img.shape[1], altura + 56), (255, 255, 255))
        pil.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (0, 56))
        ImageDraw.Draw(pil).text((12, 14), tit, font=fonte(26, True), fill=(20, 20, 20))
        partes.append(np.asarray(pil)[..., ::-1])
        partes.append(np.full((altura + 56, 16, 3), 255, np.uint8))
    salvar(destino, np.hstack(partes[:-1]))


def quadros_chave(video: Path, tempos: list[tuple[float, str]], destino: Path, fps=15):
    copia = TMP / "video_tmp.mp4"  # OpenCV não abre caminhos com acento no Windows
    shutil.copy(video, copia)
    cap = cv2.VideoCapture(str(copia))
    alvo ={round(t * fps): tit for t, tit in tempos}
    achados, i = {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        if i in alvo:
            achados[i] = f
    cap.release()
    imgs = [achados[k] for k in sorted(achados)]
    lado_a_lado(imgs, [alvo[k] for k in sorted(achados)], destino, altura=900)


def main():
    EVID.mkdir(exist_ok=True)
    for f in EVID.iterdir():
        f.unlink()
    TMP.mkdir(exist_ok=True)

    # vídeo de simulação (a partir da foto abastecida, recortada como câmera fixa)
    video = AMOS / "simulacao_ruptura.mp4"
    if not video.exists():
        subprocess.run([PY, str(BASE / "ferramentas" / "gerar_video_simulacao.py"),
                        "--imagem", str(AMOS / "prateleira_01_abastecida.jpg"),
                        "--recorte", "0,0.04,1,0.92", "--zona", "B2", "--largura", "1200",
                        "--saida", str(video)], check=True, cwd=BASE)

    for nome, args in CENARIOS:
        print("Executando cenário", nome)
        texto = rodar(args, EVID)
        (EVID / f"saida_{nome}.txt").write_text(texto, encoding="utf-8")
        (EVID / "resumo_execucao.json").replace(EVID / f"resumo_{nome}.json")
        cmd = "python main.py " + " ".join(args)
        linhas = [l for l in texto.splitlines() if not l.startswith("[FIM]")]
        renderizar(linhas, EVID / f"print_terminal_{nome}.png", cmd)

    # comparativo 1: modelo genérico COCO x modelo treinado
    sem = TMP / "coco"
    texto = rodar(["--source", str(AMOS / "prateleira_01_abastecida.jpg"), "--roi", "0,0.04,1,0.92",
                   "--weights", "yolov8n.pt"], sem, cwd=TMP)
    n_coco = json.loads((sem / "resumo_execucao.json").read_text(encoding="utf-8"))["resultados"][0]["produtos"]
    n_trein = json.loads((EVID / "resumo_01_abastecida.json").read_text(encoding="utf-8"))["resultados"][0]["produtos"]
    lado_a_lado([ler(sem / "prateleira_01_abastecida_resultado.jpg"),
                 ler(EVID / "prateleira_01_abastecida_resultado.jpg")],
                [f"YOLOv8n genérico (COCO): {n_coco} produtos",
                 f"YOLOv8s treinado (SKU-110K): {n_trein} produtos"],
                EVID / "comparativo_coco_x_treinado.jpg")

    # comparativo 2: sem ROI x com ROI
    sem = TMP / "sem_roi"
    rodar(["--source", str(AMOS / "prateleira_03_enquadramento.jpg")], sem)
    lado_a_lado([ler(sem / "prateleira_03_enquadramento_resultado.jpg"),
                 ler(EVID / "prateleira_03_enquadramento_resultado.jpg")],
                ["Sem ROI: parede e chão viram 'ruptura'", "Com ROI (só a gôndola)"],
                EVID / "comparativo_sem_roi_x_com_roi.jpg")

    # quadros-chave do vídeo anotado
    rv = json.loads((EVID / "resumo_05_video_simulacao.json").read_text(encoding="utf-8"))["resultados"][0]
    tempos = [(3.6, "1) Cliente passa: sem alerta")]
    for ev in rv["eventos"]:
        rot = {"ALERTA": f"ALERTA {ev['status']}", "NORMALIZADO": "Reposição: NORMALIZADO"}.get(ev["evento"], ev["evento"])
        tempos.append((ev["tempo_video_s"] + 0.3, f"{len(tempos) + 1}) t={ev['tempo_video_s']:.1f}s {rot}"))
    quadros_chave(EVID / "simulacao_ruptura_resultado.mp4", tempos, EVID / "video_quadros_chave.jpg")

    # vídeo anotado em H.264 (menor; opcional) e limpeza dos arquivos auxiliares
    converter_h264(EVID / "simulacao_ruptura_resultado.mp4")
    shutil.rmtree(TMP, ignore_errors=True)
    print("Evidências geradas em", EVID)


if __name__ == "__main__":
    main()

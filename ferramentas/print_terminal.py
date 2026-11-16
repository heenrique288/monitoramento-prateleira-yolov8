# -*- coding: utf-8 -*-
"""
Converte a saída capturada do terminal (texto) em uma imagem PNG no estilo de
janela de terminal, para usar como "print" no relatório.

Uso:
  python main.py --source amostras/x.jpg > evidencias/saida.txt
  python ferramentas/print_terminal.py evidencias/saida.txt evidencias/print.png --comando "python main.py --source amostras/x.jpg"
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def fonte(tam, negrito=False):
    for nome in (["consolab.ttf", "DejaVuSansMono-Bold.ttf"] if negrito else
                 ["consola.ttf", "DejaVuSansMono.ttf"]):
        for pasta in ["", "C:/Windows/Fonts/", "/usr/share/fonts/truetype/dejavu/"]:
            try:
                return ImageFont.truetype(pasta + nome, tam)
            except OSError:
                pass
    return ImageFont.load_default()


def cor_da_linha(l: str):
    if "!! ALERTA" in l:
        return (255, 110, 100) if "-> RUPTURA" in l else (255, 196, 70)
    if "RUPTURA" in l and "!!" in l:
        return (255, 110, 100)
    if "ESTOQUE BAIXO" in l and " ! " in l:
        return (255, 196, 70)
    if "NORMALIZADO" in l or "REPOSIÇÃO" in l:
        return (110, 220, 130)
    if "(ignorado)" in l:
        return (150, 150, 150)
    if l.startswith("[") or l.strip().startswith("=") or "MONITORAMENTO" in l:
        return (120, 180, 255)
    return (220, 220, 220)


def renderizar(linhas: list[str], saida: Path, comando: str, tam: int = 16):
    linhas = [f"PS C:\\...\\N3_Monitoramento_Prateleira_YOLOv8> {comando}"] + linhas
    f, fb = fonte(tam), fonte(tam, True)
    alt_l = tam + 6
    larg = max(900, int(max(f.getlength(l) for l in linhas)) + 40)
    alt = 40 + alt_l * len(linhas) + 16
    img = Image.new("RGB", (larg, alt), (12, 12, 16))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, larg, 30], fill=(40, 40, 46))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([12 + i * 22, 9, 24 + i * 22, 21], fill=c)
    d.text((90, 6), "Windows PowerShell", font=fonte(15), fill=(200, 200, 200))
    y = 40
    for i, l in enumerate(linhas):
        d.text((16, y), l, font=fb if i == 0 else f,
               fill=(250, 250, 120) if i == 0 else cor_da_linha(l))
        y += alt_l
    img.save(saida)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("saida")
    ap.add_argument("--comando", default="python main.py")
    ap.add_argument("--max-linhas", type=int, default=80)
    a = ap.parse_args()
    linhas = Path(a.entrada).read_text(encoding="utf-8").splitlines()[: a.max_linhas]
    renderizar(linhas, Path(a.saida), a.comando)
    print("ok", a.saida)


if __name__ == "__main__":
    main()

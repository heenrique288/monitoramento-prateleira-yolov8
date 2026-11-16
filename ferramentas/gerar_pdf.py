# -*- coding: utf-8 -*-
"""
Gera o relatório técnico em PDF a partir do README.md (Markdown -> HTML -> PDF).

Uso (na pasta do projeto):
  python ferramentas/gerar_pdf.py
Requer: pip install fpdf2 markdown
"""
import re
import shutil
import tempfile
from pathlib import Path

import markdown
from fpdf import FPDF
from fpdf.fonts import FontFace
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
FONTES = Path("C:/Windows/Fonts")


class Relatorio(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", size=8)
        self.set_text_color(120)
        self.cell(0, 8, f"Monitoramento de Prateleira e Alerta de Ruptura com YOLOv8  |  página {self.page_no()}",
                  align="C")


def main():
    md = (BASE / "README.md").read_text(encoding="utf-8")
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    html = markdown.markdown(md, extensions=["tables", "fenced_code"])
    # imagens: paisagem ocupa a largura útil (174 mm); retrato limitado a 150 mm de altura
    tmp = Path(tempfile.mkdtemp(prefix="pdf_imgs_"))

    def img(m):
        caminho = BASE / m.group(2)
        with Image.open(caminho) as im:
            w, h = im.size
            if caminho.stat().st_size > 250_000:  # comprime imagens grandes p/ o PDF ficar leve
                im = im.convert("RGB")
                im.thumbnail((1300, 1300), Image.LANCZOS)
                caminho = tmp / (caminho.stem + ".jpg")
                im.save(caminho, quality=76, optimize=True)
        larg_mm = min(174.0, 150.0 * w / h)
        return f'<img src="{caminho.as_posix()}" width="{larg_mm * 72 / 25.4:.0f}"'
    html = re.sub(r'<img alt="([^"]*)" src="([^"]+)"', img, html)
    html = re.sub(r"<p>(<img [^>]+>)</p>", r'<p align="center">\1</p>', html)
    html = html.replace("<hr />", "<br>")
    # fpdf2 só aceita texto puro dentro de células de tabela: remove tags internas
    html = re.sub(r"<(td|th)(\s[^>]*)?>(.*?)</\1>",
                  lambda m: f"<{m.group(1)}{m.group(2) or ''}>{re.sub(r'<[^>]+>', '', m.group(3))}</{m.group(1)}>",
                  html, flags=re.S)
    # larguras das colunas proporcionais ao conteúdo (raiz do tamanho, com piso)
    def larguras(m):
        tabela = m.group(0)
        linhas = re.findall(r"<tr>(.*?)</tr>", tabela, flags=re.S)
        celulas = [re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", l, flags=re.S) for l in linhas]
        n = len(celulas[0])
        peso = [max(len(c[i]) if i < len(c) else 0 for c in celulas) ** 0.6 + 2 for i in range(n)]
        pct = [100 * p / sum(peso) for p in peso]
        primeira = linhas[0]
        nova, k = primeira, iter(pct)
        nova = re.sub(r"<th(\s[^>]*)?>", lambda _: f'<th width="{next(k):.1f}%">', primeira)
        return tabela.replace(primeira, nova, 1)
    html = re.sub(r"<table>.*?</table>", larguras, html, flags=re.S)
    # código inline (fora de <pre>) em fonte monoespaçada
    partes = re.split(r"(<pre>.*?</pre>)", html, flags=re.S)
    html = "".join(p if p.startswith("<pre>") else
                   re.sub(r"<code>(.*?)</code>", r'<font face="Consolas">\1</font>', p, flags=re.S)
                   for p in partes)

    pdf = Relatorio(format="A4")
    pdf.set_margins(18, 15, 18)
    pdf.set_auto_page_break(True, 16)
    pdf.add_font("Arial", "", FONTES / "arial.ttf")
    pdf.add_font("Arial", "B", FONTES / "arialbd.ttf")
    pdf.add_font("Arial", "I", FONTES / "ariali.ttf")
    pdf.add_font("Arial", "BI", FONTES / "arialbi.ttf")
    pdf.add_font("Consolas", "", FONTES / "consola.ttf")
    pdf.add_font("Consolas", "B", FONTES / "consolab.ttf")
    pdf.set_font("Arial", size=10)
    pdf.add_page()
    pdf.write_html(
        html,
        font_family="Arial",
        table_line_separators=True,
        tag_styles={
            "h1": FontFace(color=(25, 60, 120), size_pt=17, emphasis="BOLD"),
            "h2": FontFace(color=(25, 60, 120), size_pt=13.5, emphasis="BOLD"),
            "h3": FontFace(color=(40, 40, 40), size_pt=11.5, emphasis="BOLD"),
            "pre": FontFace(family="Consolas", size_pt=8.2, color=(30, 30, 30), fill_color=(242, 242, 242)),
            "code": FontFace(family="Consolas", size_pt=8.2),
        },
    )
    destino = BASE / "Relatorio_Tecnico_N3.pdf"
    pdf.output(str(destino))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"PDF gerado: {destino} ({destino.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Converte um .mp4 (codec mp4v gerado pelo OpenCV) para H.264, que fica bem menor
e abre em qualquer navegador/player. Opcional: requer `pip install imageio-ffmpeg`.

Uso:  python ferramentas/video_h264.py arquivo.mp4
"""
import subprocess
import sys
from pathlib import Path


def converter_h264(caminho: Path, crf: int = 26) -> bool:
    try:
        import imageio_ffmpeg
    except ImportError:
        print("[AVISO] imageio-ffmpeg não instalado; vídeo mantido em mp4v.")
        return False
    caminho = Path(caminho)
    tmp = caminho.with_name(caminho.stem + "_h264_tmp.mp4")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(caminho),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "slow",
           "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    if subprocess.run(cmd).returncode != 0 or not tmp.exists():
        print("[AVISO] Falha na conversão para H.264; vídeo mantido em mp4v.")
        tmp.unlink(missing_ok=True)
        return False
    antes = caminho.stat().st_size
    tmp.replace(caminho)
    print(f"Vídeo convertido para H.264: {caminho.name} ({antes / 1e6:.1f} MB -> {caminho.stat().st_size / 1e6:.1f} MB)")
    return True


if __name__ == "__main__":
    for arq in sys.argv[1:]:
        converter_h264(Path(arq))

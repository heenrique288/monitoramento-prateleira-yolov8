# -*- coding: utf-8 -*-
"""
Treino (fine-tuning) do YOLOv8 para detecção de produtos em prateleiras.

Parte do modelo YOLOv8s pré-treinado no COCO (transfer learning) e ajusta-o
ao subconjunto do SKU-110K preparado por preparar_dataset.py.
Ao final copia best.pt e os gráficos de desempenho para ../modelo/.

Uso:
  python treinar.py --data C:/datasets/sku110k/sku110k.yaml --epochs 40 --imgsz 640 --batch 8
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

PASTA_MODELO = Path(__file__).resolve().parent.parent / "modelo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="arquivo .yaml do dataset")
    ap.add_argument("--modelo", default="yolov8s.pt", help="pesos iniciais (pré-treinados COCO)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="runs_treino")
    ap.add_argument("--name", default="sku110k_yolov8s")
    args = ap.parse_args()

    modelo = YOLO(args.modelo)
    modelo.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        max_det=1000,      # prateleiras têm centenas de produtos por imagem
        cos_lr=True,
        close_mosaic=5,
        patience=15,
        seed=42,
        plots=True,
    )

    # Avaliação final no conjunto de TESTE (imagens nunca vistas no treino)
    pasta_run = Path(modelo.trainer.save_dir)
    best = pasta_run / "weights" / "best.pt"
    metricas = YOLO(best).val(data=args.data, split="test", imgsz=args.imgsz,
                              max_det=1000, batch=args.batch, plots=True,
                              project=args.project, name=args.name + "_teste", exist_ok=True)
    print(f"[TESTE] mAP50={metricas.box.map50:.3f} | mAP50-95={metricas.box.map:.3f} | "
          f"Precisão={metricas.box.mp:.3f} | Recall={metricas.box.mr:.3f}")

    # Copia os artefatos para a pasta de entrega
    PASTA_MODELO.mkdir(exist_ok=True)
    shutil.copy(best, PASTA_MODELO / "best.pt")
    graficos = PASTA_MODELO / "graficos_treino"
    graficos.mkdir(exist_ok=True)
    for arq in ["results.png", "results.csv", "BoxPR_curve.png", "BoxF1_curve.png",
                "PR_curve.png", "F1_curve.png", "val_batch0_pred.jpg", "val_batch0_labels.jpg",
                "labels.jpg", "args.yaml"]:
        if (pasta_run / arq).exists():
            shutil.copy(pasta_run / arq, graficos / arq)
    (graficos / "metricas_teste.txt").write_text(
        f"mAP50={metricas.box.map50:.4f}\nmAP50-95={metricas.box.map:.4f}\n"
        f"precisao={metricas.box.mp:.4f}\nrecall={metricas.box.mr:.4f}\n", encoding="utf-8")
    print(f"best.pt copiado para {PASTA_MODELO / 'best.pt'}")


if __name__ == "__main__":
    main()

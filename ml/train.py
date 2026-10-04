"""
train.py
--------
Entrena un modelo YOLOv8n sobre el dataset preparado.

Métricas que se registran (runs/detect/<name>/results.csv + gráficas PNG):
  - Loss de train y val por época: box_loss, cls_loss, dfl_loss
  - precision, recall, mAP50, mAP50-95 sobre valid
Al terminar se evalúa best.pt en valid y test, y se calcula una "accuracy por
imagen" (clase de la detección más confiable == clase real) que es lo que
equivale a la decisión que toma la app.

Uso:
  python train.py --data dataset/data.yaml --epochs 50
  python train.py --data dataset/data.yaml --epochs 1 --imgsz 320   # prueba rápida
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import yaml

ML_DIR = Path(__file__).resolve().parent

# Las dependencias están fijadas en requirements.txt: que Ultralytics no haga
# pip install en caliente (con onnx sin wheel se queda compilando indefinidamente).
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

# Consolas Windows (cp1252) no pueden imprimir emojis/acentos si la salida se redirige
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


def print_epoch_table(results_csv: Path):
    if not results_csv.exists():
        print("  (results.csv no encontrado)")
        return
    with results_csv.open(newline="") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return
    cols = [
        ("epoch", "época"),
        ("train/box_loss", "tr_box"),
        ("train/cls_loss", "tr_cls"),
        ("val/box_loss", "val_box"),
        ("val/cls_loss", "val_cls"),
        ("metrics/precision(B)", "prec"),
        ("metrics/recall(B)", "recall"),
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
    ]
    cols = [(k, h) for k, h in cols if k in rows[0]]
    print("  " + " ".join(f"{h:>9s}" for _, h in cols))
    for r in rows:
        print("  " + " ".join(
            f"{int(float(r[k])):>9d}" if k == "epoch" else f"{float(r[k]):>9.4f}" for k, _ in cols
        ))


def image_accuracy(model, data_yaml: Path, split: str, imgsz: int, device: str, conf: float = 0.25):
    """
    Para cada imagen cuyas etiquetas tienen una sola clase, compara esa clase con
    la de la detección de mayor confianza. Imágenes negativas (sin etiquetas)
    cuentan como acierto si no hay detecciones.
    """
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if split not in cfg:
        return None
    img_dir = Path(cfg["path"]) / cfg[split]
    lbl_dir = img_dir.parent / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    correct = total = 0
    for img in images:
        lbl = lbl_dir / (img.stem + ".txt")
        classes = {int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()} if lbl.exists() else set()
        if len(classes) > 1:
            continue  # imagen con varios gestos: no hay una única respuesta correcta
        res = model.predict(str(img), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        if len(res.boxes) == 0:
            pred = None
        else:
            pred = int(res.boxes.cls[res.boxes.conf.argmax()].item())
        truth = next(iter(classes)) if classes else None
        correct += int(pred == truth)
        total += 1
    return (correct / total, correct, total) if total else None


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento YOLOv8 para gestos de mano.")
    parser.add_argument("--data", required=True, help="Ruta a data.yaml del dataset preparado")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen")
    parser.add_argument("--batch", type=int, default=16, help="Tamaño de batch (-1 = automático en GPU)")
    parser.add_argument("--model", default="yolov8n.pt", help="Modelo base YOLOv8")
    parser.add_argument("--name", default="gesture", help="Nombre del experimento")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping: épocas sin mejora")
    parser.add_argument("--device", default=None, help="'0' (GPU), 'cpu'… Por defecto: GPU si hay CUDA")
    parser.add_argument("--workers", type=int, default=2, help="Procesos del dataloader")
    parser.add_argument("--seed", type=int, default=0, help="Semilla para reproducibilidad")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise SystemExit(
            f"❌ data.yaml no encontrado: {data_path}\n"
            "Ejecuta primero: python prepare_dataset.py ..."
        )

    from ultralytics import YOLO  # import tardío: --help no carga torch

    device = pick_device(args.device)
    print("🚀 Iniciando entrenamiento YOLOv8")
    print(f"   Modelo base : {args.model}")
    print(f"   Dataset     : {data_path}")
    print(f"   Épocas      : {args.epochs}")
    print(f"   Imagen      : {args.imgsz}px")
    print(f"   Dispositivo : {device}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=str(ML_DIR / "runs" / "detect"),
        exist_ok=False,          # cada ejecución en carpeta nueva (gesture, gesture2, …)
        patience=args.patience,
        device=device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        plots=True,              # results.png, confusion_matrix.png, PR curves
        # Augmentaciones ligeras para dataset pequeño
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,              # mano izquierda/derecha
        mosaic=0.5,
        mixup=0.0,
    )

    # Ultralytics incrementa el nombre si la carpeta existe: usar la ruta real.
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"❌ No se generó best.pt en {best.parent}")

    print("\n📈 Métricas por época (results.csv):")
    print_epoch_table(save_dir / "results.csv")

    print("\n🔎 Evaluando best.pt")
    best_model = YOLO(str(best))
    for split in ("val", "test"):
        if split == "test" and "test" not in yaml.safe_load(data_path.read_text(encoding="utf-8")):
            continue
        m = best_model.val(data=str(data_path), split=split, imgsz=args.imgsz, device=device,
                           plots=False, verbose=False, project=str(save_dir), name=f"eval_{split}")
        print(f"   [{split:4s}] precision={m.box.mp:.3f}  recall={m.box.mr:.3f}  "
              f"mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")
        acc = image_accuracy(best_model, data_path, split, args.imgsz, device)
        if acc:
            print(f"          accuracy por imagen = {acc[0]:.3f} ({acc[1]}/{acc[2]})")

    # Copia estable para no depender del sufijo gesture/gesture2/…
    stable = ML_DIR / "weights" / "gesture_best.pt"
    stable.parent.mkdir(exist_ok=True)
    shutil.copy2(best, stable)

    print("\n✅ Entrenamiento completado.")
    print(f"   Carpeta del run : {save_dir}")
    print(f"   Pesos (best)    : {best}")
    print(f"   Copia estable   : {stable}")
    print("\nSiguiente paso:")
    print(f"  python export_onnx.py --weights {stable.relative_to(ML_DIR)}")


if __name__ == "__main__":
    main()

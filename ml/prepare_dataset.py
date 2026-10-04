"""
prepare_dataset.py
------------------
Toma el dataset descargado de Roboflow en formato YOLOv8 y lo reorganiza en
`dataset/` con los nombres de clase correctos.

Mapeo (verificado visualmente sobre el dataset v1 de Roboflow):
  Rock    (id 0 Roboflow) → puno_cerrado (id 1)  → '0' → LED OFF
  Paper   (id 1 Roboflow) → mano_abierta (id 0)  → '1' → LED ON
  Scissors(id 2 Roboflow) → se elimina la anotación. Las imágenes que sólo
                            contienen Scissors se conservan como negativos
                            (etiqueta vacía) para que el modelo NO dispare con
                            ese gesto. Desactivable con --no-negatives.

Validaciones:
  - Imágenes corruptas o ilegibles → se descartan.
  - Líneas de etiqueta mal formadas, con NaN o fuera de [0,1] → se descartan.
  - Polígonos (formato segmentación, que Roboflow mezcla en este dataset) →
    se convierten a caja (xc, yc, w, h).
  - Imágenes sin archivo .txt → se descartan (no sabemos si son fondo).
  - Aviso si la misma imagen original aparece en dos splits (fuga de datos).
  - Error si train o valid quedan vacíos.

La normalización de píxeles (/255, letterbox a imgsz) la hace Ultralytics al
entrenar; aquí sólo se garantiza que las coordenadas estén normalizadas a [0,1].

Uso:
  python prepare_dataset.py --source "..\\Rock paper scissors.v1i.yolov8" --out dataset
"""

import argparse
import math
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

# Mapeo: id_origen → id_destino. Ids ausentes se descartan.
REMAP = {
    1: 0,  # Paper → mano_abierta → LED ON
    0: 1,  # Rock  → puno_cerrado → LED OFF
    # 2: Scissors → descartado
}
CLASS_NAMES = ["mano_abierta", "puno_cerrado"]  # debe coincidir con frontend/src/config.ts

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "valid", "test")
EPS = 1e-3  # tolerancia para coordenadas ligeramente fuera de [0,1]

# Consolas Windows (cp1252) no pueden imprimir emojis/acentos si la salida se redirige
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_label_line(line: str):
    """
    Convierte una línea YOLO (caja o polígono) en (cls, xc, yc, w, h).
    Devuelve None si la línea es inválida.
    """
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls = int(parts[0])
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in coords):
        return None

    if len(coords) == 4:
        xc, yc, w, h = coords
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        # Polígono x1 y1 x2 y2 ... → caja envolvente
        xs, ys = coords[0::2], coords[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        if min(x1, y1) < -EPS or max(x2, y2) > 1 + EPS:
            return None
        x1, y1 = max(x1, 0.0), max(y1, 0.0)
        x2, y2 = min(x2, 1.0), min(y2, 1.0)
        xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    else:
        return None

    if w <= 0 or h <= 0:
        return None
    if min(xc, yc) < -EPS or max(xc, yc, w, h) > 1 + EPS:
        return None
    clamp = lambda v: min(max(v, 0.0), 1.0)
    return cls, clamp(xc), clamp(yc), clamp(w), clamp(h)


def read_label(src_label: Path, stats: Counter):
    """
    Lee y remapea un .txt. Devuelve (líneas_destino, nº_anotaciones_origen).
    """
    lines_out = []
    n_src = 0
    try:
        text = src_label.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        stats["etiqueta_ilegible"] += 1
        return None, 0

    for line in text.splitlines():
        if not line.strip():
            continue
        parsed = parse_label_line(line)
        if parsed is None:
            stats["linea_invalida"] += 1
            continue
        n_src += 1
        cls, xc, yc, w, h = parsed
        if cls not in REMAP:
            stats["anotacion_descartada(Scissors/otra)"] += 1
            continue
        lines_out.append(f"{REMAP[cls]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines_out, n_src


def image_ok(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:  # verify() no decodifica; load() sí
            im.load()
        return True
    except Exception:
        return False


def source_id(img_name: str) -> str:
    """'101_jpg.rf.abc123.jpg' → '101' (imagen original antes de augmentación Roboflow)."""
    return re.sub(r"\.rf\.[0-9a-f]+$", "", Path(img_name).stem)


def process_split(src_root: Path, dst_root: Path, split: str, keep_negatives: bool,
                  class_counts: Counter, origins: dict) -> int:
    src_img = src_root / split / "images"
    src_lbl = src_root / split / "labels"

    if not src_img.exists():
        print(f"  [SKIP] {split}: carpeta de imágenes no encontrada")
        return 0

    dst_img = dst_root / split / "images"
    dst_lbl = dst_root / split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    for img_path in sorted(src_img.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        stats["imagenes_origen"] += 1

        lbl_path = src_lbl / (img_path.stem + ".txt")
        if not lbl_path.exists():
            stats["sin_etiqueta"] += 1
            continue

        lines_out, n_src = read_label(lbl_path, stats)
        if lines_out is None:
            continue
        if not lines_out:
            # Sin anotaciones útiles: sólo es negativo válido si había anotaciones
            # (p.ej. sólo Scissors). Un .txt vacío/corrupto de origen se descarta.
            if not (keep_negatives and n_src > 0):
                stats["sin_anotaciones_validas"] += 1
                continue

        if not image_ok(img_path):
            stats["imagen_corrupta"] += 1
            continue

        shutil.copy2(img_path, dst_img / img_path.name)
        (dst_lbl / (img_path.stem + ".txt")).write_text(
            "\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8"
        )
        for l in lines_out:
            class_counts[(split, int(l.split()[0]))] += 1
        stats["negativos" if not lines_out else "positivos"] += 1
        origins.setdefault(source_id(img_path.name), set()).add(split)

    copied = stats["positivos"] + stats["negativos"]
    detail = ", ".join(f"{k}={v}" for k, v in sorted(stats.items()) if k not in ("positivos", "negativos", "imagenes_origen"))
    print(f"  {split:5s}: {copied:4d}/{stats['imagenes_origen']} imágenes "
          f"(positivas={stats['positivos']}, negativas={stats['negativos']})"
          + (f"  descartes: {detail}" if detail else ""))
    return copied


def write_data_yaml(dst_root: Path, has_test: bool) -> Path:
    data = {
        "path": str(dst_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
    }
    if has_test:
        data["test"] = "test/images"
    data["nc"] = len(CLASS_NAMES)
    data["names"] = {i: n for i, n in enumerate(CLASS_NAMES)}
    yaml_path = dst_root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepara dataset para entrenamiento YOLOv8.")
    parser.add_argument("--source", required=True, help="Ruta al dataset de Roboflow descargado")
    parser.add_argument("--out", default="dataset", help="Directorio de salida")
    parser.add_argument("--no-negatives", action="store_true",
                        help="Descarta imágenes que sólo contienen Scissors en vez de usarlas como fondo")
    args = parser.parse_args()

    src = Path(args.source)
    dst = Path(args.out)

    if not src.is_dir():
        raise SystemExit(f"❌ Directorio fuente no encontrado: {src.resolve()}")
    if not (src / "train" / "images").is_dir():
        raise SystemExit(f"❌ {src} no parece un export YOLOv8 de Roboflow (falta train/images)")
    if dst.resolve() == src.resolve():
        raise SystemExit("❌ --out no puede ser el mismo directorio que --source")

    if dst.exists():
        print(f"Limpiando directorio previo: {dst}")
        shutil.rmtree(dst)

    print(f"Fuente : {src.resolve()}")
    print(f"Destino: {dst.resolve()}\n")

    class_counts: Counter = Counter()
    origins: dict = {}
    counts = {s: process_split(src, dst, s, not args.no_negatives, class_counts, origins) for s in SPLITS}

    if counts["train"] == 0 or counts["valid"] == 0:
        raise SystemExit("❌ train o valid quedaron vacíos tras el filtrado; revisa el dataset fuente.")

    print("\nAnotaciones por clase:")
    for split in SPLITS:
        if counts[split]:
            row = "  ".join(f"{n}={class_counts[(split, i)]}" for i, n in enumerate(CLASS_NAMES))
            print(f"  {split:5s}: {row}")
            for i, n in enumerate(CLASS_NAMES):
                if class_counts[(split, i)] == 0:
                    print(f"  ⚠️  {split} no tiene ninguna anotación de '{n}'")

    leaked = sorted(k for k, v in origins.items() if len(v) > 1)
    if leaked:
        print(f"\n⚠️  {len(leaked)} imágenes originales aparecen en varios splits "
              f"(fuga de datos, métricas optimistas): {leaked[:10]}")

    yaml_path = write_data_yaml(dst, has_test=counts["test"] > 0)
    print(f"\ndata.yaml escrito en: {yaml_path}")
    print("\n✅ Dataset preparado correctamente.")
    print(f"\nSiguiente paso:\n  python train.py --data {yaml_path}")


if __name__ == "__main__":
    main()

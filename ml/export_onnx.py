"""
export_onnx.py
--------------
Exporta los pesos .pt entrenados a formato ONNX para uso con onnxruntime-web
y verifica el resultado.

Contrato del modelo exportado (también se guarda en <out>.json):
  entrada  "images" : float32 [1, 3, imgsz, imgsz], RGB, valores 0..1, letterbox
  salida   "output0": float32 [1, 4 + nc, N]  (N = 8400 para 640px)
           filas 0..3 = cx, cy, w, h en píxeles de la entrada
           filas 4..  = score por clase (ya con sigmoide). SIN NMS: hay que
           aplicarlo en el cliente.

Nota: el modelo (~12 MB) es para el navegador/PC. Un Arduino Uno (2 KB de RAM)
no puede ejecutarlo; el Arduino sólo recibe el comando '0'/'1' por serie.

Uso:
  python export_onnx.py --weights weights/gesture_best.pt
  python export_onnx.py --weights runs/detect/gesture/weights/best.pt --out ..\\frontend\\public\\models\\gesture.onnx
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = ML_DIR.parent / "frontend" / "public" / "models" / "gesture.onnx"

# Las dependencias están fijadas en requirements.txt: que Ultralytics no haga
# pip install en caliente (con onnx sin wheel se queda compilando indefinidamente).
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

# Consolas Windows (cp1252) no pueden imprimir emojis/acentos si la salida se redirige
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def verify_onnx(onnx_path: Path, imgsz: int, nc: int) -> dict:
    import numpy as np
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    opset = max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    expected_in = [1, 3, imgsz, imgsz]
    if list(inp.shape) != expected_in:
        raise SystemExit(f"Entrada inesperada {inp.shape}, se esperaba {expected_in}")

    y = sess.run(None, {inp.name: np.random.rand(*expected_in).astype(np.float32)})[0]
    if y.ndim != 3 or y.shape[0] != 1 or y.shape[1] != 4 + nc:
        raise SystemExit(f"Salida inesperada {y.shape}, se esperaba [1, {4 + nc}, N]")
    if not np.isfinite(y).all():
        raise SystemExit("La salida contiene NaN/Inf")

    return {"opset": opset, "input_name": inp.name, "input_shape": expected_in,
            "output_name": out.name, "output_shape": list(y.shape)}


def main():
    parser = argparse.ArgumentParser(description="Exporta YOLOv8 a ONNX.")
    parser.add_argument("--weights", required=True, help="Ruta al archivo best.pt")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Ruta de salida del archivo .onnx")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Tamaño de exportación (debe coincidir con el del preprocesado en el frontend)")
    parser.add_argument("--opset", type=int, default=12,
                        help="Opset ONNX (12 es seguro para onnxruntime-web 1.20 WASM)")
    args = parser.parse_args()

    weights_path = Path(args.weights).resolve()
    out_path = Path(args.out).resolve()

    if not weights_path.exists():
        raise SystemExit(
            f"Archivo de pesos no encontrado: {weights_path}\n"
            "Ejecuta primero: python train.py ..."
        )

    from ultralytics import YOLO

    print(f"Exportando {weights_path} → ONNX")
    model = YOLO(str(weights_path))
    names = {int(k): v for k, v in model.names.items()}
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,     # shape fija para el navegador
        half=False,        # FP16 no está soportado por el backend WASM
        nms=False,
        device="cpu",
    )
    auto_onnx = Path(exported) if exported else weights_path.with_suffix(".onnx")
    if not auto_onnx.exists():
        raise SystemExit(f"Archivo ONNX no generado en: {auto_onnx}")

    print("🔎 Verificando ONNX con onnx.checker + onnxruntime…")
    info = verify_onnx(auto_onnx, args.imgsz, len(names))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(auto_onnx, out_path)
    meta = {"names": names, "imgsz": args.imgsz, "input_range": "0..1 RGB, letterbox",
            "nms_included": False, **info}
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    size_mb = out_path.stat().st_size / 2**20
    print(f"   opset={info['opset']}  entrada={info['input_name']}{info['input_shape']}  "
          f"salida={info['output_name']}{info['output_shape']}  clases={names}")
    print(f"\nModelo ONNX ({size_mb:.1f} MB) copiado a: {out_path}")
    print(f"   Metadatos: {meta_path}")


if __name__ == "__main__":
    main()

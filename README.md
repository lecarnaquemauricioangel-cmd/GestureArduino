# GestureArduino 
Control de un LED en Arduino mediante reconocimiento de gestos de mano.
Stack: **Python · YOLOv8 · ONNX Runtime Web · React · Web Serial API**

---

## Estructura del proyecto

```
gesture-arduino/
├── ml/                          # Pipeline de ML (Python)
│   ├── requirements.txt
│   ├── prepare_dataset.py       # Reorganiza y remapea clases del dataset
│   ├── train.py                 # Entrenamiento YOLOv8
│   └── export_onnx.py           # Exportación a ONNX
│
├── frontend/                    # App React + Vite
│   ├── public/
│   │   ├── models/              # ← gesture.onnx va aquí (generado en paso 1)
│   │   └── ort/                 # ← WASM de onnxruntime-web (npm install los copia)
│   └── src/
│       ├── config.ts            # Parámetros ajustables (umbral, frames estables…)
│       ├── gestureDetector.ts   # Inferencia ONNX
│       ├── serialPort.ts        # Web Serial API
│       ├── useCamera.ts         # Hook de cámara
│       ├── useGestureLoop.ts    # Bucle de inferencia + estabilización
│       ├── App.tsx              # UI principal
│       └── index.css            # Estilos (dark mode)
│
└── arduino/
    └── led_gesture/
        └── led_gesture.ino      # Firmware Arduino Uno
```

---

## 1 · Entrenar el modelo (una sola vez)

### 1.1 Descargar el dataset

Ve a [Roboflow Universe – Rock Paper Scissors](https://universe.roboflow.com/judi-elmorshedy/rock-paper-scissors-mabzl)
y descárgalo en formato **YOLOv8**. Descomprímelo en la **raíz** del proyecto:

```
gesture-arduino/
└── Rock paper scissors.v1i.yolov8/   ← aquí
```

### 1.2 Preparar el entorno y entrenar

Requiere Python 3.10–3.13 (probado con 3.13).

```powershell
cd ml
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# (Opcional, GPU NVIDIA) instala torch con CUDA ANTES de requirements:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

python prepare_dataset.py --source "..\Rock paper scissors.v1i.yolov8" --out dataset

python train.py --data dataset/data.yaml --epochs 1 --imgsz 320   # prueba rápida
python train.py --data dataset/data.yaml --epochs 50              # entrenamiento real

python export_onnx.py --weights weights/gesture_best.pt   # → ..\frontend\public\models\gesture.onnx (+ .json)
```

`train.py` copia siempre el mejor modelo a `ml/weights/gesture_best.pt` y deja las
métricas por época en `ml/runs/detect/<name>/results.csv` (+ `results.png`).

**Mapeo de clases aplicado:**

| Clase origen | id Roboflow | Clase destino   | id | Comando |
|---|---|---|---|---|
| Paper    | 1 | mano_abierta | 0 | `'1'` → LED ON  |
| Rock     | 0 | puno_cerrado | 1 | `'0'` → LED OFF |
| Scissors | 2 | *descartada* (imágenes sólo-Scissors se usan como fondo) | — | — |

---

## 2 · Ejecutar la app

```powershell
cd frontend
npm install      # también copia los WASM de onnxruntime-web a public/ort/
npm run dev
```

Abre <http://localhost:5173>:

1. Pulsa **Actualizar cámaras** → elige dispositivo → **Iniciar cámara**
2. Pulsa **Conectar Arduino** y selecciona el puerto COM
3. Muestra la mano:  
   - 🖐️ Palma abierta → LED **ON**  
   - ✊ Puño cerrado  → LED **OFF**

---

## 3 · Subir el firmware

**Circuito:** `D13 → ánodo LED → cátodo → resistencia 220 Ω → GND`

```powershell
arduino-cli compile -b arduino:avr:uno arduino/led_gesture
arduino-cli upload -b arduino:avr:uno -p COM3 arduino/led_gesture
```

O abre `arduino/led_gesture/led_gesture.ino` en el IDE de Arduino y usa el botón **Subir**.

---

## Problemas frecuentes

| Síntoma | Causa probable |
|---|---|
| "No se pudo cargar el modelo" | Falta `frontend/public/models/gesture.onnx` → ejecuta el paso 1 |
| "Este navegador no soporta Web Serial" | Usa **Chrome** o **Edge** de escritorio sobre `localhost` |
| El LED parpadea | Sube `CONF_THRESHOLD` en `src/config.ts` |
| No aparece el selector de puerto | El Arduino ya está abierto en el Monitor Serial: ciérralo |
| No aparece la cámara externa | Conéctala, pulsa **Actualizar cámaras** y concede permiso de cámara |

---

## Ajustes en `src/config.ts`

| Parámetro | Por defecto | Efecto |
|---|---|---|
| `CONF_THRESHOLD` | `0.55` | Confianza mínima para una detección válida |
| `STABLE_FRAMES`  | `5`    | Frames consecutivos antes de enviar comando al Arduino |
| `SERIAL_BAUD`    | `9600` | Baudrate (debe coincidir con el firmware) |

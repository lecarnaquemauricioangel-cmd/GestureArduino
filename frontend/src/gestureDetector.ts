/**
 * gestureDetector.ts
 * ------------------
 * Detección de la mano con MediaPipe HandLandmarker (@mediapipe/tasks-vision)
 * y clasificación geométrica por conteo de dedos extendidos:
 *
 *   sin mano          → null (no hay predicción, no se envía nada)
 *   4–5 dedos         → mano_abierta  → LED ON  ('1')
 *   0–1 dedos         → puno_cerrado  → LED OFF ('0')
 *   2–3 dedos         → indefinido    → no se envía nada
 *
 * El runtime WASM y el modelo se sirven en local desde public/:
 *   /mediapipe/wasm/vision_wasm_internal.{js,wasm}
 *   /models/hand_landmarker.task
 */

import {
  FilesetResolver,
  HandLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import {
  HAND_CONFIDENCE,
  MAX_FINGERS_CLOSED,
  MIN_FINGERS_OPEN,
} from "./config";
import type { ClassName } from "./config";

let landmarker: HandLandmarker | null = null;
let loadPromise: Promise<void> | null = null;
let lastTimestamp = 0;

/** Carga MediaPipe HandLandmarker (una sola vez, seguro ante StrictMode). */
export function loadModel(): Promise<void> {
  if (!loadPromise) {
    loadPromise = doLoad().catch((e) => {
      loadPromise = null; // permitir reintento
      throw e;
    });
  }
  return loadPromise;
}

// La versión del WASM debe coincidir con la del paquete JS instalado
// (vendor/mediapipe-tasks-vision = 1.0.1). "@latest" se rompería en cuanto
// Google publique otra versión con un WASM incompatible.
const MP_VERSION = "1.0.1";

/** Orígenes de assets, en orden de preferencia. */
const SOURCES = [
  {
    name: "CDN oficial",
    wasm: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`,
    model:
      "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  },
  {
    name: "local (public/)",
    wasm: "/mediapipe/wasm",
    model: "/models/hand_landmarker.task",
  },
];

/** Convierte cualquier cosa lanzada (Error, Event de <script>, string…) en texto útil. */
function describeError(e: unknown): string {
  if (e instanceof Error) return `${e.name}: ${e.message}`;
  if (e instanceof Event) {
    const src = (e.target as HTMLScriptElement | null)?.src;
    return `Evento "${e.type}" al cargar ${src ?? "un recurso"} (¿red bloqueada o archivo inexistente?)`;
  }
  return String(e);
}

async function doLoad(): Promise<void> {
  const failures: string[] = [];

  for (const src of SOURCES) {
    let fileset;
    try {
      console.log(`[MediaPipe] Cargando runtime WASM desde ${src.name}: ${src.wasm}`);
      fileset = await FilesetResolver.forVisionTasks(src.wasm);
    } catch (e) {
      const msg = `${src.name} · FilesetResolver: ${describeError(e)}`;
      console.error(`[MediaPipe] ${msg}`, e);
      failures.push(msg);
      continue;
    }

    for (const delegate of ["GPU", "CPU"] as const) {
      try {
        console.log(`[MediaPipe] Creando HandLandmarker (${delegate}) con modelo ${src.model}`);
        landmarker = await HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: src.model, delegate },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: HAND_CONFIDENCE,
          minHandPresenceConfidence: HAND_CONFIDENCE,
          minTrackingConfidence: 0.5,
        });
        console.log(`[MediaPipe] HandLandmarker listo (${src.name}, ${delegate})`);
        return;
      } catch (e) {
        const msg = `${src.name} · HandLandmarker ${delegate}: ${describeError(e)}`;
        console.error(`[MediaPipe] ${msg}`, e);
        failures.push(msg);
      }
    }
  }

  console.error("[MediaPipe] Todos los intentos fallaron:\n" + failures.join("\n"));
  throw new Error(failures.join(" | "));
}

export function isModelLoaded(): boolean {
  return landmarker !== null;
}

export interface Detection {
  /** Gesto reconocido, o null si la postura es intermedia (2–3 dedos) */
  className: ClassName | null;
  /** Dedos extendidos (0–5) */
  fingers: number;
  /** Confianza de MediaPipe en que es una mano */
  confidence: number;
  /** 21 landmarks en píxeles del vídeo */
  landmarks: { x: number; y: number }[];
  /** Caja en píxeles del vídeo (centro x/y, ancho, alto) */
  box: { x: number; y: number; w: number; h: number };
}

/**
 * Procesa el frame actual del vídeo. Devuelve null si no hay mano.
 * Es síncrono y rápido (~30 fps), se llama una vez por frame de vídeo.
 */
export function runInference(video: HTMLVideoElement): Detection | null {
  if (!landmarker) return null;
  if (video.readyState < 2 || !video.videoWidth) return null;

  // MediaPipe exige timestamps estrictamente crecientes
  let ts = performance.now();
  if (ts <= lastTimestamp) ts = lastTimestamp + 1;
  lastTimestamp = ts;

  const result = landmarker.detectForVideo(video, ts);
  const lm = result.landmarks[0];
  if (!lm || lm.length < 21) return null; // sin mano → sin predicción

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  // A píxeles: las coords normalizadas de x e y tienen escalas distintas
  const pts = lm.map((p: NormalizedLandmark) => ({ x: p.x * vw, y: p.y * vh }));

  const fingers = countExtendedFingers(pts);
  const className: ClassName | null =
    fingers >= MIN_FINGERS_OPEN
      ? "mano_abierta"
      : fingers <= MAX_FINGERS_CLOSED
      ? "puno_cerrado"
      : null;

  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);

  return {
    className,
    fingers,
    confidence: result.handedness[0]?.[0]?.score ?? 1,
    landmarks: pts,
    box: {
      x: (minX + maxX) / 2,
      y: (minY + maxY) / 2,
      w: maxX - minX,
      h: maxY - minY,
    },
  };
}

/**
 * Cuenta dedos extendidos con los 21 landmarks:
 *   0 = muñeca · 1-4 pulgar · 5-8 índice · 9-12 medio · 13-16 anular · 17-20 meñique
 *
 * Índice a meñique: extendido si la punta (TIP) está más lejos de la muñeca
 * que la articulación PIP. Para la mano vertical equivale a "punta por encima
 * del nudillo", pero también funciona si la mano está inclinada o de lado.
 */
function countExtendedFingers(p: { x: number; y: number }[]): number {
  const d = (a: number, b: number) => Math.hypot(p[a].x - p[b].x, p[a].y - p[b].y);
  const palm = d(0, 9) || 1; // tamaño de referencia: muñeca → base del medio

  let count = 0;
  // [TIP, PIP] de índice, medio, anular, meñique
  for (const [tip, pip] of [[8, 6], [12, 10], [16, 14], [20, 18]]) {
    if (d(0, tip) > d(0, pip) * 1.1) count++;
  }
  // Pulgar: punta alejada de la base del índice y más lejos del meñique que su IP
  if (d(4, 5) > palm * 0.55 && d(4, 17) > d(3, 17)) count++;

  return count;
}

// ─── Dibujo ──────────────────────────────────────────────────────────────────

const COLOR: Record<ClassName | "none", string> = {
  mano_abierta: "#22c55e",
  puno_cerrado: "#f97316",
  none: "#eab308",
};

/**
 * Dibuja el esqueleto de la mano (líneas azules, puntos verdes), la caja y la
 * etiqueta con el gesto y los dedos contados. El canvas adopta la resolución
 * real del vídeo, así los landmarks (en píxeles del vídeo) se usan tal cual.
 */
export function drawDetection(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  det: Detection | null
): void {
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;
  if (canvas.width !== vw || canvas.height !== vh) {
    canvas.width = vw;
    canvas.height = vh;
  }
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, vw, vh);
  if (!det) return;

  const k = vw / 640; // escala de trazo/fuente según resolución
  const pts = det.landmarks;

  // Esqueleto
  ctx.strokeStyle = "#3b82f6";
  ctx.lineWidth = 3 * k;
  ctx.lineCap = "round";
  ctx.beginPath();
  for (const { start, end } of HandLandmarker.HAND_CONNECTIONS) {
    ctx.moveTo(pts[start].x, pts[start].y);
    ctx.lineTo(pts[end].x, pts[end].y);
  }
  ctx.stroke();

  // Puntos
  ctx.fillStyle = "#22c55e";
  for (const pt of pts) {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 4 * k, 0, Math.PI * 2);
    ctx.fill();
  }

  // Caja
  const color = COLOR[det.className ?? "none"];
  const pad = 12 * k;
  const bx = Math.max(0, det.box.x - det.box.w / 2 - pad);
  const by = Math.max(0, det.box.y - det.box.h / 2 - pad);
  const bw = Math.min(vw - bx, det.box.w + pad * 2);
  const bh = Math.min(vh - by, det.box.h + pad * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 * k;
  ctx.setLineDash([8 * k, 6 * k]);
  ctx.strokeRect(bx, by, bw, bh);
  ctx.setLineDash([]);

  // Etiqueta: "mano_abierta · 5 dedos"
  const label = `${det.className ?? "indefinido"} · ${det.fingers} dedos`;
  ctx.font = `bold ${Math.round(16 * k)}px Inter, sans-serif`;
  const lpad = 6 * k;
  const lh = 24 * k;
  const tw = ctx.measureText(label).width + lpad * 2;
  const ly = by >= lh ? by - lh : by + bh;
  ctx.fillStyle = color;
  ctx.fillRect(bx, ly, tw, lh);
  ctx.fillStyle = "#000";
  ctx.textBaseline = "middle";
  ctx.fillText(label, bx + lpad, ly + lh / 2);
}

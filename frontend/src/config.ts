// ─── Configuración global de la aplicación ─────────────────────────────────

/** Cuántos frames consecutivos con el mismo gesto antes de enviar comando */
export const STABLE_FRAMES = 1;

/** Gestos reconocidos */
export const CLASS_NAMES = ["mano_abierta", "puno_cerrado"] as const;
export type ClassName = (typeof CLASS_NAMES)[number];

/** Comando serie que se envía al Arduino para cada gesto */
export const CLASS_COMMAND: Record<ClassName, string> = {
  mano_abierta: "1",  // LED ON
  puno_cerrado: "0",  // LED OFF
};

/** Baudrate del puerto serie */
export const SERIAL_BAUD = 9600;

/** Espera tras abrir el puerto: el Arduino se reinicia al conectar (bootloader) */
export const SERIAL_BOOT_MS = 2000;

// ─── MediaPipe Hands ───────────────────────────────────────────────────────

/** Confianza mínima para aceptar que hay una mano (evita falsos positivos) */
export const HAND_CONFIDENCE = 0.6;

/** Dedos extendidos para considerar mano abierta (4 o 5) */
export const MIN_FINGERS_OPEN = 4;

/** Máximo de dedos extendidos para considerar puño cerrado (0 o 1) */
export const MAX_FINGERS_CLOSED = 1;

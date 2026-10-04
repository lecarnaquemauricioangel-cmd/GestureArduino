/**
 * serialPort.ts
 * -------------
 * Wrapper sobre la Web Serial API para comunicación con Arduino.
 * Protocolo: '1' (0x31) → LED ON, '0' (0x30) → LED OFF. 9600 baudios.
 */

import { SERIAL_BAUD, SERIAL_BOOT_MS } from "./config";

const BYTE_ON = new Uint8Array([0x31]); // '1'
const BYTE_OFF = new Uint8Array([0x30]); // '0'

let port: SerialPort | null = null;
let ready = false;
/** Último estado enviado con éxito al Arduino (null = desconocido) */
let ledState: boolean | null = null;
/** Cola de escrituras: cada envío espera al anterior, así nunca hay dos
 *  writers bloqueando el stream a la vez. */
let queue: Promise<void> = Promise.resolve();

export interface SerialLogEntry {
  time: string;
  cmd: "1" | "0";
  ok: boolean;
  detail: string;
}

let onLost: (() => void) | null = null;
let onLog: ((e: SerialLogEntry) => void) | null = null;

/** Callback cuando el puerto se pierde (cable desconectado). */
export function onSerialLost(cb: (() => void) | null): void {
  onLost = cb;
}

/** Callback con el resultado de cada intento de envío (para la UI). */
export function onSerialLog(cb: ((e: SerialLogEntry) => void) | null): void {
  onLog = cb;
}

function log(cmd: "1" | "0", ok: boolean, detail: string) {
  const time = new Date().toLocaleTimeString();
  (ok ? console.log : console.error)(`[Serial] ${time} '${cmd}' → ${detail}`);
  onLog?.({ time, cmd, ok, detail });
}

/** Solicita al usuario seleccionar un puerto serie y lo abre a 9600 baudios. */
export async function connectSerial(): Promise<void> {
  if (port) await disconnectSerial();

  const p = await navigator.serial.requestPort();
  await p.open({ baudRate: SERIAL_BAUD }); // 9600
  port = p;
  ledState = null;
  ready = false;
  console.log(`[Serial] Puerto abierto a ${SERIAL_BAUD} baudios, esperando arranque del Arduino…`);

  // Abrir el puerto reinicia el Arduino (señal DTR): los bytes enviados
  // durante el bootloader se pierden, por eso se espera antes de escribir.
  await new Promise((r) => setTimeout(r, SERIAL_BOOT_MS));
  ready = true;
  console.log("[Serial] Listo para enviar");
}

/** Escribe un byte directamente en el puerto (getWriter → write → releaseLock). */
async function writeBytes(bytes: Uint8Array): Promise<void> {
  if (!port?.writable) throw new Error("Puerto serie no conectado");
  const writer = port.writable.getWriter();
  try {
    await writer.write(bytes);
  } finally {
    writer.releaseLock();
  }
}

/**
 * Envía '1' (on=true) o '0' (on=false). Siempre escribe, sin deduplicar.
 * Usado por los botones de prueba y por el bucle de detección.
 */
export function sendLed(on: boolean): Promise<void> {
  const cmd = on ? "1" : "0";
  console.log(`[Serial] Intentando enviar '${cmd}'…`);

  const job = queue.then(async () => {
    if (!port) {
      log(cmd, false, "NO enviado: puerto no conectado");
      throw new Error("Puerto serie no conectado. Pulsa \"Conectar Arduino\".");
    }
    if (!ready) {
      log(cmd, false, "NO enviado: el Arduino aún está arrancando");
      throw new Error("El Arduino aún está arrancando, espera 2 s.");
    }
    try {
      await writeBytes(on ? BYTE_ON : BYTE_OFF);
      ledState = on;
      log(cmd, true, "enviado OK");
    } catch (e) {
      log(cmd, false, `error de escritura: ${(e as Error).message}`);
      await disconnectSerial();
      onLost?.();
      throw e;
    }
  });
  // La cola sigue aunque este envío falle
  queue = job.catch(() => {});
  return job;
}

/** Estado del LED según el último envío correcto (null si nunca se envió). */
export function getLedState(): boolean | null {
  return ledState;
}

/** Cierra la conexión serie. */
export async function disconnectSerial(): Promise<void> {
  ready = false;
  ledState = null;
  const p = port;
  port = null;
  try {
    await p?.close();
  } catch (e) {
    console.warn("[Serial] Error al cerrar", e);
  }
  console.log("[Serial] Puerto desconectado");
}

/** Indica si hay un puerto abierto y listo para recibir. */
export function isConnected(): boolean {
  return port !== null && ready;
}

/** Detecta si el navegador soporta Web Serial API. */
export function isWebSerialSupported(): boolean {
  return "serial" in navigator;
}

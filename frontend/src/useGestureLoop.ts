/**
 * useGestureLoop.ts
 * -----------------
 * Hook que procesa cada frame nuevo de la cámara con MediaPipe y dispara el
 * comando serie en cuanto cambia el gesto:
 *   mano_abierta con LED apagado   → '1'
 *   puno_cerrado con LED encendido → '0'
 *   sin mano / indefinido          → no se envía nada
 */

import { useCallback, useRef, useState } from "react";
import { runInference, type Detection } from "./gestureDetector";
import { sendLed, getLedState, isConnected } from "./serialPort";
import { STABLE_FRAMES } from "./config";
import type { ClassName } from "./config";

export interface GestureState {
  detection: Detection | null;
  ledOn: boolean;
  /** Frames por segundo que entrega la cámara */
  cameraFps: number;
  /** Frames por segundo procesados por MediaPipe */
  inferenceFps: number;
}

type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: () => void) => number;
  cancelVideoFrameCallback?: (h: number) => void;
};

export function useGestureLoop(videoRef: React.RefObject<HTMLVideoElement>) {
  const [state, setState] = useState<GestureState>({
    detection: null,
    ledOn: false,
    cameraFps: 0,
    inferenceFps: 0,
  });

  const handleRef = useRef<{ id: number; rvfc: boolean } | null>(null);
  const stableCount = useRef(0);
  const lastClass = useRef<ClassName | null>(null);
  const ledOnRef = useRef(false);
  const sending = useRef(false);
  const fpsTimer = useRef(0);
  const camFrames = useRef(0);
  const infFrames = useRef(0);
  const lastVideoTime = useRef(-1);
  const fps = useRef({ cam: 0, inf: 0 });
  const isRunning = useRef(false);

  /** Envío no bloqueante: no se espera la escritura para seguir procesando. */
  const fireCommand = useCallback((wantOn: boolean) => {
    if (!isConnected() || sending.current) return;
    if (getLedState() === wantOn) return; // el LED ya está en ese estado
    sending.current = true;
    sendLed(wantOn)
      .catch((e) => console.error("[Serial] fallo al enviar desde el bucle", e))
      .finally(() => {
        sending.current = false;
      });
  }, []);

  const processFrame = useCallback(() => {
    if (!isRunning.current) return;
    const video = videoRef.current as VideoWithRVFC | null;
    if (!video) return;

    // Programar el siguiente frame primero (requestVideoFrameCallback se
    // dispara una vez por frame de cámara; si no existe, rAF)
    if (video.requestVideoFrameCallback) {
      handleRef.current = { id: video.requestVideoFrameCallback(processFrame), rvfc: true };
    } else {
      handleRef.current = { id: requestAnimationFrame(processFrame), rvfc: false };
    }

    // Saltar si no hay frame nuevo
    if (video.readyState < 2 || video.currentTime === lastVideoTime.current) return;
    lastVideoTime.current = video.currentTime;
    camFrames.current++;

    let det: Detection | null = null;
    try {
      det = runInference(video);
      infFrames.current++;
    } catch (e) {
      console.error("[MediaPipe]", e);
    }

    // FPS cada segundo
    const now = performance.now();
    const elapsed = now - fpsTimer.current;
    if (elapsed >= 1000) {
      fps.current = {
        cam: Math.round((camFrames.current * 1000) / elapsed),
        inf: Math.round((infFrames.current * 1000) / elapsed),
      };
      camFrames.current = 0;
      infFrames.current = 0;
      fpsTimer.current = now;
    }

    // Estabilización (STABLE_FRAMES = 1 → dispara en el primer frame)
    const cls = det?.className ?? null;
    if (cls && cls === lastClass.current) {
      stableCount.current++;
    } else {
      stableCount.current = 1;
      lastClass.current = cls;
    }

    // Solo hay comando si hay mano Y el gesto es claro (abierta o puño)
    if (cls && stableCount.current >= STABLE_FRAMES) {
      ledOnRef.current = cls === "mano_abierta";
      fireCommand(ledOnRef.current);
    }

    setState({
      detection: det,
      ledOn: ledOnRef.current,
      cameraFps: fps.current.cam,
      inferenceFps: fps.current.inf,
    });
  }, [videoRef, fireCommand]);

  const start = useCallback(() => {
    if (isRunning.current) return;
    isRunning.current = true;
    stableCount.current = 0;
    lastClass.current = null;
    lastVideoTime.current = -1;
    fpsTimer.current = performance.now();
    camFrames.current = 0;
    infFrames.current = 0;
    fps.current = { cam: 0, inf: 0 };
    processFrame();
  }, [processFrame]);

  const stop = useCallback(() => {
    isRunning.current = false;
    const h = handleRef.current;
    const video = videoRef.current as VideoWithRVFC | null;
    if (h) {
      if (h.rvfc) video?.cancelVideoFrameCallback?.(h.id);
      else cancelAnimationFrame(h.id);
      handleRef.current = null;
    }
    ledOnRef.current = false;
    if (isConnected() && getLedState() !== false) sendLed(false).catch(() => {});
    setState({ detection: null, ledOn: false, cameraFps: 0, inferenceFps: 0 });
  }, [videoRef]);

  return { state, start, stop };
}

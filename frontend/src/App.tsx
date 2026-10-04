/**
 * App.tsx — GestureArduino
 * Usa MediaPipe HandLandmarker (@mediapipe/tasks-vision) + conteo de dedos.
 * Web Serial API para control del LED del Arduino.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { loadModel, drawDetection } from "./gestureDetector";
import {
  connectSerial,
  disconnectSerial,
  isConnected,
  isWebSerialSupported,
  onSerialLost,
  onSerialLog,
  sendLed,
  type SerialLogEntry,
} from "./serialPort";
import { useCamera } from "./useCamera";
import { useGestureLoop } from "./useGestureLoop";
import type { Detection } from "./gestureDetector";
import { MAX_FINGERS_CLOSED, MIN_FINGERS_OPEN } from "./config";

type ModelStatus = "idle" | "loading" | "ready" | "error";

export default function App() {
  // ── MediaPipe Hands ────────────────────────────────────────────────────────
  const [modelStatus, setModelStatus] = useState<ModelStatus>("idle");
  const [modelError, setModelError] = useState<string | null>(null);
  const [modelAttempt, setModelAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setModelStatus("loading");
    setModelError(null);
    loadModel()
      .then(() => !cancelled && setModelStatus("ready"))
      .catch((e: Error) => {
        if (cancelled) return;
        console.error("[MediaPipe]", e);
        setModelStatus("error");
        setModelError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [modelAttempt]);

  // ── Cámara ────────────────────────────────────────────────────────────
  const {
    videoRef,
    devices,
    selectedDeviceId,
    isStreaming,
    error: cameraError,
    refreshDevices,
    startCamera,
    stopCamera,
    switchCamera,
  } = useCamera();

  // ── Inferencia ────────────────────────────────────────────────────────
  const { state: gestureState, start: startLoop, stop: stopLoop } =
    useGestureLoop(videoRef as React.RefObject<HTMLVideoElement>);

  const handleStartCamera = useCallback(async () => {
    await startCamera();
  }, [startCamera]);

  const handleStopCamera = useCallback(() => {
    stopLoop();
    stopCamera();
  }, [stopLoop, stopCamera]);

  // Arrancar el loop de inferencia cuando modelo Y cámara estén listos
  useEffect(() => {
    if (isStreaming && modelStatus === "ready") {
      startLoop();
    } else if (!isStreaming) {
      stopLoop();
    }
  }, [isStreaming, modelStatus, startLoop, stopLoop]);

  // ── Puerto Serie ──────────────────────────────────────────────────────
  const [serialConnected, setSerialConnected] = useState(false);
  const [serialConnecting, setSerialConnecting] = useState(false);
  const [serialError, setSerialError] = useState<string | null>(null);

  const [serialLog, setSerialLog] = useState<SerialLogEntry[]>([]);

  useEffect(() => {
    onSerialLost(() => {
      setSerialConnected(false);
      setSerialError("Se perdió la conexión con el Arduino");
    });
    onSerialLog((entry) => setSerialLog((prev) => [entry, ...prev].slice(0, 6)));
    return () => {
      onSerialLost(null);
      onSerialLog(null);
    };
  }, []);

  // Botones de prueba: escriben '1' / '0' directamente, sin pasar por el modelo
  const handleForce = useCallback(async (on: boolean) => {
    try {
      setSerialError(null);
      await sendLed(on);
    } catch (e) {
      setSerialError((e as Error).message);
    }
  }, []);

  const handleConnectSerial = useCallback(async () => {
    try {
      setSerialError(null);
      setSerialConnecting(true);
      await connectSerial();
      setSerialConnected(isConnected());
    } catch (e) {
      // El usuario cerró el selector sin elegir puerto
      if ((e as Error).name !== "NotFoundError") {
        setSerialError((e as Error).message);
      }
    } finally {
      setSerialConnecting(false);
    }
  }, []);

  const handleDisconnectSerial = useCallback(async () => {
    await disconnectSerial();
    setSerialConnected(false);
  }, []);

  // ── Overlay canvas (bounding box) ─────────────────────────────────────
  const overlayRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    drawDetection(canvas, video, isStreaming ? gestureState.detection : null);
  }, [gestureState.detection, isStreaming, videoRef]);

  const webSerialOk = isWebSerialSupported();

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">✋</span>
            <span className="logo-text">GestureArduino</span>
          </div>
          <div className="header-badges">
            <ModelBadge status={modelStatus} />
            <SerialBadge connected={serialConnected} />
          </div>
        </div>
      </header>

      <main className="main">
        {/* Video + Overlay */}
        <section className="video-section">
          <div className="video-wrapper">
            <video
              ref={videoRef}
              id="camera-feed"
              className="video-feed"
              muted
              playsInline
              autoPlay
            />
            <canvas
              ref={overlayRef}
              className="overlay-canvas"
              width={640}
              height={480}
            />
            {!isStreaming && (
              <div className="video-placeholder">
                <span className="video-placeholder-icon">📷</span>
                <p>Inicia la cámara para comenzar</p>
              </div>
            )}
            {isStreaming && gestureState.cameraFps > 0 && (
              <div className="fps-badge">
                cámara {gestureState.cameraFps} fps · modelo {gestureState.inferenceFps} fps
              </div>
            )}
            {modelStatus === "loading" && (
              <div className="model-loading-overlay">
                <div className="spinner" />
                <p>Cargando MediaPipe Hands…</p>
              </div>
            )}
          </div>

          <StatusPanel
            active={isStreaming && modelStatus === "ready"}
            detection={gestureState.detection}
            ledOn={gestureState.ledOn}
            cameraFps={gestureState.cameraFps}
            inferenceFps={gestureState.inferenceFps}
          />
        </section>

        {/* Controles */}
        <aside className="controls">
          {/* Cámara */}
          <ControlCard title="📷 Cámara de video" id="camera-card">
            <div className="control-row">
              <button
                id="btn-refresh-cameras"
                className="btn btn-secondary"
                onClick={refreshDevices}
              >
                Actualizar cámaras
              </button>
            </div>
            {devices.length > 0 && (
              <div className="control-row">
                <label className="label" htmlFor="camera-select">
                  Dispositivo
                </label>
                <select
                  id="camera-select"
                  className="select"
                  value={selectedDeviceId}
                  onChange={(e) => switchCamera(e.target.value)}
                >
                  {devices.map((d) => (
                    <option key={d.deviceId} value={d.deviceId}>
                      {d.label}
                    </option>
                  ))}
                </select>
              </div>
            )}
            <div className="control-row">
              {!isStreaming ? (
                <button
                  id="btn-start-camera"
                  className="btn btn-primary"
                  onClick={handleStartCamera}
                >
                  ▶ Iniciar cámara
                </button>
              ) : (
                <button
                  id="btn-stop-camera"
                  className="btn btn-danger"
                  onClick={handleStopCamera}
                >
                  ⏹ Detener cámara
                </button>
              )}
            </div>
            {cameraError && <p className="error-text">{cameraError}</p>}
          </ControlCard>

          {/* Arduino */}
          <ControlCard title="🔌 Arduino" id="arduino-card">
            {!webSerialOk ? (
              <p className="warning-text">
                ⚠️ Usa <strong>Chrome</strong> o <strong>Edge</strong> de
                escritorio sobre <code>localhost</code>.
              </p>
            ) : (
              <>
                <div className="control-row">
                  {!serialConnected ? (
                    <button
                      id="btn-connect-arduino"
                      className="btn btn-primary"
                      onClick={handleConnectSerial}
                      disabled={serialConnecting}
                    >
                      {serialConnecting ? "⏳ Conectando…" : "Conectar Arduino"}
                    </button>
                  ) : (
                    <button
                      id="btn-disconnect-arduino"
                      className="btn btn-danger"
                      onClick={handleDisconnectSerial}
                    >
                      Desconectar
                    </button>
                  )}
                </div>
                {serialConnected && (
                  <p className="success-text">✅ Puerto serie conectado (9600 baudios)</p>
                )}

                <p className="label force-label">Prueba manual del LED D13</p>
                <div className="force-row">
                  <button
                    id="btn-force-on"
                    className="btn btn-force btn-force-on"
                    onClick={() => handleForce(true)}
                  >
                    FORZAR ON ('1')
                  </button>
                  <button
                    id="btn-force-off"
                    className="btn btn-force btn-force-off"
                    onClick={() => handleForce(false)}
                  >
                    FORZAR OFF ('0')
                  </button>
                </div>

                {serialError && <p className="error-text">{serialError}</p>}

                {serialLog.length > 0 && (
                  <ul className="serial-log">
                    {serialLog.map((e, i) => (
                      <li key={i} className={e.ok ? "log-ok" : "log-fail"}>
                        <span>{e.time}</span> '{e.cmd}' {e.detail}
                      </li>
                    ))}
                  </ul>
                )}
              </>
            )}
          </ControlCard>

          {/* Modelo */}
          <ControlCard title="🤖 Modelo" id="model-card">
            <div className="model-info">
              <InfoRow label="Motor" value="MediaPipe HandLandmarker" />
              <InfoRow label="Estado" value={modelStatusLabel(modelStatus)} />
              <InfoRow label="Mano abierta" value={`≥ ${MIN_FINGERS_OPEN} dedos → '1'`} />
              <InfoRow label="Puño cerrado" value={`≤ ${MAX_FINGERS_CLOSED} dedo → '0'`} />
            </div>
            {modelError && (
              <>
                <p className="error-text model-error">{modelError}</p>
                <div className="control-row">
                  <button
                    id="btn-retry-model"
                    className="btn btn-secondary"
                    onClick={() => setModelAttempt((n) => n + 1)}
                  >
                    ↻ Reintentar carga del modelo
                  </button>
                </div>
              </>
            )}
          </ControlCard>

          {/* LED indicator */}
          <div
            className={`led-indicator ${
              gestureState.ledOn ? "led-on" : "led-off"
            }`}
          >
            <div className="led-bulb" />
            <span className="led-label">LED D13</span>
            <span className="led-state">
              {gestureState.ledOn ? "ON" : "OFF"}
            </span>
          </div>
        </aside>
      </main>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function ModelBadge({ status }: { status: ModelStatus }) {
  const map: Record<ModelStatus, { cls: string; text: string }> = {
    idle:    { cls: "badge badge-neutral", text: "Modelo: —" },
    loading: { cls: "badge badge-warning", text: "Modelo: cargando…" },
    ready:   { cls: "badge badge-success", text: "Modelo: listo ✓" },
    error:   { cls: "badge badge-error",   text: "Modelo: error" },
  };
  const { cls, text } = map[status];
  return <span className={cls}>{text}</span>;
}

function SerialBadge({ connected }: { connected: boolean }) {
  return (
    <span
      className={connected ? "badge badge-success" : "badge badge-neutral"}
    >
      {connected ? "Serial: conectado" : "Serial: desconectado"}
    </span>
  );
}

function ControlCard({
  title,
  id,
  children,
}: {
  title: string;
  id?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="control-card" id={id}>
      <h2 className="control-card-title">{title}</h2>
      {children}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="info-row">
      <span className="info-label">{label}</span>
      <span className="info-value">{value}</span>
    </div>
  );
}

function StatusPanel({
  active,
  detection,
  ledOn,
  cameraFps,
  inferenceFps,
}: {
  active: boolean;
  detection: Detection | null;
  ledOn: boolean;
  cameraFps: number;
  inferenceFps: number;
}) {
  const gesture = detection?.className ?? null;
  const label = !active
    ? "DETENIDO"
    : !detection
    ? "SIN MANO"
    : gesture === "mano_abierta"
    ? "🖐️ MANO ABIERTA"
    : gesture === "puno_cerrado"
    ? "✊ PUÑO CERRADO"
    : "🤏 GESTO INDEFINIDO";
  const cls = !gesture
    ? "status-none"
    : gesture === "mano_abierta"
    ? "status-open"
    : "status-closed";

  return (
    <div className={`status-panel ${cls}`}>
      <div className="status-main">
        <span className="status-text">{label}</span>
        {detection && (
          <span className="status-conf">
            {detection.fingers} dedos extendidos · mano {(detection.confidence * 100).toFixed(0)}%
          </span>
        )}
      </div>
      <div className="status-metrics">
        <div className="metric">
          <span className="metric-value">{cameraFps}</span>
          <span className="metric-label">FPS cámara</span>
        </div>
        <div className="metric">
          <span className="metric-value">{inferenceFps}</span>
          <span className="metric-label">FPS modelo</span>
        </div>
        <div className={`metric ${ledOn ? "metric-on" : "metric-off"}`}>
          <span className="metric-value">{ledOn ? "ON" : "OFF"}</span>
          <span className="metric-label">LED D13</span>
        </div>
      </div>
    </div>
  );
}

function modelStatusLabel(s: ModelStatus) {
  return {
    idle:    "—",
    loading: "Cargando ⏳",
    ready:   "Listo ✅",
    error:   "Error ❌",
  }[s];
}

/**
 * useCamera.ts
 * ------------
 * Hook para gestionar el acceso a la cámara y la enumeración de dispositivos.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export interface CameraDevice {
  deviceId: string;
  label: string;
}

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const refreshDevices = useCallback(async () => {
    try {
      // Solicita permiso genérico primero para que los labels sean visibles
      const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
      tmp.getTracks().forEach((t) => t.stop());

      const all = await navigator.mediaDevices.enumerateDevices();
      const cams = all
        .filter((d) => d.kind === "videoinput")
        .map((d, i) => ({
          deviceId: d.deviceId,
          label: d.label || `Cámara ${i + 1}`,
        }));
      setDevices(cams);
      if (cams.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(cams[0].deviceId);
      }
    } catch (e) {
      setError("No se pudo acceder a las cámaras: " + (e as Error).message);
    }
  }, [selectedDeviceId]);

  const startCamera = useCallback(async (deviceId?: string) => {
    const id = deviceId ?? selectedDeviceId;
    try {
      // Detener stream anterior si existe
      streamRef.current?.getTracks().forEach((t) => t.stop());

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: id ? { exact: id } : undefined, width: 640, height: 480 },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setIsStreaming(true);
      setError(null);
    } catch (e) {
      setError("Error al iniciar la cámara: " + (e as Error).message);
    }
  }, [selectedDeviceId]);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  }, []);

  // Cambio de cámara en caliente
  const switchCamera = useCallback(async (deviceId: string) => {
    setSelectedDeviceId(deviceId);
    if (isStreaming) {
      await startCamera(deviceId);
    }
  }, [isStreaming, startCamera]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return {
    videoRef,
    devices,
    selectedDeviceId,
    isStreaming,
    error,
    refreshDevices,
    startCamera,
    stopCamera,
    switchCamera,
  };
}

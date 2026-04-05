import { useEffect, useEffectEvent, useRef, useState } from "react";
import type { DetectedMarker, Point } from "./aruco";
import { estimateDistanceMeters } from "./arucoAnchor";
import { ASSUMED_MARKER_SIZE_METERS } from "./arucoConfig";
import type { WorkerMessage } from "./arucoWorkerProtocol";

const DETECTION_INTERVAL_MS = 1000 / 60;
const DEFAULT_ASPECT_RATIO = 4 / 3;

export function useArucoCamera() {
  const viewportRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const stagedFrameCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const requestTokenRef = useRef(0);
  const streamRef = useRef<MediaStream | null>(null);
  const detectorPendingRef = useRef(false);
  const displayedFrameHeightRef = useRef(0);
  const displayedFrameWidthRef = useRef(0);
  const lastDetectionAtRef = useRef(0);
  const lastWorkerSeqRef = useRef(0);
  const nextWorkerSeqRef = useRef(0);

  const [aspectRatio, setAspectRatio] = useState(DEFAULT_ASPECT_RATIO);
  const [errorMessage, setErrorMessage] = useState<string | null>(() =>
    typeof Worker === "undefined"
      ? "Web Workers are required for this prototype."
      : null,
  );
  const [frameMs, setFrameMs] = useState(0);
  const [markers, setMarkers] = useState<DetectedMarker[]>([]);

  const clearViewport = useEffectEvent(() => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }

    const context = viewport.getContext("2d");
    if (!context) {
      return;
    }

    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, viewport.width, viewport.height);
  });

  const handleWorkerMessage = useEffectEvent((message: WorkerMessage) => {
    if (message.token !== requestTokenRef.current) {
      return;
    }

    detectorPendingRef.current = false;

    if (message.type === "error") {
      setMarkers([]);
      clearViewport();
      setErrorMessage(message.message);
      return;
    }

    if (message.seq < lastWorkerSeqRef.current) {
      return;
    }

    lastWorkerSeqRef.current = message.seq;
    drawStagedFrame(viewportRef.current, stagedFrameCanvasRef.current);
    drawDetectedMarkers(
      viewportRef.current,
      displayedFrameWidthRef.current,
      displayedFrameHeightRef.current,
      message.result.markers,
    );
    setMarkers(message.result.markers);
    setFrameMs(message.result.frameMs);
  });

  useEffect(() => {
    if (typeof Worker === "undefined") {
      return;
    }

    const worker = new Worker(new URL("./aruco.worker.ts", import.meta.url), {
      type: "module",
    });

    workerRef.current = worker;
    worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
      handleWorkerMessage(event.data);
    };
    worker.onerror = () => {
      detectorPendingRef.current = false;
      clearViewport();
      if (workerRef.current === worker) {
        workerRef.current = null;
      }
      worker.terminate();
      setErrorMessage("Worker detection failed.");
    };

    return () => {
      if (workerRef.current === worker) {
        workerRef.current = null;
      }
      worker.terminate();
    };
  }, []);

  const stopCamera = useEffectEvent(() => {
    requestTokenRef.current += 1;
    detectorPendingRef.current = false;
    displayedFrameHeightRef.current = 0;
    displayedFrameWidthRef.current = 0;
    lastDetectionAtRef.current = 0;
    lastWorkerSeqRef.current = 0;
    nextWorkerSeqRef.current = 0;

    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }

    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
    }

    setFrameMs(0);
    setMarkers([]);
    setAspectRatio(DEFAULT_ASPECT_RATIO);
    clearViewport();
  });

  const dispatchDetection = useEffectEvent(
    (video: HTMLVideoElement, now: number, token: number) => {
      const worker = workerRef.current;
      if (!worker) {
        setErrorMessage("Web Workers are required for this prototype.");
        return;
      }

      if (typeof createImageBitmap !== "function") {
        setErrorMessage("createImageBitmap is required for worker detection.");
        return;
      }

      lastDetectionAtRef.current = now;
      detectorPendingRef.current = true;
      const seq = nextWorkerSeqRef.current + 1;
      nextWorkerSeqRef.current = seq;

      void createWorkerFrame(video)
        .then((frame) => {
          if (requestTokenRef.current !== token) {
            frame.close();
            return;
          }

          displayedFrameWidthRef.current = frame.width;
          displayedFrameHeightRef.current = frame.height;
          stageCapturedFrame(stagedFrameCanvasRef, frame);

          worker.postMessage(
            {
              frame,
              seq,
              sourceHeight: video.videoHeight,
              sourceWidth: video.videoWidth,
              token,
              type: "detect",
            },
            [frame],
          );
        })
        .catch((error) => {
          if (requestTokenRef.current !== token) {
            return;
          }

          detectorPendingRef.current = false;
          setErrorMessage(
            error instanceof Error
              ? error.message
              : "Worker frame capture failed.",
          );
        });
    },
  );

  const runStartCamera = useEffectEvent(async () => {
    stopCamera();
    const token = requestTokenRef.current;
    setErrorMessage(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMessage(
        "This browser does not expose the MediaDevices camera API.",
      );
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: "environment" },
          frameRate: { ideal: 30, max: 60 },
          height: { ideal: 480, max: 480 },
          width: { ideal: 640, max: 640 },
        },
      });

      if (requestTokenRef.current !== token) {
        for (const track of stream.getTracks()) {
          track.stop();
        }
        return;
      }

      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) {
        return;
      }

      video.srcObject = stream;
      await video.play();

      if (requestTokenRef.current !== token) {
        for (const track of stream.getTracks()) {
          track.stop();
        }
        return;
      }

      if (video.videoWidth && video.videoHeight) {
        setAspectRatio(video.videoWidth / video.videoHeight);
      }

      const renderFrame = (frameAt: number) => {
        if (requestTokenRef.current !== token) {
          return;
        }

        const activeVideo = videoRef.current;
        const viewport = viewportRef.current;

        if (!activeVideo || !viewport) {
          return;
        }

        if (activeVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
          if (
            !detectorPendingRef.current &&
            frameAt - lastDetectionAtRef.current >= DETECTION_INTERVAL_MS
          ) {
            dispatchDetection(activeVideo, frameAt, token);
          }
        } else {
          clearViewport();
        }

        animationFrameRef.current = requestAnimationFrame(renderFrame);
      };

      animationFrameRef.current = requestAnimationFrame(renderFrame);
    } catch (error) {
      const message =
        error instanceof DOMException && error.name === "NotAllowedError"
          ? "Camera access was blocked. Allow camera permissions and retry."
          : error instanceof Error
            ? error.message
            : "The camera feed could not be started.";

      setErrorMessage(message);
    }
  });

  useEffect(() => {
    void runStartCamera();

    return () => {
      stopCamera();
    };
  }, []);

  return {
    aspectRatio,
    errorMessage,
    frameMs,
    markers,
    videoRef,
    viewportRef,
  };
}

async function createWorkerFrame(video: HTMLVideoElement) {
  return createImageBitmap(video);
}

function clearCanvas(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
) {
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, width, height);
}

function drawFrameToViewport(
  viewport: HTMLCanvasElement | null,
  frame: CanvasImageSource,
) {
  const scene = getViewportScene(viewport);
  if (!scene) {
    return;
  }

  const { context, dpr, height, viewportCanvas, width } = scene;
  clearCanvas(context, viewportCanvas.width, viewportCanvas.height);
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.imageSmoothingEnabled = true;
  context.drawImage(frame, 0, 0, width, height);
}

function drawFrameToBuffer(
  canvas: HTMLCanvasElement,
  frame: CanvasImageSource,
  width: number,
  height: number,
) {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }

  clearCanvas(context, canvas.width, canvas.height);
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.imageSmoothingEnabled = true;
  context.drawImage(frame, 0, 0, width, height);
}

function stageCapturedFrame(
  stagedFrameCanvasRef: { current: HTMLCanvasElement | null },
  frame: ImageBitmap,
) {
  let stagedFrameCanvas = stagedFrameCanvasRef.current;

  if (!stagedFrameCanvas) {
    stagedFrameCanvas = document.createElement("canvas");
    stagedFrameCanvasRef.current = stagedFrameCanvas;
  }

  drawFrameToBuffer(stagedFrameCanvas, frame, frame.width, frame.height);
}

function drawStagedFrame(
  viewport: HTMLCanvasElement | null,
  stagedFrameCanvas: HTMLCanvasElement | null,
) {
  if (!stagedFrameCanvas) {
    return;
  }

  drawFrameToViewport(viewport, stagedFrameCanvas);
}

function drawDetectedMarkers(
  viewport: HTMLCanvasElement | null,
  sourceWidth: number,
  sourceHeight: number,
  markers: DetectedMarker[],
) {
  const scene = getViewportScene(viewport);
  if (!scene || !sourceWidth || !sourceHeight) {
    return;
  }

  const { context, dpr, height, width } = scene;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.lineCap = "round";
  context.lineJoin = "round";
  const scaleX = width / sourceWidth;
  const scaleY = height / sourceHeight;

  for (const marker of markers) {
    const corners = marker.corners.map((corner) => ({
      x: corner.x * scaleX,
      y: corner.y * scaleY,
    }));

    context.strokeStyle = "rgba(255, 0, 0, 1)";
    context.lineWidth = 2.5;
    context.beginPath();
    context.moveTo(corners[0].x, corners[0].y);
    for (let index = 1; index < corners.length; index += 1) {
      context.lineTo(corners[index].x, corners[index].y);
    }
    context.closePath();
    context.stroke();

    const labelX = Math.min(...corners.map((corner) => corner.x));
    const labelY = Math.min(...corners.map((corner) => corner.y));
    const distanceMeters = estimateDistanceMeters(
      marker.pose,
      ASSUMED_MARKER_SIZE_METERS,
    );
    const label = `${marker.id} ${distanceMeters.toFixed(2)}m`;
    context.font = "600 12px ui-monospace, SFMono-Regular, Menlo, monospace";
    context.textBaseline = "top";
    const textWidth = context.measureText(label).width;
    const boxX = Math.max(0, labelX);
    const boxY = Math.max(0, labelY - 20);

    context.fillStyle = "rgba(255, 0, 0, 0.9)";
    context.fillRect(boxX, boxY, textWidth + 10, 18);
    context.fillStyle = "#fff";
    context.fillText(label, boxX + 5, boxY + 3);

    drawAxis(
      context,
      scalePoint(marker.axis.origin, scaleX, scaleY),
      scalePoint(marker.axis.x, scaleX, scaleY),
      "#ff6a3d",
      3.4,
    );
    drawAxis(
      context,
      scalePoint(marker.axis.origin, scaleX, scaleY),
      scalePoint(marker.axis.y, scaleX, scaleY),
      "#14b87a",
      3.4,
    );
    drawAxis(
      context,
      scalePoint(marker.axis.origin, scaleX, scaleY),
      scalePoint(marker.axis.z, scaleX, scaleY),
      "#3b7cff",
      3.8,
    );
  }
}

function getViewportScene(viewport: HTMLCanvasElement | null) {
  if (!viewport) {
    return null;
  }

  const context = viewport.getContext("2d");
  if (!context) {
    return null;
  }

  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  const nextWidth = Math.round(width * dpr);
  const nextHeight = Math.round(height * dpr);

  if (viewport.width !== nextWidth || viewport.height !== nextHeight) {
    viewport.width = nextWidth;
    viewport.height = nextHeight;
  }

  return {
    context,
    dpr,
    height,
    viewportCanvas: viewport,
    width,
  };
}

function drawAxis(
  context: CanvasRenderingContext2D,
  start: Point,
  end: Point,
  strokeStyle: string,
  lineWidth: number,
) {
  context.strokeStyle = strokeStyle;
  context.lineWidth = lineWidth;
  context.beginPath();
  context.moveTo(start.x, start.y);
  context.lineTo(end.x, end.y);
  context.stroke();
}

function scalePoint(point: Point, scaleX: number, scaleY: number): Point {
  return {
    x: point.x * scaleX,
    y: point.y * scaleY,
  };
}

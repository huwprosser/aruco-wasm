import { CAMERA_INTRINSICS } from "./arucoConfig";

export interface Point {
  x: number;
  y: number;
}

export interface AxisProjection {
  origin: Point;
  x: Point;
  y: Point;
  z: Point;
}

export interface MarkerPose {
  quaternion: [number, number, number, number];
  translation: [number, number, number];
}

export interface DetectedMarker {
  confidence: number;
  id: number;
  corners: [Point, Point, Point, Point];
  axis: AxisProjection;
  pose: MarkerPose;
}

export interface DetectionResult {
  frameMs: number;
  markers: DetectedMarker[];
  processWidth: number;
}

export const MAX_DETECTOR_DIMENSION = 320;

type ArucoWasmExports = WebAssembly.Exports & {
  memory: WebAssembly.Memory;
  detector_new: () => number;
  detector_free: (detector: number) => void;
  detector_configure_frame: (
    detector: number,
    sourceWidth: number,
    sourceHeight: number,
  ) => number;
  detector_set_camera_intrinsics: (
    detector: number,
    focalLengthX: number,
    focalLengthY: number,
    principalX: number,
    principalY: number,
    focalLengthScale: number,
  ) => void;
  detector_prepare_rgba: (detector: number, length: number) => number;
  detector_set_input_size: (
    detector: number,
    inputWidth: number,
    inputHeight: number,
  ) => number;
  detector_detect: (
    detector: number,
    sourceWidth: number,
    sourceHeight: number,
  ) => void;
  detector_result_ptr: (detector: number) => number;
  detector_result_len: (detector: number) => number;
};

const RESULT_HEADER_LEN = 1;
const RESULT_MARKER_LEN = 25;

let wasmModulePromise: Promise<ArucoWasmExports> | null = null;

export async function createArucoDetector(): Promise<ArucoDetector> {
  const exports = await loadArucoWasm();
  const handle = exports.detector_new();

  if (!handle) {
    throw new Error("Failed to create the ArUco detector.");
  }

  exports.detector_set_camera_intrinsics(
    handle,
    CAMERA_INTRINSICS.focalLengthX ?? 0,
    CAMERA_INTRINSICS.focalLengthY ?? 0,
    CAMERA_INTRINSICS.principalX ?? 0,
    CAMERA_INTRINSICS.principalY ?? 0,
    CAMERA_INTRINSICS.focalLengthScale,
  );

  return new ArucoDetector(exports, handle);
}

export class ArucoDetector {
  private readonly canvas: OffscreenCanvas;
  private readonly context: OffscreenCanvasRenderingContext2D;
  private readonly exports: ArucoWasmExports;
  private readonly handle: number;

  private disposed = false;

  constructor(exports: ArucoWasmExports, handle: number) {
    this.exports = exports;
    this.handle = handle;

    if (typeof OffscreenCanvas === "undefined") {
      throw new Error("OffscreenCanvas is required for worker detection.");
    }

    this.canvas = new OffscreenCanvas(1, 1);
    const context = this.canvas.getContext("2d", {
      alpha: false,
      desynchronized: true,
      willReadFrequently: true,
    }) as OffscreenCanvasRenderingContext2D | null;

    if (!context) {
      throw new Error("2D canvas is required for the ArUco detector.");
    }

    context.imageSmoothingEnabled = false;
    this.context = context;
  }

  detect(
    source: ImageBitmap,
    sourceWidth: number,
    sourceHeight: number,
  ): DetectionResult {
    this.assertActive();
    const pipelineStartedAt = performance.now();
    const inputHeight = source.height;
    const inputWidth = source.width;

    if (!sourceWidth || !sourceHeight || !inputWidth || !inputHeight) {
      return emptyDetectionResult();
    }

    const desiredPackedSize = this.exports.detector_configure_frame(
      this.handle,
      sourceWidth,
      sourceHeight,
    );
    let processWidth = desiredPackedSize & 0xffff;
    let processHeight = desiredPackedSize >>> 16;

    if (inputWidth <= processWidth && inputHeight <= processHeight) {
      const inputPackedSize = this.exports.detector_set_input_size(
        this.handle,
        inputWidth,
        inputHeight,
      );
      processWidth = inputPackedSize & 0xffff;
      processHeight = inputPackedSize >>> 16;
    }

    if (!processWidth || !processHeight) {
      return emptyDetectionResult();
    }

    if (
      this.canvas.width !== processWidth ||
      this.canvas.height !== processHeight
    ) {
      this.canvas.width = processWidth;
      this.canvas.height = processHeight;
    }

    this.context.drawImage(source, 0, 0, processWidth, processHeight);
    const rgba = this.context.getImageData(
      0,
      0,
      processWidth,
      processHeight,
    ).data;
    const rgbaPtr = this.exports.detector_prepare_rgba(
      this.handle,
      rgba.length,
    );
    new Uint8Array(this.exports.memory.buffer, rgbaPtr, rgba.length).set(rgba);
    this.exports.detector_detect(this.handle, sourceWidth, sourceHeight);

    return parseDetectionResult(
      this.exports,
      this.handle,
      performance.now() - pipelineStartedAt,
      processWidth,
    );
  }

  dispose() {
    if (this.disposed) {
      return;
    }

    this.exports.detector_free(this.handle);
    this.disposed = true;
  }

  private assertActive() {
    if (this.disposed) {
      throw new Error("The ArUco detector has already been disposed.");
    }
  }
}

async function loadArucoWasm() {
  if (!wasmModulePromise) {
    wasmModulePromise = instantiateArucoWasm();
  }

  return wasmModulePromise;
}

async function instantiateArucoWasm(): Promise<ArucoWasmExports> {
  const moduleUrl = new URL("./wasm/aruco_core.wasm", import.meta.url);
  const imports: WebAssembly.Imports = {
    env: {
      wasm_now: () => performance.now(),
    },
  };
  const response = await fetch(moduleUrl);

  if ("instantiateStreaming" in WebAssembly) {
    try {
      const result = await WebAssembly.instantiateStreaming(
        response.clone(),
        imports,
      );
      return result.instance.exports as ArucoWasmExports;
    } catch {
      // Some dev servers still serve WASM with the wrong MIME type.
    }
  }

  const bytes = await response.arrayBuffer();
  const result = await WebAssembly.instantiate(bytes, imports);
  return result.instance.exports as ArucoWasmExports;
}

function emptyDetectionResult(
  processWidth = 0,
  frameMs = 0,
): DetectionResult {
  return {
    frameMs,
    markers: [],
    processWidth,
  };
}

function parseDetectionResult(
  exports: ArucoWasmExports,
  handle: number,
  frameMs: number,
  processWidth: number,
): DetectionResult {
  const resultPtr = exports.detector_result_ptr(handle);
  const resultLen = exports.detector_result_len(handle);

  if (!resultPtr || resultLen < RESULT_HEADER_LEN) {
    return emptyDetectionResult(processWidth, frameMs);
  }

  const values = new Float64Array(exports.memory.buffer, resultPtr, resultLen);
  const markerCount = Math.max(0, Math.floor(values[0] ?? 0));
  const markers: DetectedMarker[] = [];

  for (let markerIndex = 0; markerIndex < markerCount; markerIndex += 1) {
    const offset = RESULT_HEADER_LEN + markerIndex * RESULT_MARKER_LEN;
    if (offset + RESULT_MARKER_LEN > values.length) {
      break;
    }

    markers.push({
      confidence: values[offset + 1],
      id: values[offset],
      corners: [
        { x: values[offset + 2], y: values[offset + 3] },
        { x: values[offset + 4], y: values[offset + 5] },
        { x: values[offset + 6], y: values[offset + 7] },
        { x: values[offset + 8], y: values[offset + 9] },
      ],
      axis: {
        origin: {
          x: values[offset + 10],
          y: values[offset + 11],
        },
        x: {
          x: values[offset + 12],
          y: values[offset + 13],
        },
        y: {
          x: values[offset + 14],
          y: values[offset + 15],
        },
        z: {
          x: values[offset + 16],
          y: values[offset + 17],
        },
      },
      pose: {
        translation: [
          values[offset + 18],
          values[offset + 19],
          values[offset + 20],
        ],
        quaternion: [
          values[offset + 21],
          values[offset + 22],
          values[offset + 23],
          values[offset + 24],
        ],
      },
    });
  }

  return {
    frameMs,
    markers,
    processWidth,
  };
}

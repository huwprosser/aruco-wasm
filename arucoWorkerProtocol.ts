import type { DetectionResult } from "./aruco";

export type DetectRequestMessage = {
  frame: ImageBitmap;
  seq: number;
  sourceHeight: number;
  sourceWidth: number;
  token: number;
  type: "detect";
};

export type WorkerResultMessage = {
  result: DetectionResult;
  seq: number;
  token: number;
  type: "result";
};

export type WorkerErrorMessage = {
  message: string;
  token: number;
  type: "error";
};

export type WorkerMessage = WorkerErrorMessage | WorkerResultMessage;

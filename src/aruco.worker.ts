/// <reference lib="webworker" />

import { createArucoDetector } from "./aruco";
import type {
  DetectRequestMessage,
  WorkerErrorMessage,
  WorkerResultMessage,
} from "./arucoWorkerProtocol";

const detectorPromise = createArucoDetector();
const workerScope = self as DedicatedWorkerGlobalScope;

workerScope.onmessage = (event: MessageEvent<DetectRequestMessage>) => {
  void handleDetectRequest(event.data);
};

async function handleDetectRequest(message: DetectRequestMessage) {
  const { frame, seq, sourceHeight, sourceWidth, token } = message;

  try {
    const detector = await detectorPromise;
    const result = detector.detect(frame, sourceWidth, sourceHeight);
    postWorkerMessage({
      result,
      seq,
      token,
      type: "result",
    });
  } catch (error) {
    postWorkerMessage({
      message:
        error instanceof Error ? error.message : "Worker detection failed.",
      token,
      type: "error",
    });
  } finally {
    frame.close();
  }
}

function postWorkerMessage(message: WorkerErrorMessage | WorkerResultMessage) {
  workerScope.postMessage(message);
}

export {};

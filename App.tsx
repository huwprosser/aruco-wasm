import { useEffect } from "react";
import "./App.css";
import {
  classifyProximity,
  estimateDistanceMeters,
} from "./arucoAnchor";
import { ASSUMED_MARKER_SIZE_METERS } from "./arucoConfig";
import { useArucoCamera } from "./useArucoCamera";

export default function App() {
  const { aspectRatio, errorMessage, frameMs, markers, videoRef, viewportRef } =
    useArucoCamera();

  useEffect(() => {
    if (!markers.length) {
      return;
    }

    console.table(
      markers.map((marker) => {
        const distanceMeters = estimateDistanceMeters(
          marker.pose,
          ASSUMED_MARKER_SIZE_METERS,
        );
        return {
          confidence: roundValue(marker.confidence),
          distanceMeters: roundValue(distanceMeters),
          id: marker.id,
          proximity: classifyProximity(distanceMeters),
        };
      }),
    );
  }, [markers]);

  return (
    <main className="app">
      <div className="viewport" style={{ aspectRatio }}>
        <video
          ref={videoRef}
          autoPlay
          className="camera-source"
          muted
          playsInline
        />
        <canvas ref={viewportRef} className="camera-output" />
        <p className="frame-ms">{formatFrameMs(frameMs)}</p>
        {errorMessage ? (
          <p className="error-banner" role="alert">
            {errorMessage}
          </p>
        ) : null}
      </div>
    </main>
  );
}

function formatFrameMs(value: number) {
  return (value > 0 ? value.toFixed(1) : "--") + " ms";
}

function roundValue(value: number) {
  return Number(value.toFixed(1));
}

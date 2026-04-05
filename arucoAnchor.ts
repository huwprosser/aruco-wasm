import type { DetectedMarker, MarkerPose } from "./aruco";

export interface BoardMarkerDefinition {
  id: number;
  position: [number, number, number];
  quaternion?: [number, number, number, number];
}

export interface MarkerBoardDefinition {
  id: string;
  markers: BoardMarkerDefinition[];
}

export interface BoardPose {
  boardId: string;
  confidence: number;
  position: [number, number, number];
  quaternion: [number, number, number, number];
  visibleMarkerIds: number[];
}

export type ProximityBand = "near" | "medium" | "far";

export function estimateDistanceMeters(
  pose: MarkerPose,
  markerSizeMeters: number,
) {
  return pose.translation[2] * markerSizeMeters;
}

export function classifyProximity(distanceMeters: number): ProximityBand {
  if (distanceMeters < 0.45) {
    return "near";
  }

  if (distanceMeters < 1.1) {
    return "medium";
  }

  return "far";
}

export function solveBoardPose(
  markers: readonly DetectedMarker[],
  board: MarkerBoardDefinition,
) {
  const visible = markers.filter((marker) =>
    board.markers.some((definition) => definition.id === marker.id),
  );

  if (!visible.length) {
    return null;
  }

  let weightSum = 0;
  let positionX = 0;
  let positionY = 0;
  let positionZ = 0;
  let quaternion = visible[0].pose.quaternion;

  for (const marker of visible) {
    const weight = Math.max(0.0001, marker.confidence);
    const [x, y, z] = marker.pose.translation;
    positionX += x * weight;
    positionY += y * weight;
    positionZ += z * weight;
    quaternion = averageQuaternion(quaternion, marker.pose.quaternion, weightSum, weight);
    weightSum += weight;
  }

  return {
    boardId: board.id,
    confidence: weightSum / visible.length,
    position: [
      positionX / weightSum,
      positionY / weightSum,
      positionZ / weightSum,
    ] as [number, number, number],
    quaternion,
    visibleMarkerIds: visible.map((marker) => marker.id),
  } satisfies BoardPose;
}

function averageQuaternion(
  current: [number, number, number, number],
  next: [number, number, number, number],
  currentWeight: number,
  nextWeight: number,
): [number, number, number, number] {
  let candidate = next;
  const dot =
    current[0] * next[0] +
    current[1] * next[1] +
    current[2] * next[2] +
    current[3] * next[3];

  if (dot < 0) {
    candidate = [-next[0], -next[1], -next[2], -next[3]];
  }

  const totalWeight = currentWeight + nextWeight;
  const blended: [number, number, number, number] = [
    (current[0] * currentWeight + candidate[0] * nextWeight) / totalWeight,
    (current[1] * currentWeight + candidate[1] * nextWeight) / totalWeight,
    (current[2] * currentWeight + candidate[2] * nextWeight) / totalWeight,
    (current[3] * currentWeight + candidate[3] * nextWeight) / totalWeight,
  ];

  return normalizeQuaternion(blended);
}

function normalizeQuaternion(
  quaternion: [number, number, number, number],
): [number, number, number, number] {
  const length = Math.hypot(
    quaternion[0],
    quaternion[1],
    quaternion[2],
    quaternion[3],
  );

  if (!length) {
    return [0, 0, 0, 1];
  }

  return [
    quaternion[0] / length,
    quaternion[1] / length,
    quaternion[2] / length,
    quaternion[3] / length,
  ];
}

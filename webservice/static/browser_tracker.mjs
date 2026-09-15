import {FilesetResolver, HandLandmarker, PoseLandmarker} from './vendor/mediapipe/vision_bundle.mjs';

export async function createLandmarkers(delegate = 'CPU') {
  const vision = await FilesetResolver.forVisionTasks('vendor/mediapipe/wasm');
  const hand = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {modelAssetPath: 'vendor/mediapipe/hand_landmarker.task', delegate},
    runningMode: 'VIDEO', numHands: 2,
  });
  try {
    const pose = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {modelAssetPath: 'vendor/mediapipe/pose_landmarker_lite.task', delegate},
      runningMode: 'VIDEO', numPoses: 1,
    });
    return [hand, pose];
  } catch (error) {
    hand.close();
    throw error;
  }
}

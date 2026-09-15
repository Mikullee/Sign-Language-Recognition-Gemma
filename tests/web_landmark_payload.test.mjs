import test from 'node:test';
import assert from 'node:assert/strict';
import { packLandmarks } from '../webservice/static/landmark_payload.mjs';

test('camera/replay packaging preserves xyz and visibility without mirroring', () => {
  const pose = Array.from({length: 33}, () => ({x: .123456, y: .4, z: -.1, visibility: .8, presence: .7}));
  const hand = Array.from({length: 21}, () => ({x: .2, y: .3, z: 0}));
  const result = packLandmarks({landmarks: [hand], handednesses: [[{categoryName: 'Left', score: .9}]]}, {landmarks: [pose]});
  assert.deepEqual(result.pose.landmarks[0], [.12346, .4, -.1]);
  assert.equal(result.pose.visibility[0], .7);
  assert.equal(result.hands[0].handedness, 'Left');
  assert.equal(result.hands[0].landmarks[0][0], .2);
});

test('unknown visibility is not silently treated as visible', () => {
  const pose = Array.from({length: 33}, () => ({x: .2, y: .3, z: 0}));
  assert.equal(packLandmarks({}, {landmarks: [pose]}).pose.visibility[0], 0);
  assert.deepEqual(packLandmarks({}, {}), {pose: null, hands: []});
});

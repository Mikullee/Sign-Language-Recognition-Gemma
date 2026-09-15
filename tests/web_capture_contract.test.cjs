const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const {verifyAssetBytes, servedAssetURL} = require('../scripts/capture_web_trigger_videos.cjs');

test('served assets must match actual response bytes, not just local metadata', () => {
  const data = Buffer.from('expected model');
  const hash = crypto.createHash('sha256').update(data).digest('hex');
  assert.doesNotThrow(() => verifyAssetBytes(data, hash, 'model'));
  assert.throws(() => verifyAssetBytes(Buffer.from('different model'), hash, 'model'), /mismatch/);
});
test('asset manifest maps to exactly the same browser URLs', () => {
  assert.equal(servedAssetURL('webservice/static/replay.mjs'), '/replay.mjs');
  assert.equal(servedAssetURL('webservice/vendor/mediapipe/a.task'), '/vendor/mediapipe/a.task');
  assert.throws(() => servedAssetURL('private/video.mp4'), /asset/);
});

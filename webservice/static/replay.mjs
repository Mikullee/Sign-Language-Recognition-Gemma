import {createLandmarkers} from './browser_tracker.mjs';
import {packLandmarks} from './landmark_payload.mjs';

const $ = id => document.getElementById(id);
let busy = false, lastCapture = null;
const waitEvent = (target, name) => new Promise((resolve, reject) => {
  const timer = setTimeout(() => {cleanup(); reject(new Error(name + ' timed out'));}, 15000);
  const done = () => {cleanup(); resolve();};
  const error = () => {cleanup(); reject(new Error('video decode failed'));};
  const cleanup = () => {clearTimeout(timer); target.removeEventListener(name, done); target.removeEventListener('error', error);};
  target.addEventListener(name, done, {once: true}); target.addEventListener('error', error, {once: true});
});

export async function captureForEvaluation(options = {}) {
  if (busy) throw new Error('A replay is already running');
  const file = $('file').files[0];
  if (!file) throw new Error('Select a local video');
  const fps = Number(options.fps || $('fps').value), speed = Number(options.speed || 1);
  if (!(fps >= 1 && fps <= 60 && speed >= .25 && speed <= 4)) throw new Error('Invalid FPS/speed');
  busy = true;
  const url = URL.createObjectURL(file), video = $('source'), canvas = $('frame');
  let hand, pose;
  try {
    const loaded = waitEvent(video, 'loadeddata');
    video.src = url; video.load(); await loaded;
    if (!Number.isFinite(video.duration) || video.duration > 180) throw new Error('Video must be <= 180 seconds');
    [hand, pose] = await createLandmarkers('CPU');
    canvas.width = Math.min(video.videoWidth, 960);
    canvas.height = Math.round(video.videoHeight * canvas.width / video.videoWidth);
    const ctx = canvas.getContext('2d'), frames = [], duration = video.duration / speed;
    let previousTimestamp = -1;
    for (let i = 0; i / fps < duration - .002; i++) {
      const sourceTime = i / fps * speed;
      if (Math.abs(video.currentTime - sourceTime) > .00001) {
        const sought = waitEvent(video, 'seeked'); video.currentTime = sourceTime; await sought;
      }
      const scale = Number(options.scale || 1), dx = Number(options.dx || 0), dy = Number(options.dy || 0);
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.save(); ctx.translate(canvas.width * (.5 + dx), canvas.height * (.5 + dy)); ctx.scale(scale, scale);
      ctx.drawImage(video, -canvas.width / 2, -canvas.height / 2, canvas.width, canvas.height); ctx.restore();
      if (options.crop_lower) ctx.fillRect(0, canvas.height * .7, canvas.width, canvas.height * .3);
      const jitter = Number(options.jitter || 0) * Math.sin(i * 2.399 + Number(options.seed || 42));
      const timestamp = Math.max(previousTimestamp + .001, Math.max(0, i / fps + jitter));
      previousTimestamp = timestamp;
      const packed = packLandmarks(hand.detectForVideo(canvas, Math.round(timestamp * 1000)),
                                    pose.detectForVideo(canvas, Math.round(timestamp * 1000)));
      if (options.hide_knees && packed.pose) packed.pose.visibility[25] = packed.pose.visibility[26] = 0;
      if (options.drop_hands && (i / fps) % 2 < .18) {
        packed.hands = [];
        if (packed.pose) packed.pose.visibility[15] = packed.pose.visibility[16] = 0;
      }
      frames.push({...packed, timestamp});
      if (i % 15 === 0) $('status').textContent = `擷取 ${options.name || '原片'}：${(i / fps).toFixed(1)} / ${duration.toFixed(1)} 秒`;
    }
    return {schema_version: 1, mode: 'VIDEO', delegate: 'CPU', source_duration: video.duration,
            width: canvas.width, height: canvas.height, options, frames};
  } finally {
    hand?.close(); pose?.close(); URL.revokeObjectURL(url); busy = false;
  }
}
window.captureForEvaluation = captureForEvaluation;

export async function runAutoReplay(options = {}) {
  const capture = await captureForEvaluation(options);
  const session = 'replay-' + crypto.randomUUID(), responses = [];
  for (let i = 0; i < capture.frames.length; i += 6) {
    const response = await fetch('/stream', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session, reset: i === 0, frames: capture.frames.slice(i, i + 6),
                            eof: i + 6 >= capture.frames.length})});
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.message || 'Stream failed');
    responses.push(result);
  }
  lastCapture = {...capture, responses};
  $('save').disabled = false;
  $('result').textContent = JSON.stringify({events: responses.flatMap(r => r.events), final: responses.at(-1)}, null, 2);
  $('status').textContent = '回放完成；沒有以補幀方式完成 EOF。';
  return lastCapture;
}
window.runAutoReplay = runAutoReplay;
$('run').onclick = async () => {
  $('run').disabled = true;
  try { await runAutoReplay(); } catch (e) { $('status').textContent = e.message; }
  finally { $('run').disabled = false; }
};
$('save').onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastCapture)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'knee42-private-replay.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

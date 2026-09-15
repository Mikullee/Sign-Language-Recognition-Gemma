import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

// Execute the shipped functions, stubbing only the camera, network and DOM.
// innerHTML/textContent deliberately share storage, as on a real element.
const html = fs.readFileSync(new URL('../webservice/static/index.html', import.meta.url), 'utf8');
function between(start, end) {
  const a = html.indexOf(start), b = html.indexOf(end, a);
  assert(a >= 0 && b > a);
  return html.slice(a, b);
}
function element() {
  let content = '';
  return {
    set textContent(value) { content = String(value); },
    get textContent() { return content.replace(/<[^>]*>/g, ''); },
    set innerHTML(value) { content = String(value); },
    get innerHTML() { return content; },
    classList: {toggle() {}},
  };
}
function runtime(mode) {
  const elements = new Map();
  const $ = id => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); };
  const context = vm.createContext({$, camMode: mode, camOn: true,
    requestAnimationFrame() {}, video: {readyState: 4, currentTime: 1},
    lastVideoTime: -1, lastTick: 0, fpsEMA: 0, lastMirror: '固定鏡像慣例',
    handLm: {detectForVideo() {return {};}}, poseLm: {detectForVideo() {return {};}},
    packLandmarks() {return {hands: [], pose: null};}, draw() {},
    autoBuf: [], autoT0: 0, flushAuto() {}, recording: false,
    badge: element(), camMsg: element(), say() {}, console,
  });
  vm.runInContext(between('function loop(now)', 'function draw(')
    + between('const AUTO_STATE_TEXT', 'function renderAutoResult('), context);
  return {context, $};
}

test('a camera frame must not erase the latest automatic diagnostics', () => {
  const {context, $} = runtime('auto');
  context.renderAutoState({calibrated: false, rest_distance: null,
    rest_signature_status: 'waiting_visible_knees', hands_detected: 0, reference_revision: 0});
  const status = $('s-mirror').innerHTML;
  assert.match(status, /膝蓋/);
  for (let i = 0; i < 30; i++) {
    context.video.currentTime += 1 / 30;
    context.loop(1000 + i * 1000 / 30);
    assert.equal($('s-mirror').innerHTML, status);
  }
});

test('manual mode still updates the mirror diagnostic each camera frame', () => {
  const {context, $} = runtime('manual');
  $('s-mirror').innerHTML = 'previous automatic diagnostics';
  context.loop(1000);
  assert.equal($('s-mirror').textContent, '固定鏡像慣例');
});

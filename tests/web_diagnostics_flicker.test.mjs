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
    dataset: {},
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
    badge: element(), camMsg: element(), say(el,text) {el.innerHTML=text;}, console, autoSeen:0,
  });
  vm.runInContext(between('function loop(now)', 'function draw(')
    + between('const AUTO_STATE_TEXT', 'function startRec('), context);
  return {context, $};
}

test('a warning persists during recovery but clears after a new reference',()=>{
  const {context}=runtime('auto');
  const status={calibrated:true,reference_revision:2,state:'FORCED_FINALIZE_COOLDOWN',
    last_message:'timeout_finalize'};
  context.renderAutoState(status);
  assert.match(context.camMsg.textContent,/未正常完成/);
  context.renderAutoState({...status,last_message:'',state:'REARMING'});
  assert.match(context.camMsg.textContent,/未正常完成/);
  context.renderAutoState({...status,last_message:'',state:'IDLE_BLANK',reference_revision:3});
  assert.doesNotMatch(context.camMsg.textContent,/未正常完成/);
  assert.match(context.camMsg.textContent,/重新定位/);
});

test('a later recognition result clears the previous segment warning',()=>{
  const {context}=runtime('auto');
  context.renderAutoState({calibrated:true,reference_revision:1,state:'REARMING',last_message:'tracking_gap'});
  context.renderAutoResult({index:2,start:3,end:4,frames:30,reason:'visible_rest_finalize',top:[{text:'測試',prob:.8}]});
  assert.doesNotMatch(context.camMsg.textContent,/未正常完成/);
});

test('a batch containing a failure followed by completed recovery is not stuck on an old warning',()=>{
  const {context}=runtime('auto');
  context.renderAutoState({calibrated:true,reference_revision:2,state:'IDLE_BLANK',last_message:'timeout_finalize'});
  assert.doesNotMatch(context.camMsg.textContent,/未正常完成/);
  assert.match(context.camMsg.textContent,/重新定位/);
});

test('a later failure wins over an older accepted result in the same batch',async()=>{
  const {context}=runtime('auto');
  Object.assign(context,{autoSending:false,autoBuf:[{}],autoSession:'mixed',
    fetch:async()=>({json:async()=>({ok:true,calibrated:true,reference_revision:2,
      state:'REARMING',last_message:'timeout_finalize',results:[{index:1,start:1,end:3,
        frames:60,reason:'visible_rest_finalize',top:[{text:'測試',prob:.8}]}]})})});
  vm.runInContext(between('async function flushAuto(', 'const AUTO_STATE_TEXT'),context);
  await context.flushAuto();
  assert.match(context.camMsg.textContent,/未正常完成/);
});

test('live duration and timeout warning use the actual server limit',()=>{
  const {context,$}=runtime('auto');
  context.renderAutoState({calibrated:true,state:'SIGNING_ACTIVE',segment_elapsed_sec:6.2,segment_limit_sec:10});
  assert.equal($('auto-duration').textContent,'6.2 / 10 秒');
  context.renderAutoState({calibrated:true,state:'REARMING',segment_limit_sec:10,last_message:'timeout_finalize'});
  assert.match(context.camMsg.textContent,/10 秒/);
});

test('a camera frame must not erase the latest automatic diagnostics', () => {
  const {context, $} = runtime('auto');
  context.renderAutoState({calibrated: false, rest_distance: null,
    rest_signature_status: 'waiting_visible_knees', hands_detected: 0, reference_revision: 0,
    calibration_blockers:['knees_not_visible']});
  const status = $('auto-reason').textContent;
  assert.match(status, /膝蓋/);
  for (let i = 0; i < 30; i++) {
    context.video.currentTime += 1 / 30;
    context.loop(1000 + i * 1000 / 30);
    assert.equal($('auto-reason').textContent, status);
  }
});

test('manual mode still updates the mirror diagnostic each camera frame', () => {
  const {context, $} = runtime('manual');
  $('s-mirror').innerHTML = 'previous automatic diagnostics';
  context.loop(1000);
  assert.equal($('s-mirror').textContent, '固定鏡像慣例');
});

test('automatic diagnostics have separate immediate fields instead of a rewritten paragraph', () => {
  const {context,$}=runtime('auto');
  $('s-mirror').textContent='固定鏡像慣例';
  const state={calibrated:false,rest_distance:null,rest_signature_status:'waiting_knee_calibration',
    hands_detected:2,reference_revision:0,rest_motion_score:.09,rest_motion_threshold:.15,
    calibration_blockers:['not_on_knees'],calibration_hold_sec:0,calibration_target_sec:1,wrists_trusted:true};
  context.renderAutoState(state);
  assert.equal($('s-mirror').textContent,'固定鏡像慣例');
  assert.match($('auto-reason').textContent,/膝蓋/);
  assert.match($('auto-distance').textContent,/未建立基準/);
  assert.equal($('auto-motion').textContent,'0.09 / 0.15');
  assert.doesNotMatch($('auto-reason').textContent,/算不出來/);
  context.renderAutoState({...state,calibration_blockers:[],rest_motion_score:.04,calibration_hold_sec:.6});
  assert.equal($('auto-motion').textContent,'0.04 / 0.15');
  assert.match($('auto-hold').textContent,/0.60/);
  assert.match($('auto-reason').textContent,/保持/);
});

for (const failed of [false,true]) {
  test(`late stream ${failed?'error':'response'} cannot restore automatic UI after a mode change`,async()=>{
    let release;
    let renders=0,modeChanges=0;
    const context=vm.createContext({autoSending:false,camOn:true,camMode:'auto',autoBuf:[{}],autoSession:'before',
      fetch:()=>new Promise(resolve=>{release=resolve;}),
      renderAutoState:()=>renders++,renderAutoResult:()=>renders++,
      say:()=>renders++,camMsg:{},setMode:()=>modeChanges++});
    vm.runInContext(between('async function flushAuto(', 'const AUTO_STATE_TEXT'),context);
    const pending=context.flushAuto();
    context.camMode='manual';context.autoSession='after';
    release({json:async()=>{if(failed)throw new Error('old connection failed');return {ok:true,results:[]};}});
    await pending;
    assert.equal(renders,0);assert.equal(modeChanges,0);
  });
}

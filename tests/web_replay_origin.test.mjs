import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {assertLocalReplayOrigin} from '../webservice/static/replay_origin.mjs';

test('private replay allows only explicit HTTP(S) loopback origins', () => {
  for (const url of ['http://127.0.0.1:8642/', 'https://localhost:8642/', 'http://[::1]:8642/']) {
    assert.doesNotThrow(() => assertLocalReplayOrigin(url));
  }
  for (const url of ['https://example.com/', 'http://192.168.1.2:8642/',
                     'file:///replay.html', 'https://localhost.example.com/', 'http://user@localhost/']) {
    assert.throws(() => assertLocalReplayOrigin(url), /本機/);
  }
});

test('the page gates loading replay before it can read private files', () => {
  const html = fs.readFileSync(new URL('../webservice/static/replay.html', import.meta.url), 'utf8');
  const guard = html.indexOf('assertLocalReplayOrigin(location.href)');
  const loader = html.indexOf("await import('./replay.mjs')");
  assert(guard >= 0 && loader > guard);
  assert(!/src=["'](?:\.\/)?replay\.mjs/.test(html));
});

/* Local development/evaluation runner. Raw videos are never uploaded. */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

function verifyAssetBytes(bytes, expected, asset) {
  const actual = crypto.createHash('sha256').update(bytes).digest('hex');
  if (actual !== expected) throw new Error('Served asset SHA-256 mismatch: ' + asset);
}
function servedAssetURL(asset) {
  if (asset.includes('..')) throw new Error('Invalid asset path');
  if (asset.startsWith('webservice/static/')) return '/' + asset.slice('webservice/static/'.length);
  if (asset.startsWith('webservice/vendor/')) return '/' + asset.slice('webservice/'.length);
  throw new Error('Unsupported asset path: ' + asset);
}

async function main() {
  const [manifestPath, baseURL = 'http://127.0.0.1:8642'] = process.argv.slice(2);
  if (!manifestPath) throw new Error('Usage: node scripts/capture_web_trigger_videos.cjs MANIFEST [URL]');
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const url = new URL(baseURL);
  if (!['http:', 'https:'].includes(url.protocol) || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)) throw new Error('Replay must use a loopback HTTP service');
  const {chromium} = require('playwright');
  const browser = await chromium.launch({headless: true,
    ...(process.env.PLAYWRIGHT_CHANNEL ? {channel: process.env.PLAYWRIGHT_CHANNEL} : {})});
  try {
    const page = await browser.newPage({ignoreHTTPSErrors: true});
    page.setDefaultTimeout(30000);
    for (const [asset, expected] of Object.entries(manifest.provenance.asset_sha256)) {
      const response = await page.request.get(baseURL + servedAssetURL(asset));
      if (!response.ok()) throw new Error('Cannot verify served asset: ' + asset);
      verifyAssetBytes(await response.body(), expected, asset);
    }
    console.log('Verified all served JS/WASM/model assets; browser ' + browser.version());
    await page.goto(baseURL + '/replay.html');
    await page.waitForFunction(() => typeof window.captureForEvaluation === 'function');
    for (const item of manifest.items) {
      if (fs.existsSync(item.output)) {
        const prior = JSON.parse(fs.readFileSync(item.output, 'utf8'));
        if (prior.browser_version === browser.version() && prior.asset_verification === 'served-sha256-v1') {
          console.log('cached ' + item.id); continue;
        }
      }
      await page.locator('#file').setInputFiles(item.source_path);
      const started = Date.now();
      const capture = await page.evaluate(options => window.captureForEvaluation(options), item.options);
      capture.cache_key = item.id;
      capture.source_id = item.source_id;
      capture.provenance = manifest.provenance;
      capture.browser_version = browser.version();
      capture.asset_verification = 'served-sha256-v1';
      fs.mkdirSync(path.dirname(item.output), {recursive: true});
      fs.writeFileSync(item.output, JSON.stringify(capture));
      console.log(`${item.id.slice(0, 12)} ${item.options.name}: ${capture.frames.length} frames ${(Date.now() - started) / 1000}s`);
    }
  } finally {await browser.close();}
}
module.exports = {verifyAssetBytes, servedAssetURL};
if (require.main === module) main().catch(e => {console.error(e); process.exitCode = 1;});

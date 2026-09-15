/** Private-file replay is intentionally unavailable on LAN/public origins. */
export function assertLocalReplayOrigin(value) {
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol)
      || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)
      || url.username || url.password) {
    throw new Error('私人影片回放僅限本機網址：請改用 http://127.0.0.1:8642/replay.html');
  }
}

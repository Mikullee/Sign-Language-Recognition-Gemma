// One serialization contract for live camera and recorded-video replay.
const round = (n, digits = 5) => Number.isFinite(n) ? +n.toFixed(digits) : null;
const xyz = points => points.map(p => [round(p.x), round(p.y), round(p.z)]);

export function packLandmarks(hr, pr) {
  const hands = (hr.landmarks || []).map((points, i) => ({
    handedness: hr.handednesses?.[i]?.[0]?.categoryName || 'Unknown',
    score: round(hr.handednesses?.[i]?.[0]?.score ?? 0, 3),
    landmarks: xyz(points),
  }));
  const points = pr.landmarks?.[0];
  const pose = points ? {landmarks: xyz(points), visibility: points.map(p => {
    const visibility = Number.isFinite(p.visibility) ? p.visibility : 0;
    return round(Math.min(visibility, Number.isFinite(p.presence) ? p.presence : 1));
  })} : null;
  return {pose, hands};
}

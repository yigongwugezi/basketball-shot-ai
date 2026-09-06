const assert = require("node:assert/strict");
const ui = require("../shot-analyzer-prototype/ui-model.js");

const qualified = {
  release_time: 1.73,
  trusted_flight: { temporal_span_ms: 420, point_count: 14 },
  release_state: {
    qualification: "HIGH",
    release_time: 1.73,
    speed_mps: 7.9,
    elevation_angle_deg: 51.2,
  },
};
const high = ui.analysisSummary(qualified, null);
assert.equal(high.qualified, true);
assert.equal(high.level, "HIGH");
assert.equal(high.speed, 7.9);
assert.equal(high.angle, 51.2);
assert.equal(high.observations, 14);

const unqualified = {
  release_state: {
    qualification: "UNQUALIFIED",
    speed_mps: 99,
    elevation_angle_deg: 99,
    reason: "insufficient temporal support",
  },
};
const hidden = ui.analysisSummary(unqualified, { detection_frame_count: 3 });
assert.equal(hidden.qualified, false);
assert.equal(hidden.speed, null);
assert.equal(hidden.angle, null);
assert.equal(hidden.failure.title, "篮球连续轨迹不足");

const emptyTrack = ui.trackStatus(
  { status: "ok", requested_frame_count: 19, detection_frame_count: 0, missing_frame_count: 19 },
  unqualified,
);
assert.equal(emptyTrack.label, "未形成有效飞行轨迹");

const partialTrack = ui.trackStatus(
  { status: "ok", requested_frame_count: 19, detection_frame_count: 5, missing_frame_count: 14 },
  unqualified,
);
assert.equal(partialTrack.label, "连续轨迹不足");

console.log("P-UI-V2 UI model checks: PASS");

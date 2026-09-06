const videoInput = document.getElementById("videoInput");
const analyzeButton = document.getElementById("analyzeButton");
const resetButton = document.getElementById("resetButton");
const videoPreview = document.getElementById("videoPreview");
const emptyPreview = document.getElementById("emptyPreview");
const qualityList = document.getElementById("qualityList");
const pipelineList = document.getElementById("pipelineList");
const framesGrid = document.getElementById("framesGrid");
const releaseBallEvidence = document.getElementById("releaseBallEvidence");
const ballTrackEvidence = document.getElementById("ballTrackEvidence");
const releaseFusionEvidence = document.getElementById("releaseFusionEvidence");
const metricsList = document.getElementById("metricsList");
const imageModal = document.getElementById("imageModal");
const imageModalImg = document.getElementById("imageModalImg");
const imageModalClose = document.getElementById("imageModalClose");

const pipelineSteps = [
  "读取视频元数据",
  "检查拍摄质量",
  "搜索投篮关键帧",
  "生成人体骨架和检测结果",
  "生成第一版姿态报告",
];

const framePlan = [
  { key: "setup", label: "准备", ratio: 0.14 },
  { key: "dip", label: "下沉", ratio: 0.32 },
  { key: "release", label: "出手", ratio: 0.5 },
  { key: "follow_through", label: "随球", ratio: 0.64 },
  { key: "landing", label: "落地", ratio: 0.78 },
];

let videoUrl = null;
let selectedFile = null;
let activeFrameIndex = 0;
let renderedFrames = [];

function setPipeline(doneCount = 0) {
  pipelineList.innerHTML = pipelineSteps
    .map((step, index) => {
      const done = index < doneCount ? "done" : "";
      return `<li class="${done}">${step}</li>`;
    })
    .join("");
}

function statusLabel(state, label) {
  return `<span class="status ${state}">${label}</span>`;
}

function renderQuality(items) {
  qualityList.innerHTML = items
    .map(
      (item) => `
      <div class="quality-item">
        <div>
          <strong>${item.title}</strong>
          <small>${item.detail}</small>
        </div>
        ${statusLabel(item.state, item.label)}
      </div>
    `,
    )
    .join("");
}

function renderMetrics(metrics) {
  metricsList.innerHTML = metrics
    .map(
      (metric) => `
      <div class="metric-item">
        <div>
          <strong>${metric.title}</strong>
          <small>${metric.detail}</small>
        </div>
        ${statusLabel(metric.state, metric.label)}
      </div>
    `,
    )
    .join("");
}

function getVideoQuality(video) {
  const duration = video.duration || 0;
  const width = video.videoWidth || 0;
  const height = video.videoHeight || 0;
  const landscape = width >= height;
  const shortEnough = duration >= 3 && duration <= 8;
  const enoughResolution = width >= 480 && height >= 360;

  return [
    {
      title: "视频长度",
      detail: `${duration.toFixed(1)} 秒；MVP 建议 3-8 秒。`,
      state: shortEnough ? "ok" : "warn",
      label: shortEnough ? "合格" : "需复查",
    },
    {
      title: "画面方向",
      detail: `${width} x ${height}；横屏利于篮筐、球路和全身姿态分析。`,
      state: landscape ? "ok" : "warn",
      label: landscape ? "横屏" : "竖屏",
    },
    {
      title: "清晰度",
      detail: "第一版主要检查全身骨架是否稳定贴合人物。",
      state: enoughResolution ? "ok" : "warn",
      label: enoughResolution ? "可用" : "偏低",
    },
  ];
}

function waitForMetadata(video) {
  return new Promise((resolve) => {
    if (Number.isFinite(video.duration) && video.videoWidth > 0) {
      resolve();
      return;
    }
    video.onloadedmetadata = () => resolve();
  });
}

function seekVideo(video, time) {
  return new Promise((resolve) => {
    const onSeeked = () => {
      video.removeEventListener("seeked", onSeeked);
      resolve();
    };
    video.addEventListener("seeked", onSeeked);
    video.currentTime = Math.min(Math.max(time, 0), video.duration || 0);
  });
}

async function captureFrame(video, time) {
  await seekVideo(video, time);
  const canvas = document.createElement("canvas");
  const width = video.videoWidth;
  const height = video.videoHeight;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, width, height);
  return canvas.toDataURL("image/jpeg", 0.86);
}

async function extractKeyframes(video) {
  const duration = video.duration || 0;
  const frames = [];
  for (const item of framePlan) {
    const time = duration * item.ratio;
    const dataUrl = await captureFrame(video, time);
    frames.push({ ...item, time, dataUrl });
  }
  return frames;
}

function poseSummary(frame) {
  const metrics = frame.pose?.metrics;
  if (!metrics) return "未检测到骨架";
  const parts = [`关键点 ${metrics.visible_keypoints ?? 0}/17`];
  if (metrics.shooting_side) parts.push(`投篮侧 ${metrics.shooting_side}`);
  if (metrics.shooting_elbow_angle) parts.push(`肘角 ${metrics.shooting_elbow_angle}°`);
  if (metrics.min_knee_angle) parts.push(`膝角 ${metrics.min_knee_angle}°`);
  return parts.join(" · ");
}

function frameMeta(frame) {
  const detections = frame.detections ? `${frame.detections.length} detections` : "browser preview";
  return `${frame.time.toFixed(2)}s · ${frame.key} · ${detections}`;
}

function selectionSummary(frame) {
  if (!frame.selection_method && frame.confidence == null && !frame.evidence) return "";
  const parts = [];
  if (frame.selection_method) parts.push(`选择方式：${frame.selection_method}`);
  if (frame.confidence != null) parts.push(`置信度：${Number(frame.confidence).toFixed(2)}`);
  if (typeof frame.evidence === "string") {
    parts.push(`依据：${frame.evidence}`);
  } else if (frame.evidence && typeof frame.evidence === "object") {
    const evidenceParts = [];
    if (frame.evidence.visible_keypoints != null) {
      evidenceParts.push(`关键点 ${frame.evidence.visible_keypoints}/17`);
    }
    if (frame.evidence.elbow_angle != null) {
      evidenceParts.push(`肘角 ${frame.evidence.elbow_angle}°`);
    }
    if (frame.evidence.wrist_y != null) {
      evidenceParts.push(`手腕y ${frame.evidence.wrist_y}`);
    }
    if (frame.evidence.ball_detected != null) {
      evidenceParts.push(frame.evidence.ball_detected ? "检测到球" : "未检测到球");
    }
    if (frame.evidence.ball_moving_away) {
      evidenceParts.push("球开始远离手腕");
    }
    if (evidenceParts.length) parts.push(`依据：${evidenceParts.join(" ｜ ")}`);
  }
  return parts.join(" ｜ ");
}

function clearReleaseBallEvidence() {
  if (!releaseBallEvidence) return;
  releaseBallEvidence.hidden = true;
  releaseBallEvidence.innerHTML = "";
}

function clearBallTrackEvidence() {
  if (!ballTrackEvidence) return;
  ballTrackEvidence.hidden = true;
  ballTrackEvidence.innerHTML = "";
}

function clearReleaseFusionEvidence() {
  if (!releaseFusionEvidence) return;
  releaseFusionEvidence.hidden = true;
  releaseFusionEvidence.innerHTML = "";
}

function releaseBallStatusMeta(status) {
  if (status === "ok") {
    return { state: "ok", label: "检测成功" };
  }
  if (status === "model_missing" || status === "disabled_by_missing_model") {
    return { state: "warn", label: "模型缺失" };
  }
  if (status === "error") {
    return { state: "danger", label: "检测异常" };
  }
  return { state: "warn", label: status || "未启用" };
}

function formatReleaseBallBBox(bbox) {
  if (!Array.isArray(bbox) || bbox.length !== 4) return "无";
  return `[${bbox.map((value) => Math.round(Number(value) || 0)).join(", ")}]`;
}

function renderReleaseBallEvidence(evidence) {
  if (!releaseBallEvidence) return;
  if (!evidence) {
    clearReleaseBallEvidence();
    return;
  }

  const frames = Array.isArray(evidence.frames) ? evidence.frames : [];
  const detectedFrames = frames.filter((frame) => frame?.has_detection);
  const bestFrame =
    evidence.best_frame && typeof evidence.best_frame === "object" ? evidence.best_frame : null;
  const highestConfidence = frames.reduce((max, frame) => {
    const value = Number(frame?.confidence);
    return Number.isFinite(value) ? Math.max(max, value) : max;
  }, 0);
  const statusMeta = releaseBallStatusMeta(evidence.status);

  let message = "辅助展示 release ball detector 在 release 邻域的检测结果。";
  if (evidence.status === "model_missing" || evidence.status === "disabled_by_missing_model") {
    message = "模型缺失 / detector 未启用成功，页面已自动保持旧分析结果。";
  } else if (evidence.status === "error") {
    message = "Detector 运行异常，页面已自动回退到原有分析结果。";
  } else if (!bestFrame) {
    message = "未检测到 release ball。";
  }

  releaseBallEvidence.hidden = false;
  releaseBallEvidence.innerHTML = `
    <div class="release-ball-card ${statusMeta.state}">
      <div class="release-ball-card-head">
        <div>
          <strong>Release Ball Detector Evidence</strong>
          <small>${message}</small>
        </div>
        ${statusLabel(statusMeta.state, statusMeta.label)}
      </div>
      <div class="release-ball-grid">
        <div class="release-ball-item">
          <span>status</span>
          <strong>${evidence.status || "unknown"}</strong>
        </div>
        <div class="release-ball-item">
          <span>detector_type</span>
          <strong>${evidence.detector_type || "release_ball_yolo"}</strong>
        </div>
        <div class="release-ball-item">
          <span>release_frame_index</span>
          <strong>${evidence.release_frame_index ?? "-"}</strong>
        </div>
        <div class="release-ball-item">
          <span>window_radius</span>
          <strong>${evidence.window_radius ?? "-"}</strong>
        </div>
        <div class="release-ball-item">
          <span>frames 总数</span>
          <strong>${frames.length}</strong>
        </div>
        <div class="release-ball-item">
          <span>命中帧数</span>
          <strong>${detectedFrames.length}</strong>
        </div>
        <div class="release-ball-item">
          <span>最高 confidence</span>
          <strong>${highestConfidence.toFixed(3)}</strong>
        </div>
        <div class="release-ball-item">
          <span>best_frame.frame_index</span>
          <strong>${bestFrame?.frame_index ?? "未检测到 release ball"}</strong>
        </div>
        <div class="release-ball-item">
          <span>best_frame.distance_to_release</span>
          <strong>${bestFrame?.distance_to_release ?? "-"}</strong>
        </div>
        <div class="release-ball-item">
          <span>best_frame.confidence</span>
          <strong>${bestFrame ? Number(bestFrame.confidence || 0).toFixed(3) : "-"}</strong>
        </div>
        <div class="release-ball-item release-ball-item-wide">
          <span>best_frame.bbox</span>
          <strong>${bestFrame ? formatReleaseBallBBox(bestFrame.bbox) : "无"}</strong>
        </div>
      </div>
    </div>
  `;
}

function renderBallTrackEvidence(evidence, releaseMeasurement) {
  if (!ballTrackEvidence) return;
  if (!evidence) {
    clearBallTrackEvidence();
    return;
  }

  const frames = Array.isArray(evidence.frames) ? evidence.frames : [];
  const trusted = releaseMeasurement?.trusted_flight;
  const releaseEpoch = releaseMeasurement?.release_state?.release_epoch;
  const state = evidence.status === "ok" ? "ok" : "warn";
  const timeline = frames
    .map((frame) => {
      const isTrusted =
        trusted &&
        frame.frame_index >= trusted.start_frame &&
        frame.frame_index <= trusted.end_frame;
      const isReleaseEpoch =
        releaseEpoch &&
        frame.time_s >= releaseEpoch.lower_time_s &&
        frame.time_s <= releaseEpoch.upper_time_s;
      return `<span class="ball-track-tick ${frame.status || "no_detection"} ${isTrusted ? "trusted" : ""} ${isReleaseEpoch ? "release-epoch" : ""}" title="frame ${frame.frame_index}: ${frame.status || "no_detection"}${isTrusted ? " · TrustedFlight" : ""}"></span>`;
    })
    .join("");

  ballTrackEvidence.hidden = false;
  ballTrackEvidence.innerHTML = `
    <div class="release-ball-card ${state}">
      <div class="release-ball-card-head">
        <div>
          <strong>Ball Tracking Evidence</strong>
          <small>圆点只对应实际 detector observation；橙色描边为 TrustedFlight，紫色为 ReleaseEpoch interval。</small>
        </div>
        ${statusLabel(state, evidence.status === "ok" ? "已采集" : "不可用")}
      </div>
      <div class="release-ball-grid">
        <div class="release-ball-item"><span>连续采集帧数</span><strong>${evidence.requested_frame_count ?? 0}</strong></div>
        <div class="release-ball-item"><span>检出球帧数</span><strong>${evidence.detection_frame_count ?? 0}</strong></div>
        <div class="release-ball-item"><span>缺失帧数</span><strong>${evidence.missing_frame_count ?? 0}</strong></div>
        <div class="release-ball-item release-ball-item-wide"><span>frame timeline</span><strong class="ball-track-timeline">${timeline || "无可用帧"}</strong></div>
      </div>
    </div>
  `;
}

function formatMetric(value, digits, unit) {
  return value != null && Number.isFinite(Number(value))
    ? `${Number(value).toFixed(digits)} ${unit}`
    : "Unavailable";
}

function formatMetricInterval(interval, digits, unit) {
  return Array.isArray(interval) && interval.length === 2
    ? `${Number(interval[0]).toFixed(digits)}–${Number(interval[1]).toFixed(digits)} ${unit}`
    : "Unavailable";
}

function renderReleaseMeasurement(releaseMeasurement, releaseFusion) {
  if (!releaseFusionEvidence) return;
  const measurement = releaseMeasurement || releaseFusion;
  if (!measurement) {
    clearReleaseFusionEvidence();
    return;
  }

  const isMeasurement = Boolean(releaseMeasurement);
  const status = measurement.status || "unknown";
  const source = measurement.source || measurement.final_source || "-";
  const releaseFrame = measurement.release_frame ?? measurement.pose_release_frame_index ?? "-";
  const frameDelta = measurement.frame_delta == null ? "-" : measurement.frame_delta;
  const agreementLevel = measurement.agreement_level || "-";
  const reason = measurement.reason || "未提供释放测量说明";
  const quality = measurement.measurement_quality || {};
  const releaseState = measurement.release_state || null;
  const trustedFlight = measurement.trusted_flight || null;
  const qualification = releaseState?.qualification || "UNAVAILABLE";
  const qualified = qualification === "HIGH" || qualification === "MEDIUM";
  const uncertainty = releaseState?.uncertainty || {};
  const epoch = releaseState?.release_epoch || null;
  const qualityNotes = [...(quality.issues || []), ...(quality.warnings || [])].filter(
    (note) => typeof note === "string",
  );

  let state = "danger";
  let label = "风险";
  let message = "当前释放测量存在质量风险，请谨慎解读相关指标。";
  if (isMeasurement && qualified) {
    state = qualification === "HIGH" ? "ok" : "warn";
    label = qualification;
    message = "已从实际连续球观测生成 V2 near-side release measurement。";
  } else if (isMeasurement && releaseState?.qualification === "UNQUALIFIED") {
    state = "warn";
    label = "UNQUALIFIED";
    message = releaseState.reason || "当前轨迹输出稳定性未通过 V2 availability gate，metric 数值已隐藏。";
  } else if (isMeasurement && status === "AVAILABLE") {
    state = "warn";
    label = "无 Metric";
    message = "释放帧可用，但连续球轨迹不足以生成合格的 V2 metric measurement。";
  } else if (isMeasurement && status === "INSUFFICIENT_DATA") {
    state = "warn";
    label = "证据不足";
    message = "当前视频缺少足够的释放测量证据，部分投篮指标暂不可用。";
  } else if (isMeasurement && status === "UNRELIABLE") {
    state = "danger";
    label = "谨慎解读";
    message = "当前释放测量存在质量风险，请谨慎解读相关指标。";
  } else if (!isMeasurement) {
    state = status === "ok" ? "ok" : status === "warning" || status === "uncertain" ? "warn" : "danger";
    label = state === "ok" ? "一致" : state === "warn" ? "待确认" : "风险";
    message = "当前后端未提供统一测量 contract，正在展示兼容的 release fusion 诊断。";
  }
  const qualitySummary = qualityNotes.length
    ? `包含 ${qualityNotes.length} 项质量提示。`
    : "无额外质量提示。";
  const releaseTime = qualified ? releaseState.release_time ?? measurement.release_time : null;
  const speed = qualified ? releaseState.speed_mps : null;
  const elevation = qualified ? releaseState.elevation_angle_deg : null;
  const support = trustedFlight?.temporal_span_ms != null
    ? `${(trustedFlight.temporal_span_ms / 1000).toFixed(2)} s / ${trustedFlight.point_count ?? 0} observations`
    : "Unavailable";
  const epochInterval = epoch
    ? `${Number(epoch.lower_time_s).toFixed(3)}–${Number(epoch.upper_time_s).toFixed(3)} s (representative ${Number(epoch.representative_time_s).toFixed(3)} s)`
    : "Unavailable";
  const metricReason = releaseState?.reason || reason;
  const riskCount = Array.isArray(measurement.risk_flags) ? measurement.risk_flags.length : 0;

  releaseFusionEvidence.hidden = false;
  releaseFusionEvidence.innerHTML = `
    <div class="release-fusion-card ${state}">
      <div class="release-ball-card-head">
        <div>
          <strong>Release Measurement</strong>
          <small>${message}</small>
        </div>
        ${statusLabel(state, label)}
      </div>
      <div class="release-measurement-grid">
        <div class="release-measurement-value"><span>Release Time</span><strong>${formatMetric(releaseTime, 2, "s")}</strong><small>representative epoch</small></div>
        <div class="release-measurement-value"><span>Release Speed</span><strong>${formatMetric(speed, 1, "m/s")}</strong><small>${qualified ? qualification : "UNQUALIFIED"}</small></div>
        <div class="release-measurement-value"><span>Release Angle</span><strong>${formatMetric(elevation, 1, "°")}</strong><small>${qualified ? qualification : "UNQUALIFIED"}</small></div>
        <div class="release-measurement-value"><span>Flight Support</span><strong>${support}</strong><small>actual unique observations</small></div>
        <div class="release-measurement-value"><span>Measurement Quality</span><strong>${qualification}</strong><small>V2 engineering gate</small></div>
      </div>
      <div class="release-fusion-grid">
        <div class="release-fusion-item"><span>release_frame</span><strong>${releaseFrame}</strong></div>
        <div class="release-fusion-item"><span>source</span><strong>${source}</strong></div>
        <div class="release-fusion-item"><span>agreement</span><strong>${agreementLevel} / Δ ${frameDelta}</strong></div>
        <div class="release-fusion-item release-fusion-item-wide"><span>ReleaseEpoch interval</span><strong>${epochInterval}</strong></div>
        <div class="release-fusion-item release-fusion-item-wide"><span>Speed 95% interval</span><strong>${formatMetricInterval(uncertainty.speed_interval_mps, 2, "m/s")}</strong></div>
        <div class="release-fusion-item release-fusion-item-wide"><span>Angle 95% interval</span><strong>${formatMetricInterval(uncertainty.elevation_angle_interval_deg, 1, "°")}</strong></div>
        <div class="release-fusion-item release-fusion-item-wide"><span>说明</span><strong>${metricReason}</strong></div>
        <div class="release-fusion-item release-fusion-item-wide"><span>质量与风险</span><strong>${qualitySummary} ${riskCount ? `另有 ${riskCount} 项 release 风险标记。` : ""}</strong></div>
      </div>
    </div>
  `;
}

async function addBallTrajectoryOverlay(frames, evidence, releaseMeasurement) {
  const releaseFrame = frames.find((frame) => frame.key === "release");
  const evidenceFrames = Array.isArray(evidence?.frames) ? evidence.frames : [];
  if (!releaseFrame || !evidenceFrames.length) return;
  const observations = evidenceFrames
    .filter((frame) => Array.isArray(frame.detections) && frame.detections.length === 1)
    .map((frame) => ({ ...frame, detection: frame.detections[0] }))
    .filter((frame) => Number.isFinite(frame.detection.center_x_px) && Number.isFinite(frame.detection.center_y_px));
  if (!observations.length) return;

  const image = new Image();
  await new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = reject;
    image.src = releaseFrame.dataUrl;
  });
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0);
  const trusted = releaseMeasurement?.trusted_flight;
  const isTrusted = (frame) =>
    trusted && frame.frame_index >= trusted.start_frame && frame.frame_index <= trusted.end_frame;

  for (let index = 1; index < observations.length; index += 1) {
    const previous = observations[index - 1];
    const current = observations[index];
    if (current.frame_index !== previous.frame_index + 1) continue;
    context.strokeStyle = isTrusted(previous) && isTrusted(current) ? "#f97316" : "#38bdf8";
    context.lineWidth = isTrusted(current) ? 4 : 2;
    context.beginPath();
    context.moveTo(previous.detection.center_x_px, previous.detection.center_y_px);
    context.lineTo(current.detection.center_x_px, current.detection.center_y_px);
    context.stroke();
  }
  observations.forEach((frame) => {
    context.fillStyle = isTrusted(frame) ? "#f97316" : "#38bdf8";
    context.beginPath();
    context.arc(frame.detection.center_x_px, frame.detection.center_y_px, isTrusted(frame) ? 5 : 3, 0, Math.PI * 2);
    context.fill();
  });
  context.fillStyle = "rgba(15, 23, 42, 0.82)";
  context.fillRect(8, 8, 280, 48);
  context.font = "15px sans-serif";
  context.fillStyle = "#38bdf8";
  context.fillText("● actual observation", 18, 29);
  context.fillStyle = "#f97316";
  context.fillText("● TrustedFlight", 18, 49);
  releaseFrame.visualDataUrl = canvas.toDataURL("image/jpeg", 0.9);
}

function setActiveFrame(index) {
  activeFrameIndex = index;
  renderFrames(renderedFrames);
}

function openImageModal(src) {
  imageModalImg.src = src;
  imageModal.classList.add("open");
  imageModal.setAttribute("aria-hidden", "false");
}

function closeImageModal() {
  imageModal.classList.remove("open");
  imageModal.setAttribute("aria-hidden", "true");
  imageModalImg.removeAttribute("src");
}

function renderFrames(frames) {
  renderedFrames = frames;
  if (!frames.length) {
    framesGrid.innerHTML = "";
    return;
  }

  const safeIndex = Math.min(activeFrameIndex, frames.length - 1);
  const active = frames[safeIndex];
  activeFrameIndex = safeIndex;

  framesGrid.innerHTML = `
    <div class="frame-viewer">
      <button class="main-frame" type="button" data-full-frame="${active.visualDataUrl || active.dataUrl}" aria-label="查看${active.label}关键帧大图">
        <img src="${active.visualDataUrl || active.dataUrl}" alt="${active.label}关键帧" />
      </button>
      <div class="main-frame-info">
        <div>
          <strong>${active.label}</strong>
          <small>${frameMeta(active)}</small>
          ${selectionSummary(active) ? `<small>${selectionSummary(active)}</small>` : ""}
        </div>
        <span>${poseSummary(active)}</span>
      </div>
    </div>
    <div class="frame-strip">
      ${frames
        .map(
          (frame, index) => `
          <button class="frame-thumb ${index === activeFrameIndex ? "active" : ""}" type="button" data-frame-index="${index}">
            <img src="${frame.dataUrl}" alt="${frame.label}缩略图" />
            <span>${frame.label}</span>
            <small>${frame.time.toFixed(2)}s</small>
          </button>
        `,
        )
        .join("")}
    </div>
  `;

  framesGrid.querySelector("[data-full-frame]")?.addEventListener("click", () => {
    openImageModal(active.visualDataUrl || active.dataUrl);
  });

  framesGrid.querySelectorAll("[data-frame-index]").forEach((button) => {
    button.addEventListener("click", () => setActiveFrame(Number(button.dataset.frameIndex)));
  });
}

function getPlaceholderMetrics(video) {
  const aspect = video.videoWidth / Math.max(video.videoHeight, 1);
  return [
    {
      title: "浏览器预览模式",
      detail: "后端分析失败时只抽取原始画面，不包含骨架和姿态指标。",
      state: "warn",
      label: "降级",
    },
    {
      title: "视频比例",
      detail: `当前画面比例 ${aspect.toFixed(2)}。`,
      state: aspect >= 1 ? "ok" : "warn",
      label: aspect >= 1 ? "横屏" : "竖屏",
    },
  ];
}

videoInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  selectedFile = file;

  if (videoUrl) {
    URL.revokeObjectURL(videoUrl);
  }

  videoUrl = URL.createObjectURL(file);
  videoPreview.src = videoUrl;
  videoPreview.style.display = "block";
  emptyPreview.style.display = "none";
  analyzeButton.disabled = false;
  resetButton.disabled = false;
  framesGrid.innerHTML = "";
  metricsList.innerHTML = "";
  renderedFrames = [];
  activeFrameIndex = 0;
  setPipeline(1);

  await waitForMetadata(videoPreview);
  renderQuality(getVideoQuality(videoPreview));
});

analyzeButton.addEventListener("click", async () => {
  analyzeButton.disabled = true;
  setPipeline(1);
  await waitForMetadata(videoPreview);

  try {
    if (!selectedFile) {
      throw new Error("No selected file");
    }
    setPipeline(2);
    const formData = new FormData();
    formData.append("file", selectedFile);
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    const report = await response.json();
    setPipeline(3);
    renderQuality(report.quality);
    activeFrameIndex = report.frames.findIndex((frame) => frame.key === "release");
    if (activeFrameIndex < 0) activeFrameIndex = 0;
    await addBallTrajectoryOverlay(report.frames, report.ball_track_evidence, report.release_measurement);
    renderFrames(report.frames);
    renderReleaseBallEvidence(report.release_ball_evidence);
    renderBallTrackEvidence(report.ball_track_evidence, report.release_measurement);
    renderReleaseMeasurement(report.release_measurement, report.release_fusion);
    setPipeline(4);
    renderMetrics(report.metrics);
    setPipeline(5);
  } catch (error) {
    console.warn("Backend analysis failed; using browser fallback.", error);
    const frames = await extractKeyframes(videoPreview);
    setPipeline(3);
    activeFrameIndex = 0;
    renderFrames(frames);
    clearReleaseBallEvidence();
    clearBallTrackEvidence();
    clearReleaseFusionEvidence();
    setPipeline(4);
    renderMetrics(getPlaceholderMetrics(videoPreview));
    setPipeline(5);
  } finally {
    analyzeButton.disabled = false;
  }
});

resetButton.addEventListener("click", () => {
  if (videoUrl) {
    URL.revokeObjectURL(videoUrl);
  }
  videoUrl = null;
  videoInput.value = "";
  selectedFile = null;
  videoPreview.removeAttribute("src");
  videoPreview.load();
  videoPreview.style.display = "none";
  emptyPreview.style.display = "block";
  analyzeButton.disabled = true;
  resetButton.disabled = true;
  qualityList.innerHTML = "";
  framesGrid.innerHTML = "";
  clearReleaseBallEvidence();
  clearBallTrackEvidence();
  clearReleaseFusionEvidence();
  metricsList.innerHTML = "";
  renderedFrames = [];
  activeFrameIndex = 0;
  closeImageModal();
  setPipeline(0);
});

imageModalClose.addEventListener("click", closeImageModal);
imageModal.addEventListener("click", (event) => {
  if (event.target === imageModal) closeImageModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && imageModal.classList.contains("open")) {
    closeImageModal();
  }
});

setPipeline(0);
renderQuality([
  {
    title: "等待视频",
    detail: "上传后会检查时长、分辨率和画面方向。",
    state: "warn",
    label: "未开始",
  },
]);

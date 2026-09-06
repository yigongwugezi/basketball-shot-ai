const $ = (id) => document.getElementById(id);
const videoInput = $("videoInput");
const analyzeButton = $("analyzeButton");
const resetButton = $("resetButton");
const videoPreview = $("videoPreview");
const emptyPreview = $("emptyPreview");
const qualityList = $("qualityList");
const pipelineList = $("pipelineList");
const videoInfoList = $("videoInfoList");
const analysisSummary = $("analysisSummary");
const flightVisualization = $("flightVisualization");
const flightSummary = $("flightSummary");
const flightStatus = $("flightStatus");
const framesGrid = $("framesGrid");
const releaseBallEvidence = $("releaseBallEvidence");
const ballTrackEvidence = $("ballTrackEvidence");
const releaseFusionEvidence = $("releaseFusionEvidence");
const rawDeveloperDetails = $("rawDeveloperDetails");
const metricsList = $("metricsList");
const imageModal = $("imageModal");
const imageModalImg = $("imageModalImg");
const imageModalClose = $("imageModalClose");
const ui = window.BasketballUI;

const pipelineSteps = ["读取视频信息", "检查拍摄适用性", "定位动作关键帧", "分析人体与篮球证据", "生成测量结果"];

let videoUrl = null;
let selectedFile = null;
let activeFrameIndex = 0;
let renderedFrames = [];

function statusLabel(state, label) {
  return `<span class="status ${state}">${label}</span>`;
}

function setPipeline(doneCount = 0) {
  pipelineList.innerHTML = pipelineSteps.map((step, index) => `<li class="${index < doneCount ? "done" : ""}">${step}</li>`).join("");
}

function renderWaitingState() {
  analysisSummary.className = "panel summary-panel waiting";
  analysisSummary.innerHTML = `<div><p class="eyebrow">Analysis Summary（本次分析）</p><h2>等待视频分析</h2><p class="section-copy">测量结果会在这里显示；证据不足时，系统会说明原因并保留可用的动作分析。</p></div>`;
  qualityList.innerHTML = `<div class="empty-state compact">上传视频后显示拍摄适用性。</div>`;
  videoInfoList.innerHTML = `<div class="empty-state compact">暂无视频信息</div>`;
  flightStatus.className = "status neutral";
  flightStatus.textContent = "等待分析";
  flightVisualization.innerHTML = `<div class="empty-state">分析后将在这里显示真实观测到的篮球轨迹。</div>`;
  flightSummary.innerHTML = "";
}

function formatNumber(value, digits, unit = "") {
  return value != null && Number.isFinite(Number(value)) ? `${Number(value).toFixed(digits)}${unit ? ` ${unit}` : ""}` : "—";
}

function renderVideoInfo(meta = {}) {
  const width = meta.width ?? videoPreview.videoWidth ?? 0;
  const height = meta.height ?? videoPreview.videoHeight ?? 0;
  const duration = meta.duration ?? videoPreview.duration ?? 0;
  const fps = meta.fps;
  videoInfoList.innerHTML = [
    ["时长", formatNumber(duration, 2, "秒")],
    ["分辨率", width && height ? `${width} × ${height}` : "—"],
    ["画面方向", width && height ? (width >= height ? "横屏" : "竖屏") : "—"],
    ["帧率", fps != null ? formatNumber(fps, 1, "fps（帧/秒）") : "分析后读取"],
  ].map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function renderLocalSuitability() {
  qualityList.innerHTML = `<div class="quality-item"><div><strong>视频已读取</strong><small>拍摄适用性将在分析篮球轨迹和测量稳定性后判断。</small></div>${statusLabel("neutral", "待分析")}</div>`;
  renderVideoInfo();
}

function suitabilityItem(title, detail, state, label) {
  return `<div class="quality-item"><div><strong>${title}</strong><small>${detail}</small></div>${statusLabel(state, label)}</div>`;
}

function renderSuitability(report) {
  const measurement = report.release_measurement;
  const track = report.ball_track_evidence;
  const trusted = measurement?.trusted_flight;
  const level = ui.qualification(measurement);
  const camera = report.camera;
  const detected = Number(track?.detection_frame_count || 0);
  const requested = Number(track?.requested_frame_count || 0);
  const continuity = requested ? detected / requested : 0;
  const trackMeta = ui.trackStatus(track, measurement);
  const viewLabel = camera?.view_label || "后端未提供独立机位判断";
  const viewState = camera?.angle_reliability === "medium" ? "ok" : "warn";
  const items = [
    suitabilityItem("视角适用性", viewLabel, viewState, camera ? (viewState === "ok" ? "可支持" : "需谨慎") : "未判断"),
    suitabilityItem("出手后篮球可见性", requested ? `${requested} 个采集帧中有 ${detected} 帧观察到篮球。` : "没有连续篮球采集证据。", detected ? "ok" : "warn", detected ? "有观测" : "无连续观测"),
    suitabilityItem("篮球轨迹连续性", trackMeta.text, trackMeta.state, trackMeta.label),
    suitabilityItem("飞行时间支持", trusted?.temporal_span_ms != null ? `${(trusted.temporal_span_ms / 1000).toFixed(2)} 秒 / ${trusted.point_count ?? 0} 个有效观测。` : "尚未形成 TrustedFlight（可信飞行段）。", trusted ? "ok" : "warn", trusted ? "已支持" : "不足"),
    suitabilityItem("最终测量稳定性", level === "UNQUALIFIED" ? "当前视频未通过冻结的 V2 工程资格检查。" : "结果已通过后端冻结的 V2 工程资格检查。", level === "HIGH" ? "ok" : "warn", ui.reliabilityLabels[level]),
  ];
  if (track && requested) items.splice(3, 0, suitabilityItem("连续观测比例", `${detected}/${requested}（${Math.round(continuity * 100)}%）；这里只描述证据，不单独决定资格。`, continuity >= 0.5 ? "ok" : "warn", "证据信息"));
  qualityList.innerHTML = items.join("");
  renderVideoInfo(report.metadata);
}

function metricCard(label, value, detail) {
  return `<div class="summary-metric"><span>${label}</span><strong>${value}</strong><small>${detail}</small></div>`;
}

function renderAnalysisSummary(measurement, track) {
  const model = ui.analysisSummary(measurement, track);
  analysisSummary.className = `panel summary-panel ${model.qualified ? "qualified" : "unqualified"}`;
  if (model.qualified) {
    analysisSummary.innerHTML = `
      <div class="summary-heading"><div><p class="eyebrow">Analysis Summary（本次分析）</p><h2>本次投篮已完成可靠测量</h2></div>${statusLabel(model.level === "HIGH" ? "ok" : "warn", model.reliabilityLabel)}</div>
      <div class="summary-metrics">
        ${metricCard("Release Time（出手时间）", formatNumber(model.releaseTime, 2, "秒"), "ReleaseEpoch（出手时刻区间）的代表值")}
        ${metricCard("Release Speed（出手速度）", formatNumber(model.speed, 1, "m/s（米/秒）"), model.reliabilityLabel)}
        ${metricCard("Release Angle（出手角度）", formatNumber(model.angle, 1, "°"), model.reliabilityLabel)}
        ${metricCard("Flight Support（飞行支持）", model.supportMs != null ? `${(model.supportMs / 1000).toFixed(2)} 秒` : "—", `${model.observations} 个有效观测`)}
        ${metricCard("Measurement（测量可靠性）", model.reliabilityLabel, "不是动作好坏评分")}
      </div>`;
    return;
  }
  analysisSummary.innerHTML = `
    <div class="summary-heading"><div><p class="eyebrow">Analysis Summary（本次分析）</p><h2>本次无法可靠测量出手速度与角度</h2></div>${statusLabel("warn", model.reliabilityLabel)}</div>
    <div class="failure-callout"><strong>${model.failure.title}</strong><p>${model.failure.advice}</p><p>系统未输出可能误导的精确物理量。</p></div>
    <div class="fallback-note"><span>物理测量：<strong>当前视频不支持可靠速度 / 角度</strong></span><span>动作分析：<strong>仍可用</strong></span></div>`;
}

function generalBallObservation(frames) {
  const ballFrames = frames.filter((frame) => (frame.detections || []).some((detection) => detection.class_name === "ball"));
  return { count: ballFrames.length, observed: ballFrames.length > 0 };
}

function renderFlightSummary(report) {
  const track = report.ball_track_evidence;
  const measurement = report.release_measurement;
  const status = ui.trackStatus(track, measurement);
  const general = generalBallObservation(report.frames || []);
  flightStatus.className = `status ${status.state}`;
  flightStatus.textContent = status.label;
  const singleFrameText = general.observed ? `关键帧篮球观察：已在 ${general.count} 个动作关键帧看到篮球。` : "关键帧篮球观察：未观察到篮球。";
  const continuousText = track?.detection_frame_count ? `连续飞行轨迹：采集 ${track.requested_frame_count ?? 0} 帧，检出 ${track.detection_frame_count} 帧，缺失 ${track.missing_frame_count ?? 0} 帧。` : "连续飞行轨迹：未形成可用于物理测量的连续轨迹。";
  flightSummary.innerHTML = `<div class="flight-stat"><strong>${singleFrameText}</strong><span>${continuousText}</span><small>单帧看到篮球不代表已经获得足够稳定的连续飞行观测。</small></div>`;
}

function uniqueObservations(evidence) {
  return (evidence?.frames || []).filter((frame) => Array.isArray(frame.detections) && frame.detections.length === 1).map((frame) => ({ ...frame, detection: frame.detections[0] })).filter((frame) => Number.isFinite(frame.detection.center_x_px) && Number.isFinite(frame.detection.center_y_px));
}

async function capturePlainFrame(time) {
  if (!videoPreview.videoWidth || !videoPreview.videoHeight) return null;
  const previousTime = videoPreview.currentTime;
  await seekVideo(videoPreview, time);
  const canvas = document.createElement("canvas");
  canvas.width = videoPreview.videoWidth;
  canvas.height = videoPreview.videoHeight;
  canvas.getContext("2d").drawImage(videoPreview, 0, 0, canvas.width, canvas.height);
  videoPreview.currentTime = previousTime;
  return canvas;
}

async function renderFlightVisualization(report) {
  const evidence = report.ball_track_evidence;
  const measurement = report.release_measurement;
  const observations = uniqueObservations(evidence);
  if (!observations.length) {
    flightVisualization.innerHTML = `<div class="empty-state">没有可绘制的连续篮球观测；动作关键帧仍可在下方查看。</div>`;
    return;
  }
  const releaseFrame = (report.frames || []).find((frame) => frame.key === "release");
  const canvas = await capturePlainFrame(releaseFrame?.time || observations[0].time_s || 0);
  if (!canvas) return;
  const context = canvas.getContext("2d");
  const trusted = measurement?.trusted_flight;
  const epoch = measurement?.release_state?.release_epoch;
  const isTrusted = (frame) => trusted && frame.frame_index >= trusted.start_frame && frame.frame_index <= trusted.end_frame;
  for (let index = 1; index < observations.length; index += 1) {
    const previous = observations[index - 1];
    const current = observations[index];
    if (current.frame_index !== previous.frame_index + 1) continue;
    context.strokeStyle = isTrusted(previous) && isTrusted(current) ? "#f97316" : "#38bdf8";
    context.lineWidth = isTrusted(current) ? 5 : 3;
    context.beginPath();
    context.moveTo(previous.detection.center_x_px, previous.detection.center_y_px);
    context.lineTo(current.detection.center_x_px, current.detection.center_y_px);
    context.stroke();
  }
  observations.forEach((frame) => {
    const inEpoch = epoch && frame.time_s >= epoch.lower_time_s && frame.time_s <= epoch.upper_time_s;
    context.fillStyle = inEpoch ? "#a855f7" : isTrusted(frame) ? "#f97316" : "#38bdf8";
    context.beginPath();
    context.arc(frame.detection.center_x_px, frame.detection.center_y_px, inEpoch ? 8 : isTrusted(frame) ? 6 : 4, 0, Math.PI * 2);
    context.fill();
  });
  flightVisualization.innerHTML = `<img src="${canvas.toDataURL("image/jpeg", 0.9)}" alt="实际篮球观测轨迹、可信飞行段与出手时刻区间" />`;
}

function frameMeta(frame) {
  return `${Number(frame.time || 0).toFixed(2)} 秒 · 第 ${frame.frame_index ?? "—"} 帧`;
}

function poseSummary(frame) {
  const metrics = frame.pose?.metrics;
  if (!metrics) return "未检测到骨架";
  const parts = [`可见关键点 ${metrics.visible_keypoints ?? 0}/17`];
  if (metrics.shooting_elbow_angle != null) parts.push(`肘角 ${metrics.shooting_elbow_angle}°`);
  if (metrics.min_knee_angle != null) parts.push(`膝角 ${metrics.min_knee_angle}°`);
  return parts.join(" · ");
}

function renderFrames(frames) {
  renderedFrames = frames || [];
  if (!renderedFrames.length) {
    framesGrid.innerHTML = `<div class="empty-state">暂无动作关键帧</div>`;
    return;
  }
  activeFrameIndex = Math.min(Math.max(activeFrameIndex, 0), renderedFrames.length - 1);
  const active = renderedFrames[activeFrameIndex];
  framesGrid.innerHTML = `
    <div class="frame-viewer"><button class="main-frame" type="button" aria-label="查看${active.label}关键帧大图"><img src="${active.dataUrl}" alt="${active.label}关键帧" /></button><div class="main-frame-info"><div><strong>${active.label}</strong><small>${frameMeta(active)}</small></div><span>${poseSummary(active)}</span></div></div>
    <div class="frame-strip">${renderedFrames.map((frame, index) => `<button class="frame-thumb ${index === activeFrameIndex ? "active" : ""}" type="button" data-frame-index="${index}"><img src="${frame.dataUrl}" alt="${frame.label}缩略图" /><span>${frame.label}</span><small>${Number(frame.time || 0).toFixed(2)} 秒</small></button>`).join("")}</div>`;
  framesGrid.querySelector(".main-frame")?.addEventListener("click", () => openImageModal(active.dataUrl));
  framesGrid.querySelectorAll("[data-frame-index]").forEach((button) => button.addEventListener("click", () => { activeFrameIndex = Number(button.dataset.frameIndex); renderFrames(renderedFrames); }));
}

function renderMetrics(metrics) {
  const allowed = (metrics || []).filter((metric) => ["出手帧肘角", "下沉帧膝角", "躯干倾斜", "人体骨架", "2D角度可信度"].includes(metric.title));
  metricsList.innerHTML = allowed.length ? allowed.map((metric) => `<div class="metric-item"><div><strong>${metric.title}</strong><small>${metric.detail}</small></div>${statusLabel(metric.state, metric.label)}</div>`).join("") : `<div class="empty-state compact">当前视频没有可展示的 V1 动作事实。</div>`;
}

function intervalText(interval, digits, unit) {
  return Array.isArray(interval) && interval.length === 2 ? `${Number(interval[0]).toFixed(digits)}–${Number(interval[1]).toFixed(digits)} ${unit}` : "未提供";
}

function renderMeasurementEvidence(measurement) {
  if (!measurement) {
    releaseFusionEvidence.innerHTML = `<div class="empty-state compact">后端未返回统一测量结果。</div>`;
    return;
  }
  const state = measurement.release_state;
  const trusted = measurement.trusted_flight;
  const epoch = state?.release_epoch;
  const uncertainty = state?.uncertainty || {};
  releaseFusionEvidence.innerHTML = `<div class="evidence-grid">
    <div><span>有效观测数</span><strong>${trusted?.point_count ?? measurement.evidence?.trusted_window_point_count ?? "未提供"}</strong></div>
    <div><span>Flight Support（飞行支持）</span><strong>${trusted?.temporal_span_ms != null ? `${(trusted.temporal_span_ms / 1000).toFixed(3)} 秒` : "未提供"}</strong></div>
    <div><span>TrustedFlight（可信飞行段）</span><strong>${trusted ? `第 ${trusted.start_frame}–${trusted.end_frame} 帧` : "未形成"}</strong></div>
    <div><span>ReleaseEpoch（出手时刻区间）</span><strong>${epoch ? `${Number(epoch.lower_time_s).toFixed(3)}–${Number(epoch.upper_time_s).toFixed(3)} 秒` : "未形成"}</strong></div>
    <div><span>速度 95% 区间</span><strong>${intervalText(uncertainty.speed_interval_mps, 2, "m/s（米/秒）")}</strong></div>
    <div><span>角度 95% 区间</span><strong>${intervalText(uncertainty.elevation_angle_interval_deg, 1, "°")}</strong></div>
  </div>`;
}

function renderBallTrackEvidence(evidence, measurement) {
  if (!evidence) {
    ballTrackEvidence.innerHTML = "";
    return;
  }
  const status = ui.trackStatus(evidence, measurement);
  const trusted = measurement?.trusted_flight;
  const epoch = measurement?.release_state?.release_epoch;
  const timeline = (evidence.frames || []).map((frame) => {
    const inTrusted = trusted && frame.frame_index >= trusted.start_frame && frame.frame_index <= trusted.end_frame;
    const inEpoch = epoch && frame.time_s >= epoch.lower_time_s && frame.time_s <= epoch.upper_time_s;
    return `<i class="track-tick ${frame.detections?.length ? "observed" : "missing"} ${inTrusted ? "trusted" : ""} ${inEpoch ? "epoch" : ""}" title="第 ${frame.frame_index} 帧"></i>`;
  }).join("");
  ballTrackEvidence.innerHTML = `<div class="track-evidence"><div><strong>连续飞行观测</strong>${statusLabel(status.state, status.label)}</div><p>${status.text}</p><div class="track-timeline">${timeline}</div><small>${evidence.requested_frame_count ?? 0} 个采集帧 · ${evidence.detection_frame_count ?? 0} 个检出帧 · ${evidence.missing_frame_count ?? 0} 个缺失帧</small></div>`;
}

function renderDeveloperDetails(report) {
  const evidence = report.release_ball_evidence;
  if (evidence) {
    releaseBallEvidence.innerHTML = `<div class="developer-grid"><div><span>Detector type（检测器类型）</span><strong>${evidence.detector_type || "release_ball_yolo"}</strong></div><div><span>Status（状态）</span><strong>${evidence.status || "unknown"}</strong></div><div><span>Window radius（窗口半径）</span><strong>${evidence.window_radius ?? "—"}</strong></div><div><span>Release frame index（出手帧索引）</span><strong>${evidence.release_frame_index ?? "—"}</strong></div></div>`;
  } else releaseBallEvidence.innerHTML = "";
  rawDeveloperDetails.textContent = JSON.stringify({ release_measurement: report.release_measurement, release_fusion: report.release_fusion, release_ball_evidence: report.release_ball_evidence, ball_track_evidence: report.ball_track_evidence }, null, 2);
}

function waitForMetadata(video) {
  return new Promise((resolve, reject) => {
    if (Number.isFinite(video.duration) && video.videoWidth > 0) return resolve();
    const timeout = setTimeout(() => reject(new Error("浏览器无法读取视频信息，请尝试 H.264 MP4 视频。")), 8000);
    video.onloadedmetadata = () => { clearTimeout(timeout); resolve(); };
    video.onerror = () => { clearTimeout(timeout); reject(new Error("浏览器无法播放该视频格式。")); };
  });
}

function seekVideo(video, time) {
  return new Promise((resolve) => {
    const onSeeked = () => { video.removeEventListener("seeked", onSeeked); resolve(); };
    video.addEventListener("seeked", onSeeked);
    video.currentTime = Math.min(Math.max(Number(time) || 0, 0), video.duration || 0);
  });
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

videoInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  selectedFile = file;
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = URL.createObjectURL(file);
  videoPreview.src = videoUrl;
  videoPreview.style.display = "block";
  emptyPreview.style.display = "none";
  analyzeButton.disabled = false;
  resetButton.disabled = false;
  setPipeline(1);
  try { await waitForMetadata(videoPreview); renderLocalSuitability(); } catch (error) {
    qualityList.innerHTML = `<div class="quality-item"><div><strong>视频预览不可用</strong><small>${error.message}</small></div>${statusLabel("danger", "无法读取")}</div>`;
  }
});

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  analyzeButton.textContent = "分析中…";
  try {
    setPipeline(2);
    const body = new FormData();
    body.append("file", selectedFile);
    const response = await fetch("/api/analyze", { method: "POST", body });
    if (!response.ok) throw new Error(`服务返回 ${response.status}`);
    const report = await response.json();
    setPipeline(3);
    renderSuitability(report);
    activeFrameIndex = Math.max(0, report.frames.findIndex((frame) => frame.key === "release"));
    renderFrames(report.frames);
    setPipeline(4);
    renderAnalysisSummary(report.release_measurement, report.ball_track_evidence);
    renderFlightSummary(report);
    await renderFlightVisualization(report);
    renderMetrics(report.metrics);
    renderMeasurementEvidence(report.release_measurement);
    renderBallTrackEvidence(report.ball_track_evidence, report.release_measurement);
    renderDeveloperDetails(report);
    setPipeline(5);
  } catch (error) {
    console.warn("Analysis failed（分析失败）", error);
    analysisSummary.className = "panel summary-panel error";
    analysisSummary.innerHTML = `<div><p class="eyebrow">Analysis Error（分析错误）</p><h2>这次分析没有完成</h2><p class="section-copy">${error.message}。视频仍可预览，请检查服务后重试。</p></div>`;
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "开始分析";
  }
});

resetButton.addEventListener("click", () => {
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = null;
  selectedFile = null;
  videoInput.value = "";
  videoPreview.removeAttribute("src");
  videoPreview.load();
  videoPreview.style.display = "none";
  emptyPreview.style.display = "block";
  analyzeButton.disabled = true;
  resetButton.disabled = true;
  framesGrid.innerHTML = "";
  metricsList.innerHTML = "";
  releaseBallEvidence.innerHTML = "";
  ballTrackEvidence.innerHTML = "";
  releaseFusionEvidence.innerHTML = "";
  rawDeveloperDetails.textContent = "";
  renderedFrames = [];
  activeFrameIndex = 0;
  closeImageModal();
  setPipeline(0);
  renderWaitingState();
});

imageModalClose.addEventListener("click", closeImageModal);
imageModal.addEventListener("click", (event) => { if (event.target === imageModal) closeImageModal(); });
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeImageModal(); });

setPipeline(0);
renderWaitingState();

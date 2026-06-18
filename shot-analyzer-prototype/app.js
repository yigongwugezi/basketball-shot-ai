const videoInput = document.getElementById("videoInput");
const analyzeButton = document.getElementById("analyzeButton");
const resetButton = document.getElementById("resetButton");
const videoPreview = document.getElementById("videoPreview");
const emptyPreview = document.getElementById("emptyPreview");
const qualityList = document.getElementById("qualityList");
const pipelineList = document.getElementById("pipelineList");
const framesGrid = document.getElementById("framesGrid");
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
      <button class="main-frame" type="button" data-full-frame="${active.dataUrl}" aria-label="查看${active.label}关键帧大图">
        <img src="${active.dataUrl}" alt="${active.label}关键帧" />
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
    openImageModal(active.dataUrl);
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
    renderFrames(report.frames);
    setPipeline(4);
    renderMetrics(report.metrics);
    setPipeline(5);
  } catch (error) {
    console.warn("Backend analysis failed; using browser fallback.", error);
    const frames = await extractKeyframes(videoPreview);
    setPipeline(3);
    activeFrameIndex = 0;
    renderFrames(frames);
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

(function exposeUiModel(root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root) root.BasketballUI = api;
})(typeof window !== "undefined" ? window : globalThis, () => {
  const reliabilityLabels = {
    HIGH: "HIGH（高可靠）",
    MEDIUM: "MEDIUM（中等可靠）",
    UNQUALIFIED: "UNQUALIFIED（未通过资格检查）",
  };

  const failureRules = [
    { pattern: /size|scale|reciprocal|metric.profile|fit/i, title: "球尺寸 / metric profile（度量剖面）不稳定", advice: "下次尽量保持近侧面机位，让篮球在出手后持续清晰可见。" },
    { pattern: /view|camera|near.side|out.of.plane/i, title: "拍摄角度超出当前支持范围", advice: "下次从球员侧面或近侧面稳定拍摄，减少明显的纵深方向球路。" },
    { pattern: /epoch|separation|contact/i, title: "出手时间区间不稳定", advice: "下次完整拍到持球、出手和出手后的连续飞行。" },
    { pattern: /uncertainty|stability|unstable|disagreement|unqualified/i, title: "测量不确定性过大", advice: "保持机位稳定，并让篮球在画面中更清晰、更连续。" },
    { pattern: /track|flight|temporal|support|observation|detection|missing/i, title: "篮球连续轨迹不足", advice: "下次完整拍到出手后至少约半秒的篮球飞行，并避免球被遮挡。" },
  ];

  function qualification(measurement) {
    const value = measurement?.release_state?.qualification;
    return value === "HIGH" || value === "MEDIUM" ? value : "UNQUALIFIED";
  }

  function failureReason(measurement, track) {
    const values = [measurement?.release_state?.reason, measurement?.reason, track?.reason, ...(measurement?.risk_flags || []), ...(measurement?.measurement_quality?.issues || []), ...(measurement?.measurement_quality?.warnings || []), ...(track?.warnings || [])].filter(Boolean);
    const rule = values.map((value) => failureRules.find((item) => item.pattern.test(value))).find(Boolean);
    if (rule) return { ...rule, raw: values };
    return { title: "当前证据不足以支持可靠物理量", advice: "保持近侧面机位稳定，并完整拍到球员与出手后的篮球飞行。", raw: values };
  }

  function analysisSummary(measurement, track) {
    const state = measurement?.release_state;
    const level = qualification(measurement);
    const qualified = level === "HIGH" || level === "MEDIUM";
    const flight = measurement?.trusted_flight;
    return {
      qualified,
      level,
      reliabilityLabel: reliabilityLabels[level],
      releaseTime: qualified ? state?.release_time ?? measurement?.release_time ?? null : null,
      speed: qualified ? state?.speed_mps ?? null : null,
      angle: qualified ? state?.elevation_angle_deg ?? null : null,
      supportMs: flight?.temporal_span_ms ?? null,
      observations: flight?.point_count ?? track?.detection_frame_count ?? 0,
      failure: qualified ? null : failureReason(measurement, track),
    };
  }

  function trackStatus(track, measurement) {
    const detected = Number(track?.detection_frame_count || 0);
    if (!track) return { state: "warn", label: "未启用", text: "连续飞行证据当前不可用。" };
    if (measurement?.trusted_flight) return { state: "ok", label: "已形成可信飞行段", text: "连续观测已支持 TrustedFlight（可信飞行段）。" };
    if (!detected) return { state: "warn", label: "未形成有效飞行轨迹", text: "采集窗口内没有形成可用于物理测量的连续篮球观测。" };
    return { state: "warn", label: "连续轨迹不足", text: "已观察到部分篮球位置，但连续性不足以形成 TrustedFlight（可信飞行段）。" };
  }

  return { reliabilityLabels, qualification, failureReason, analysisSummary, trackStatus };
});

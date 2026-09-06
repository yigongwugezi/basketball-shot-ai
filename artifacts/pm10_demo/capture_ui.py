from playwright.sync_api import sync_playwright


MEASUREMENT = {
    "status": "AVAILABLE",
    "release_frame": 169,
    "release_time": None,
    "source": "pose_release",
    "risk_flags": [],
    "measurement_quality": {"status": "AVAILABLE", "issues": [], "warnings": []},
    "trusted_flight": {
        "status": "AVAILABLE",
        "start_frame": 170,
        "end_frame": 184,
        "point_count": 15,
        "temporal_span_ms": 466.6667,
    },
    "release_state": {
        "status": "UNRELIABLE",
        "release_time": None,
        "speed_mps": None,
        "elevation_angle_deg": None,
        "qualification": "UNQUALIFIED",
        "release_epoch": {
            "lower_time_s": 5.6333333,
            "upper_time_s": 5.6666667,
            "representative_time_s": 5.65,
        },
        "reason": "Camera-space profile solutions are unstable; metric values are withheld.",
    },
}


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.goto("http://127.0.0.1:8000/", wait_until="networkidle")
    page.evaluate("measurement => renderReleaseMeasurement(measurement, null)", MEASUREMENT)
    panel = page.locator("#releaseFusionEvidence")
    panel.screenshot(path="artifacts/pm10_demo/IMG_7227_ui.png")
    print(panel.inner_text())
    browser.close()

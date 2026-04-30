import {
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const page = document.body;
const defaultExerciseName = page.dataset.defaultExerciseName || "Training";
const defaultFocus = page.dataset.defaultFocus || "Full Body";
const resultUrl = page.dataset.resultUrl || "/practice-score";
const selectedExerciseId = page.dataset.exerciseId || "";
const selectedMode = page.dataset.mode || "a";

const labels = {
  ready: page.dataset.labelReady || "Ready",
  starting: page.dataset.labelStarting || "Starting",
  tracking: page.dataset.labelTracking || "Tracking",
  paused: page.dataset.labelPaused || "Paused",
  running: page.dataset.labelRunning || "Running",
  error: page.dataset.labelError || "Error",
};

const messages = {
  default: page.dataset.messageDefault || "Click Start to begin live tracking.",
  fallback: page.dataset.messageFallback || "Keep practicing steadily.",
  error: page.dataset.messageError || "Unable to load live tracking.",
  secureContext:
    page.dataset.messageSecureContext ||
    "Camera access requires HTTPS or localhost.",
  cameraDenied:
    page.dataset.messageCameraDenied ||
    "Camera access was denied. Please allow camera permissions.",
  loadingCamera:
    page.dataset.messageLoadingCamera || "Preparing your camera and pose model.",
  moveIntoView:
    page.dataset.messageMoveIntoView ||
    "Move into the camera view so we can analyze your posture.",
};

const CONNECTIONS = [
  [11, 12],
  [11, 13], [13, 15],
  [12, 14], [14, 16],
  [11, 23], [12, 24],
  [23, 24],
  [23, 25], [25, 27],
  [24, 26], [26, 28],
  [27, 31],
  [28, 32],
];

const startBtn = document.getElementById("start-btn");
const pauseBtn = document.getElementById("pause-btn");
const resetBtn = document.getElementById("reset-btn");
const finishBtn = document.getElementById("finish-btn");
const audioToggleBtn = document.getElementById("audio-toggle-btn");

const referenceVideo = document.getElementById("reference-video");
const leftReferenceSkeleton = document.getElementById("left-reference-skeleton");
const cameraVideo = document.getElementById("camera-video");
const cameraCanvas = document.getElementById("camera-user-skeleton");
const rightReferenceOverlay = document.getElementById("right-reference-overlay");

const scoreValue = document.getElementById("score-value");
const scoreTop = document.getElementById("score-top");
const scoreRing = document.getElementById("score-ring");
const feedbackMessage = document.getElementById("feedback-message");
const feedbackMessageTop = document.getElementById("feedback-message-top");
const trackingStatus = document.getElementById("tracking-status");
const sessionState = document.getElementById("session-state");
const exerciseName = document.getElementById("exercise-name");
const focusName = document.getElementById("focus-name");
const focusNameTop = document.getElementById("focus-name-top");
const feedbackBanner = document.getElementById("feedback-banner");
const feedbackBannerText = document.getElementById("feedback-banner-text");
const audioIcon = audioToggleBtn ? audioToggleBtn.querySelector("[data-audio-icon]") : null;
const audioLabel = audioToggleBtn ? audioToggleBtn.querySelector("[data-audio-label]") : null;

const toggleReferenceSkeletonBtns = document.querySelectorAll("[data-action='toggle-reference-skeleton']");
const toggleReferenceOverlayBtns = document.querySelectorAll("[data-action='toggle-reference-overlay']");
const toggleUserSkeletonBtns = document.querySelectorAll("[data-action='toggle-user-skeleton']");

let pollingTimer = null;
let isRunning = false;
let isFetchingFrame = false;
let showReferenceSkeleton = true;
let showReferenceOverlay = true;
let showUserSkeleton = true;
let sessionStartedAt = null;
let sessionSaved = false;
let latestScore = 0;
let latestUserLandmarks = null;
let poseLandmarker = null;
let poseSetupPromise = null;
let mediaStream = null;
let isFetchingReferenceFrame = false;

function updateAudioButton() {
  if (!audioToggleBtn || !referenceVideo) {
    return;
  }

  const isMuted = referenceVideo.muted;
  audioToggleBtn.setAttribute("aria-pressed", isMuted ? "false" : "true");

  if (audioIcon) {
    audioIcon.textContent = isMuted ? "volume_off" : "volume_up";
  }

  if (audioLabel) {
    audioLabel.textContent = isMuted
      ? audioToggleBtn.dataset.labelOff || "Sound Off"
      : audioToggleBtn.dataset.labelOn || "Sound On";
  }
}

function setText(el, value) {
  if (el) {
    el.textContent = value;
  }
}

function setImage(el, dataUrlPrefix, payload) {
  if (el && payload) {
    el.src = `${dataUrlPrefix}${payload}`;
  }
}

function isCanvasElement(el) {
  return el && el.tagName && el.tagName.toLowerCase() === "canvas";
}

function syncElementCanvasSize(canvas, sourceEl) {
  if (!canvas) {
    return { width: 0, height: 0 };
  }

  const rect = canvas.getBoundingClientRect();
  const fallbackWidth = sourceEl?.clientWidth || rect.width || 720;
  const fallbackHeight = sourceEl?.clientHeight || rect.height || 1280;
  const width = Math.max(Math.round(rect.width || fallbackWidth), 1);
  const height = Math.max(Math.round(rect.height || fallbackHeight), 1);

  if (canvas.width !== width) {
    canvas.width = width;
  }
  if (canvas.height !== height) {
    canvas.height = height;
  }

  return { width, height };
}

function scaledReferencePoint(point, width, height) {
  if (!point) {
    return null;
  }

  return {
    x: Number(point.x ?? 0) * width,
    y: Number(point.y ?? 0) * height,
  };
}

function logReferenceSkeletonLayer() {
  if (!leftReferenceSkeleton) {
    console.log("[reference-skeleton] layer missing");
    return;
  }

  const rect = leftReferenceSkeleton.getBoundingClientRect();
  const videoRect = referenceVideo?.getBoundingClientRect?.();
  const styles = window.getComputedStyle(leftReferenceSkeleton);
  const parent = leftReferenceSkeleton.parentElement;
  const parentRect = parent?.getBoundingClientRect?.();
  const zeroSize = rect.width <= 0 || rect.height <= 0;
  const hiddenReasons = [];
  if (!isCanvasElement(leftReferenceSkeleton)) {
    hiddenReasons.push("wrong DOM node: reference skeleton layer is not a canvas");
  }
  if (styles.display === "none" || styles.visibility === "hidden" || Number(styles.opacity) === 0) {
    hiddenReasons.push("hidden layer");
  }
  if (zeroSize) {
    hiddenReasons.push("zero size");
  }
  if (parent && (parentRect.width <= 0 || parentRect.height <= 0)) {
    hiddenReasons.push("wrong parent container or zero-size parent");
  }

  console.log("[reference-skeleton] visible canvas verification", {
    id: leftReferenceSkeleton.id,
    className: leftReferenceSkeleton.className,
    tagName: leftReferenceSkeleton.tagName,
    display: styles.display,
    opacity: styles.opacity,
    zIndex: styles.zIndex,
    pointerEvents: styles.pointerEvents,
    zeroSize,
    hiddenReasons,
    canvasWidth: leftReferenceSkeleton.width,
    canvasHeight: leftReferenceSkeleton.height,
    canvasRect: {
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height,
    },
    videoVideoWidth: referenceVideo?.videoWidth || 0,
    videoVideoHeight: referenceVideo?.videoHeight || 0,
    videoClientWidth: referenceVideo?.clientWidth || 0,
    videoClientHeight: referenceVideo?.clientHeight || 0,
    videoRect: videoRect ? {
      x: videoRect.x,
      y: videoRect.y,
      width: videoRect.width,
      height: videoRect.height,
    } : null,
    parentId: parent?.id || "",
    parentClassName: parent?.className || "",
    parentRect: parentRect ? {
      x: parentRect.x,
      y: parentRect.y,
      width: parentRect.width,
      height: parentRect.height,
    } : null,
  });

  console.log("[reference-skeleton] layer state", {
    display: styles.display,
    visibility: styles.visibility,
    opacity: styles.opacity,
    zIndex: styles.zIndex,
    width: rect.width,
    height: rect.height,
    hasSrc: Boolean(leftReferenceSkeleton.getAttribute("src")),
    mode: selectedMode,
    enabled: showReferenceSkeleton,
  });
}

function drawReferenceCanvasProbe(data) {
  console.log("PROBE FUNCTION ENTERED");
  if (!isCanvasElement(leftReferenceSkeleton)) {
    console.warn("[reference-skeleton] forced visual test cannot draw: visible layer is not a canvas", {
      id: leftReferenceSkeleton?.id || "",
      tagName: leftReferenceSkeleton?.tagName || "",
    });
    setImage(leftReferenceSkeleton, "data:image/png;base64,", data.reference_image);
    return;
  }

  const ctx = leftReferenceSkeleton.getContext("2d");
  const { width, height } = syncElementCanvasSize(leftReferenceSkeleton, referenceVideo);
  const landmarks = Array.isArray(data?.reference_landmarks) ? data.reference_landmarks : [];
  const selectedIndexes = [0, 11, 12];
  const selectedLandmarks = selectedIndexes.map((index) => landmarks[index] || null);
  const scaledPoints = selectedLandmarks.map((point) => scaledReferencePoint(point, width, height));
  const first3Landmarks = landmarks.slice(0, 3);
  const first3ScaledPoints = first3Landmarks.map((point) => scaledReferencePoint(point, width, height));

  ctx.clearRect(0, 0, width, height);

  if (!showReferenceSkeleton) {
    console.log("[reference-skeleton] draw skipped after clear because video skeleton mode is inactive");
    return;
  }

  ctx.save();
  ctx.strokeStyle = "rgba(255,0,0,1)";
  ctx.lineWidth = 8;
  ctx.strokeRect(4, 4, width - 8, height - 8);

  ctx.fillStyle = "rgba(0,255,0,1)";
  ctx.beginPath();
  ctx.arc(width / 2, height / 2, 18, 0, Math.PI * 2);
  ctx.fill();

  ctx.font = "700 16px sans-serif";
  ctx.textBaseline = "top";
  ctx.lineWidth = 5;
  ctx.strokeStyle = "rgba(0,0,0,0.85)";
  ctx.strokeText("REFERENCE CANVAS ACTIVE", 12, 12);
  ctx.fillStyle = "rgba(255,255,255,1)";
  ctx.fillText("REFERENCE CANVAS ACTIVE", 12, 12);

  selectedLandmarks.forEach((point, offset) => {
    const scaled = scaledPoints[offset];
    if (!point || !scaled) {
      return;
    }
    ctx.beginPath();
    ctx.arc(scaled.x, scaled.y, 14, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,0,255,0.95)";
    ctx.fill();
    ctx.lineWidth = 5;
    ctx.strokeStyle = "rgba(255,255,0,1)";
    ctx.stroke();
  });
  ctx.restore();

  console.log("[reference-skeleton] forced draw geometry", {
    canvasWidth: leftReferenceSkeleton.width,
    canvasHeight: leftReferenceSkeleton.height,
    canvasRect: leftReferenceSkeleton.getBoundingClientRect(),
    videoVideoWidth: referenceVideo?.videoWidth || 0,
    videoVideoHeight: referenceVideo?.videoHeight || 0,
    videoClientWidth: referenceVideo?.clientWidth || 0,
    videoClientHeight: referenceVideo?.clientHeight || 0,
    first3LandmarkCoordinatesBeforeScaling: first3Landmarks,
    first3ScaledPixelCoordinates: first3ScaledPoints,
    renderedLandmarkCoordinatesBeforeScaling: selectedLandmarks,
    renderedScaledPixelCoordinates: scaledPoints,
    drawnLandmarkIndexes: selectedIndexes,
    drawnLandmarkCount: scaledPoints.filter(Boolean).length,
  });
}

function drawReferenceSkeleton(data) {
  console.log("[reference-skeleton] drawReferenceSkeleton", {
    hasReferencePose: data?.has_reference_pose,
    landmarkCount: data?.reference_landmark_count,
    imageLength: data?.reference_image?.length || 0,
    rawLandmarkCount: data?.reference_landmarks?.length || 0,
  });
  applyLayerVisibility();
  drawReferenceCanvasProbe(data);
  logReferenceSkeletonLayer();
}

function clearImage(el) {
  if (!el) {
    return;
  }

  if (isCanvasElement(el)) {
    const ctx = el.getContext("2d");
    ctx.clearRect(0, 0, el.width, el.height);
  } else {
    el.src = "";
  }
}

function setButtonState(buttons, active) {
  buttons.forEach((btn) => {
    btn.classList.toggle("is-active", active);
    btn.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

function applyLayerVisibility() {
  if (leftReferenceSkeleton) {
    leftReferenceSkeleton.style.display = showReferenceSkeleton ? "block" : "none";
  }
  if (rightReferenceOverlay) {
    rightReferenceOverlay.style.display = showReferenceOverlay ? "block" : "none";
  }
  if (cameraCanvas) {
    cameraCanvas.style.display = showUserSkeleton ? "block" : "none";
  }

  setButtonState(toggleReferenceSkeletonBtns, showReferenceSkeleton);
  setButtonState(toggleReferenceOverlayBtns, showReferenceOverlay);
  setButtonState(toggleUserSkeletonBtns, showUserSkeleton);
}

function updateUI(data) {
  console.log("Received pose data:", data);
  drawReferenceSkeleton(data);
  setImage(rightReferenceOverlay, "data:image/png;base64,", data.reference_overlay_image);

  const score = Math.round(data.score || 0);
  latestScore = score;
  setText(scoreValue, String(score));
  setText(scoreTop, String(score));
  if (scoreRing) {
    scoreRing.style.setProperty("--score", `${score}%`);
  }

  const message = data.message || messages.fallback;
  setText(feedbackMessage, message);
  setText(feedbackMessageTop, message);
  setText(trackingStatus, isRunning ? labels.tracking : labels.paused);
  setText(sessionState, isRunning ? labels.running : labels.paused);
  setText(exerciseName, data.exercise_name || defaultExerciseName);

  const readableFocus = (data.focus || defaultFocus)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

  setText(focusName, readableFocus);
  setText(focusNameTop, readableFocus);

  if (feedbackBanner && feedbackBannerText) {
    if (message.trim()) {
      feedbackBanner.classList.remove("hidden");
      feedbackBannerText.textContent = message;
    } else {
      feedbackBanner.classList.add("hidden");
    }
  }
}

async function fetchReferenceSkeletonFrame() {
  if (!showReferenceSkeleton || isFetchingReferenceFrame || !referenceVideo) {
    console.log("[reference-skeleton] fetch skipped", {
      showReferenceSkeleton,
      isFetchingReferenceFrame,
      hasReferenceVideo: Boolean(referenceVideo),
    });
    return;
  }

  console.log("[reference-skeleton] /api/reference-frame request", {
    referenceTime: referenceVideo.currentTime || 0,
    mode: selectedMode,
  });
  isFetchingReferenceFrame = true;

  try {
    const response = await fetch("/api/reference-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reference_time: referenceVideo.currentTime || 0,
      }),
    });

    const rawText = await response.text();
    let data;
    try {
      data = rawText ? JSON.parse(rawText) : {};
    } catch (parseError) {
      console.error("[reference-skeleton] failed to parse /api/reference-frame response", {
        status: response.status,
        statusText: response.statusText,
        rawText,
        parseError,
      });
      throw parseError;
    }

    console.log("[reference-skeleton] /api/reference-frame response", {
      ok: response.ok,
      status: response.status,
      statusText: response.statusText,
      hasReferencePose: data.has_reference_pose,
      landmarkCount: data.reference_landmark_count,
      processingMs: data.processing_ms,
      imageLength: data.reference_image?.length || 0,
      error: data.error,
      referenceDebug: data.reference_debug,
      fullPayload: data,
    });
    if (!response.ok) {
      throw new Error(data.error || "Failed to load reference skeleton");
    }

    drawReferenceSkeleton(data);
  } catch (error) {
    console.error("Failed to load reference skeleton", {
      error,
      referenceSrc: referenceVideo?.currentSrc || referenceVideo?.querySelector("source")?.src || "",
      referenceTime: referenceVideo?.currentTime || 0,
    });
  } finally {
    isFetchingReferenceFrame = false;
  }
}

function showError(message) {
  const errorText = message || messages.error;
  setText(trackingStatus, labels.error);
  setText(sessionState, labels.paused);
  setText(feedbackMessage, errorText);
  setText(feedbackMessageTop, errorText);
  if (feedbackBanner && feedbackBannerText) {
    feedbackBanner.classList.remove("hidden");
    feedbackBannerText.textContent = errorText;
  }
}

function clearCanvas() {
  if (!cameraCanvas) {
    return;
  }
  const ctx = cameraCanvas.getContext("2d");
  ctx.clearRect(0, 0, cameraCanvas.width, cameraCanvas.height);
}

function syncCanvasSize() {
  if (!cameraCanvas || !cameraVideo) {
    return { width: 720, height: 1280 };
  }

  const width = Math.max(Math.round(cameraVideo.clientWidth || 720), 1);
  const height = Math.max(Math.round(cameraVideo.clientHeight || 1280), 1);

  if (cameraCanvas.width !== width) {
    cameraCanvas.width = width;
  }
  if (cameraCanvas.height !== height) {
    cameraCanvas.height = height;
  }

  return { width, height };
}

function drawUserSkeleton(landmarks) {
  if (!cameraCanvas) {
    return;
  }

  latestUserLandmarks = landmarks;
  const ctx = cameraCanvas.getContext("2d");
  const { width, height } = syncCanvasSize();
  ctx.clearRect(0, 0, width, height);

  if (!landmarks || !showUserSkeleton) {
    return;
  }

  ctx.lineWidth = Math.max(4, Math.round(Math.min(width, height) / 160));
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  CONNECTIONS.forEach(([a, b]) => {
    const start = landmarks[a];
    const end = landmarks[b];
    if (!start || !end) {
      return;
    }
    if ((start.visibility ?? 1) < 0.5 || (end.visibility ?? 1) < 0.5) {
      return;
    }
    ctx.strokeStyle = "rgba(60,220,120,0.95)";
    ctx.beginPath();
    ctx.moveTo(start.x * width, start.y * height);
    ctx.lineTo(end.x * width, end.y * height);
    ctx.stroke();
  });

  landmarks.forEach((point) => {
    if (!point || (point.visibility ?? 1) < 0.5) {
      return;
    }
    const x = point.x * width;
    const y = point.y * height;
    const radius = Math.max(5, Math.round(Math.min(width, height) / 110));
    ctx.fillStyle = "rgba(0,255,120,1)";
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.95)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, radius + 2, 0, Math.PI * 2);
    ctx.stroke();
  });
}

function sanitizeLandmarks(landmarks) {
  return landmarks.map((point) => ({
    x: Number(point.x ?? 0),
    y: Number(point.y ?? 0),
    z: Number(point.z ?? 0),
    visibility: Number(point.visibility ?? 0),
  }));
}

async function ensureCameraStream() {
  if (mediaStream) {
    return;
  }

  if (!window.isSecureContext && !["localhost", "127.0.0.1"].includes(window.location.hostname)) {
    throw new Error(messages.secureContext);
  }

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 720 },
      height: { ideal: 1280 },
    },
  });

  cameraVideo.srcObject = mediaStream;

  await new Promise((resolve) => {
    if (cameraVideo.readyState >= 1) {
      resolve();
      return;
    }
    cameraVideo.onloadedmetadata = () => resolve();
  });

  await cameraVideo.play();
  syncCanvasSize();
}

async function ensurePoseLandmarker() {
  if (poseLandmarker) {
    return poseLandmarker;
  }

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "/assets/pose-landmarker.task",
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  return poseLandmarker;
}

async function prepareLiveTracking() {
  if (poseSetupPromise) {
    return poseSetupPromise;
  }

  setText(trackingStatus, labels.starting);
  setText(feedbackMessage, messages.loadingCamera);
  setText(feedbackMessageTop, messages.loadingCamera);

  poseSetupPromise = (async () => {
    try {
      await ensureCameraStream();
      await ensurePoseLandmarker();
    } catch (error) {
      poseSetupPromise = null;
      throw error;
    }
  })();

  return poseSetupPromise;
}

async function fetchFrame() {
  if (isFetchingFrame || !poseLandmarker || !cameraVideo) {
    return;
  }

  isFetchingFrame = true;

  try {
    syncCanvasSize();
    const detection = poseLandmarker.detectForVideo(cameraVideo, performance.now());
    const landmarks = detection.landmarks?.[0] || null;

    drawUserSkeleton(landmarks);

    if (!landmarks) {
      await fetchReferenceSkeletonFrame();
      latestScore = 0;
      if (scoreRing) {
        scoreRing.style.setProperty("--score", "0%");
      }
      setText(scoreValue, "0");
      setText(scoreTop, "0");
      setText(feedbackMessage, messages.moveIntoView);
      setText(feedbackMessageTop, messages.moveIntoView);
      return;
    }

    const response = await fetch("/api/score-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reference_time: referenceVideo ? referenceVideo.currentTime || 0 : 0,
        frame_width: cameraCanvas.width,
        frame_height: cameraCanvas.height,
        user_landmarks: sanitizeLandmarks(landmarks),
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Failed to score frame");
    }

    updateUI(data);
  } catch (error) {
    const permissionDenied = error && (error.name === "NotAllowedError" || error.name === "PermissionDeniedError");
    showError(permissionDenied ? messages.cameraDenied : (error.message || messages.error));
  } finally {
    isFetchingFrame = false;
  }
}

function scheduleNextFetch(delay = 140) {
  if (!isRunning) {
    return;
  }

  if (pollingTimer) {
    window.clearTimeout(pollingTimer);
  }

  pollingTimer = window.setTimeout(async () => {
    pollingTimer = null;
    await fetchFrame();
    scheduleNextFetch();
  }, delay);
}

async function startPolling() {
  if (isRunning) {
    return;
  }

  try {
    await prepareLiveTracking();
  } catch (error) {
    const permissionDenied = error && (error.name === "NotAllowedError" || error.name === "PermissionDeniedError");
    showError(permissionDenied ? messages.cameraDenied : (error.message || messages.error));
    return;
  }

  isRunning = true;
  if (!sessionStartedAt) {
    sessionStartedAt = Date.now();
  }
  setText(sessionState, labels.running);
  setText(trackingStatus, labels.starting);

  if (referenceVideo && referenceVideo.paused) {
    referenceVideo.play().catch(() => {});
  }

  fetchReferenceSkeletonFrame();

  fetchFrame().finally(() => {
    scheduleNextFetch();
  });
}

function stopPolling() {
  isRunning = false;
  setText(sessionState, labels.paused);
  setText(trackingStatus, labels.paused);

  if (pollingTimer) {
    window.clearTimeout(pollingTimer);
    pollingTimer = null;
  }

  if (referenceVideo && !referenceVideo.paused) {
    referenceVideo.pause();
  }
}

function resetSession() {
  stopPolling();
  sessionStartedAt = null;
  sessionSaved = false;
  latestScore = 0;
  latestUserLandmarks = null;

  if (referenceVideo) {
    referenceVideo.currentTime = 0;
  }

  setText(scoreValue, "0");
  setText(scoreTop, "0");
  if (scoreRing) {
    scoreRing.style.setProperty("--score", "0%");
  }

  setText(feedbackMessage, messages.default);
  setText(feedbackMessageTop, messages.default);

  if (feedbackBanner) {
    feedbackBanner.classList.add("hidden");
  }

  clearCanvas();
  clearImage(leftReferenceSkeleton);
  clearImage(rightReferenceOverlay);

  setText(trackingStatus, labels.ready);
  setText(sessionState, labels.paused);
  setText(exerciseName, defaultExerciseName);
  setText(focusName, defaultFocus);
  setText(focusNameTop, defaultFocus);

  applyLayerVisibility();
}

async function persistTrainingSession() {
  if (sessionSaved || !sessionStartedAt) {
    return;
  }

  const durationSec = Math.max(Math.round((Date.now() - sessionStartedAt) / 1000), 0);

  try {
    await fetch("/api/training-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        exercise_id: selectedExerciseId,
        score: latestScore,
        duration_sec: durationSec,
        mode: selectedMode,
        focus: defaultFocus.toLowerCase().replaceAll(" ", "_"),
        feedback_message: feedbackMessage ? feedbackMessage.textContent.trim() : "",
      }),
    });
    sessionSaved = true;
  } catch (error) {
    console.error("Failed to save training session", error);
  }
}

async function goToResult() {
  stopPolling();
  await persistTrainingSession();
  page.classList.add("page-exit");
  window.setTimeout(() => {
    window.location.href = resultUrl;
  }, 220);
}

function cleanupMedia() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

if (startBtn) {
  startBtn.addEventListener("click", () => {
    console.log("[training] Start clicked", {
      exerciseId: selectedExerciseId,
      mode: selectedMode,
      referenceSrc: referenceVideo?.currentSrc || referenceVideo?.querySelector("source")?.src || "",
    });
    startPolling();
  });
}

if (pauseBtn) {
  pauseBtn.addEventListener("click", stopPolling);
}

if (resetBtn) {
  resetBtn.addEventListener("click", resetSession);
}

if (finishBtn) {
  finishBtn.addEventListener("click", goToResult);
}

if (audioToggleBtn && referenceVideo) {
  audioToggleBtn.addEventListener("click", async () => {
    referenceVideo.muted = !referenceVideo.muted;

    if (!referenceVideo.paused) {
      try {
        await referenceVideo.play();
      } catch (error) {
        console.error("Failed to resume reference video audio", error);
      }
    }

    updateAudioButton();
  });
}

toggleReferenceSkeletonBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    showReferenceSkeleton = !showReferenceSkeleton;
    applyLayerVisibility();
    if (showReferenceSkeleton) {
      fetchReferenceSkeletonFrame();
    }
  });
});

toggleReferenceOverlayBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    showReferenceOverlay = !showReferenceOverlay;
    applyLayerVisibility();
  });
});

  toggleUserSkeletonBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    showUserSkeleton = !showUserSkeleton;
    drawUserSkeleton(latestUserLandmarks);
    applyLayerVisibility();
  });
});

if (referenceVideo) {
  referenceVideo.muted = false;
  referenceVideo.volume = 1;
  referenceVideo.addEventListener("ended", stopPolling);
  referenceVideo.addEventListener("timeupdate", fetchReferenceSkeletonFrame);
}

window.addEventListener("resize", () => {
  syncCanvasSize();
});

window.addEventListener("beforeunload", cleanupMedia);

applyLayerVisibility();
updateAudioButton();
resetSession();

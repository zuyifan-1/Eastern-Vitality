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

function clearImage(el) {
  if (el) {
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
  setImage(leftReferenceSkeleton, "data:image/png;base64,", data.reference_image);
  setImage(rightReferenceOverlay, "data:image/png;base64,", data.reference_overlay_image);
  applyLayerVisibility();

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
  startBtn.addEventListener("click", startPolling);
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
}

window.addEventListener("resize", () => {
  syncCanvasSize();
});

window.addEventListener("beforeunload", cleanupMedia);

applyLayerVisibility();
updateAudioButton();
resetSession();

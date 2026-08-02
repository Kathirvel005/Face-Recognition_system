import './style.css';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as faceapi from '@vladmandic/face-api';

// --- State Variables ---
let cocoModel = null;
let modelsLoaded = false;
let activeStream = null;
let isProcessing = false;
let totalScannedCount = 0;
let lastDetectedLabels = new Set();
let announcedLabels = {}; // label -> timestamp
let snapshots = [];

// Performance metrics
let fps = 0;
let lastFrameTime = performance.now();
let frameCount = 0;
let fpsInterval = setInterval(() => {
  document.getElementById('overlay-fps').textContent = String(fps).padStart(2, '0');
  document.getElementById('stat-total-scanned').textContent = String(totalScannedCount).padStart(3, '0');
  fps = frameCount;
  frameCount = 0;
}, 1000);

// App settings
const settings = {
  confidenceThreshold: 0.50,
  filters: {
    human: true,
    animal: true,
    object: true
  },
  voiceEnabled: true,
  voiceRate: 1.0,
  selectedVoiceName: ''
};

// Voices list
let availableVoices = [];

// --- DOM Elements ---
const videoEl = document.getElementById('webcam-video');
const canvasEl = document.getElementById('detection-canvas');
const ctx = canvasEl.getContext('2d');
const loaderScreen = document.getElementById('hud-loader-screen');
const loaderStatus = document.getElementById('loader-status-text');
const loaderProgress = document.getElementById('loader-progress-bar');
const errorScreen = document.getElementById('hud-error-screen');
const errorMessage = document.getElementById('error-message-text');
const retryCameraBtn = document.getElementById('retry-camera-btn');

// Stats Elements
const statActiveScans = document.getElementById('stat-active-scans');
const statBreakdown = document.getElementById('stat-breakdown');
const countHuman = document.getElementById('count-human');
const countAnimal = document.getElementById('count-animal');
const countObject = document.getElementById('count-object');

// Control Elements
const cameraSelect = document.getElementById('camera-select');
const filterHuman = document.getElementById('filter-human');
const filterAnimal = document.getElementById('filter-animal');
const filterObject = document.getElementById('filter-object');
const confidenceThreshold = document.getElementById('confidence-threshold');
const confidenceVal = document.getElementById('confidence-threshold-val');
const toggleVoice = document.getElementById('toggle-voice');
const voiceSettingsContainer = document.getElementById('voice-settings-container');
const voiceSelect = document.getElementById('voice-select');
const voiceRate = document.getElementById('voice-rate');
const voiceRateVal = document.getElementById('voice-rate-val');

// Log & Gallery Elements
const terminalLog = document.getElementById('terminal-log');
const clearLogsBtn = document.getElementById('clear-logs-btn');
const captureBtn = document.getElementById('capture-snapshot-btn');
const snapshotGallery = document.getElementById('snapshot-gallery');
const emptyGalleryMsg = document.getElementById('empty-gallery-msg');

// Target Display Elements
const activeTargetCard = document.getElementById('hud-active-target');
const activeTargetName = document.getElementById('active-target-name');
const diagnosticsTargetCard = document.getElementById('diagnostics-target-card');
const diagnosticsTargetVal = document.getElementById('diagnostics-active-target');

// --- Helper Utilities ---

// Write time-stamped text to HUD log terminal
function logToTerminal(text, type = 'muted') {
  const time = new Date().toLocaleTimeString('en-US', { hour12: false });
  const logLine = document.createElement('div');
  logLine.className = `log-line text-${type}`;
  logLine.textContent = `[${time}] ${text}`;
  terminalLog.appendChild(logLine);
  terminalLog.scrollTop = terminalLog.scrollHeight;
  
  // Prune log to max 50 lines to avoid scroll lag
  while (terminalLog.children.length > 50) {
    terminalLog.removeChild(terminalLog.firstChild);
  }
}

// Play synthesizer beep sound using Web Audio API
function playSynthBeep(frequency = 800, duration = 0.1, type = 'sine') {
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    
    osc.type = type;
    osc.frequency.setValueAtTime(frequency, audioCtx.currentTime);
    
    // Quick ramp down to prevent popping sound
    gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (err) {
    console.warn("Audio Context blocked or unsupported:", err);
  }
}

// Speak out loud (TTS) with rate limiting
function speakText(text) {
  if (!settings.voiceEnabled) return;
  
  // If synthesis is currently speaking, don't interrupt it, just skip
  if (window.speechSynthesis.speaking) return;
  
  const utterance = new SpeechSynthesisUtterance(text);
  if (settings.selectedVoiceName) {
    const voice = availableVoices.find(v => v.name === settings.selectedVoiceName);
    if (voice) utterance.voice = voice;
  }
  utterance.rate = settings.voiceRate;
  window.speechSynthesis.speak(utterance);
}

// Populate system voices
function initVoices() {
  if (typeof window === 'undefined' || !window.speechSynthesis) return;
  
  const loadVoices = () => {
    availableVoices = window.speechSynthesis.getVoices();
    voiceSelect.innerHTML = '<option value="">DEFAULT SYSTEM</option>';
    availableVoices.forEach(voice => {
      const option = document.createElement('option');
      option.value = voice.name;
      option.textContent = `${voice.name} (${voice.lang})`;
      voiceSelect.appendChild(option);
    });
  };
  
  loadVoices();
  if (window.speechSynthesis.onvoiceschanged !== undefined) {
    window.speechSynthesis.onvoiceschanged = loadVoices;
  }
}

// Update clock in HUD header
function updateClock() {
  const timeEl = document.getElementById('hud-time');
  const now = new Date();
  timeEl.textContent = now.toLocaleTimeString('en-US', { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

// Define Category and visual styling
const ANIMAL_CLASSES = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'];

function parseDetectionClass(label) {
  if (label === 'person') {
    return { category: 'human', label: 'Person', color: '#10b981' };
  } else if (ANIMAL_CLASSES.includes(label)) {
    return { category: 'animal', label: label.charAt(0).toUpperCase() + label.slice(1), color: '#f59e0b' };
  } else {
    return { category: 'object', label: label.charAt(0).toUpperCase() + label.slice(1), color: '#d946ef' };
  }
}

// --- Video Stream Acquisition ---
async function startWebcam() {
  // Stop existing streams if any
  if (activeStream) {
    activeStream.getTracks().forEach(track => track.stop());
  }

  const selectedDeviceId = cameraSelect.value;
  const useExactId = selectedDeviceId && selectedDeviceId !== "LOADING DEVICES..." && selectedDeviceId !== "NO CAMERAS DETECTED";
  
  const constraints = {
    video: useExactId ? {
      deviceId: { exact: selectedDeviceId },
      width: { ideal: 640 },
      height: { ideal: 480 }
    } : {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: 'user' // default to front camera
    },
    audio: false
  };

  try {
    logToTerminal("Connecting to webcam interface...", "cyan");
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoEl.srcObject = stream;
    activeStream = stream;
    errorScreen.classList.add('hidden');
    
    // Wait for video metadata to load so dimensions are correct
    await new Promise((resolve) => {
      videoEl.onloadedmetadata = () => {
        resolve();
      };
    });
    
    // Adjust canvas sizes
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
    document.getElementById('overlay-resolution').textContent = `${videoEl.videoWidth}x${videoEl.videoHeight}`;
    logToTerminal(`Webcam stream active at ${videoEl.videoWidth}x${videoEl.videoHeight}`, "cyan");
    
    // Populate camera devices now that we have permission
    await populateCameraDevices();
    
    // Auto-select the active camera track in the dropdown
    const activeTrack = stream.getVideoTracks()[0];
    if (activeTrack) {
      const settings = activeTrack.getSettings();
      if (settings.deviceId) {
        const optionExists = Array.from(cameraSelect.options).some(o => o.value === settings.deviceId);
        if (optionExists) {
          cameraSelect.value = settings.deviceId;
        }
      }
    }
    
    // Play subtle high tech chime
    playSynthBeep(600, 0.1, 'sine');
    setTimeout(() => playSynthBeep(900, 0.15, 'sine'), 100);
  } catch (err) {
    console.error("Camera error: ", err);
    logToTerminal(`CAMERA ERROR: ${err.message}`, "object");
    errorMessage.textContent = `Access denied or camera not found. Ensure camera permissions are granted. (Details: ${err.message})`;
    errorScreen.classList.remove('hidden');
  }
}

async function populateCameraDevices() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    
    cameraSelect.innerHTML = '';
    
    if (videoDevices.length === 0) {
      const option = document.createElement('option');
      option.value = "";
      option.textContent = "NO CAMERAS DETECTED";
      cameraSelect.appendChild(option);
      return;
    }
    
    videoDevices.forEach((device, index) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.textContent = device.label || `Camera ${index + 1}`;
      cameraSelect.appendChild(option);
    });
    
    logToTerminal(`Detected ${videoDevices.length} camera input source(s).`, "muted");
  } catch (err) {
    console.warn("Failed to enumerate devices:", err);
    logToTerminal("Failed to list camera devices.", "object");
  }
}

// --- AI Core Initializer ---
async function initializeAICore() {
  try {
    // 1. Ready TensorFlow
    loaderStatus.textContent = "INITIALIZING TENSORFLOW CORE...";
    loaderProgress.style.width = "10%";
    loaderProgress.textContent = "10%";
    await tf.ready();
    
    // Use WebGL backend
    if (tf.getBackend() !== 'webgl') {
      await tf.setBackend('webgl');
    }
    document.getElementById('webgl-status').textContent = `WEBGL: ${tf.getBackend().toUpperCase()}`;
    
    // 2. Load COCO SSD
    loaderStatus.textContent = "LOADING NEURAL OBJECT DETECTOR (COCO-SSD)...";
    loaderProgress.style.width = "25%";
    loaderProgress.textContent = "25%";
    cocoModel = await cocoSsd.load();
    logToTerminal("COCO-SSD neural network loaded.", "cyan");
    
    // 3. Load Face-API Models
    const modelPath = import.meta.env.BASE_URL + 'models';
    
    loaderStatus.textContent = "LOADING LIGHTWEIGHT BIOMETRICS (TINY DETECTOR)...";
    loaderProgress.style.width = "40%";
    loaderProgress.textContent = "40%";
    await faceapi.nets.tinyFaceDetector.loadFromUri(modelPath);
    
    loaderStatus.textContent = "LOADING ACCURATE BIOMETRICS (SSD MOBILENET)...";
    loaderProgress.style.width = "55%";
    loaderProgress.textContent = "55%";
    await faceapi.nets.ssdMobilenetv1.loadFromUri(modelPath);
    
    loaderStatus.textContent = "LOADING FACE MESH NETWORK...";
    loaderProgress.style.width = "70%";
    loaderProgress.textContent = "70%";
    await faceapi.nets.faceLandmark68Net.loadFromUri(modelPath);
    
    loaderStatus.textContent = "LOADING BIOMETRIC CLASSIFIERS (AGE, GENDER, EMOTIONS)...";
    loaderProgress.style.width = "85%";
    loaderProgress.textContent = "85%";
    await faceapi.nets.faceExpressionNet.loadFromUri(modelPath);
    await faceapi.nets.ageGenderNet.loadFromUri(modelPath);
    
    loaderStatus.textContent = "LOADING DEEP IDENTITY RECOGNITION CORE...";
    loaderProgress.style.width = "95%";
    loaderProgress.textContent = "95%";
    await faceapi.nets.faceRecognitionNet.loadFromUri(modelPath);
    
    loaderProgress.style.width = "100%";
    loaderProgress.textContent = "100%";
    loaderStatus.textContent = "AI CORE ONLINE. DEPLOYING RADAR...";
    
    logToTerminal("Face-API.js models loaded successfully.", "cyan");
    
    // Mark engine online in header
    const engineStatus = document.getElementById('ai-engine-status');
    engineStatus.textContent = "AI CORE: ONLINE";
    engineStatus.parentElement.classList.add('online');
    
    modelsLoaded = true;
    
    // Hide loader
    setTimeout(() => {
      loaderScreen.style.display = 'none';
    }, 800);
    
    logToTerminal("All neural networks online and ready.", "cyan");
  } catch (err) {
    console.error("AI Model Load Failure: ", err);
    loaderStatus.textContent = "ERROR: NEURAL MODEL CORRUPTION";
    loaderStatus.style.color = "#ef4444";
    logToTerminal(`AI MODEL LOAD ERROR: ${err.message}`, "object");
  }
}

// --- Face mesh connection utility ---
function drawFaceMesh(landmarks) {
  ctx.strokeStyle = 'rgba(0, 242, 254, 0.3)';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#00f2fe';
  
  const positions = landmarks.positions;
  
  // Helper to translate x coordinate if webcam is mirrored
  function getDrawX(ptX) {
    return isWebcamMode ? (canvasEl.width - ptX) : ptX;
  }
  
  // Helper to connect points in path
  function drawPath(indices, close = false) {
    ctx.beginPath();
    ctx.moveTo(getDrawX(positions[indices[0]].x), positions[indices[0]].y);
    for (let i = 1; i < indices.length; i++) {
      ctx.lineTo(getDrawX(positions[indices[i]].x), positions[indices[i]].y);
    }
    if (close) ctx.closePath();
    ctx.stroke();
  }
  
  // Jawline (0-16)
  drawPath([...Array(17).keys()]);
  
  // Left eyebrow (17-21)
  drawPath([17, 18, 19, 20, 21]);
  
  // Right eyebrow (22-26)
  drawPath([22, 23, 24, 25, 26]);
  
  // Nose Bridge (27-30)
  drawPath([27, 28, 29, 30]);
  
  // Nose bottom (30-35)
  drawPath([30, 31, 32, 33, 34, 35], true);
  
  // Left eye (36-41)
  drawPath([36, 37, 38, 39, 40, 41], true);
  
  // Right eye (42-47)
  drawPath([42, 43, 44, 45, 46, 47], true);
  
  // Lips Outer (48-59)
  drawPath([...Array(12).keys()].map(i => i + 48), true);
  
  // Lips Inner (60-67)
  drawPath([...Array(8).keys()].map(i => i + 60), true);
  
  // Draw landmark points
  positions.forEach(point => {
    ctx.beginPath();
    ctx.arc(getDrawX(point.x), point.y, 1.5, 0, 2 * Math.PI);
    ctx.fill();
  });
}

// --- HUD Canvas Drawing Utilities ---
function drawBoundingBoxCorners(x, y, width, height, color, label, score) {
  const drawX = isWebcamMode ? (canvasEl.width - x - width) : x;
  
  // Draw glow box corners
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.shadowColor = color;
  ctx.shadowBlur = 8;
  
  const cornerSize = Math.min(18, width * 0.25, height * 0.25);
  
  ctx.beginPath();
  // Top-left corner
  ctx.moveTo(drawX + cornerSize, y); ctx.lineTo(drawX, y); ctx.lineTo(drawX, y + cornerSize);
  // Top-right corner
  ctx.moveTo(drawX + width - cornerSize, y); ctx.lineTo(drawX + width, y); ctx.lineTo(drawX + width, y + cornerSize);
  // Bottom-left corner
  ctx.moveTo(drawX + cornerSize, y + height); ctx.lineTo(drawX, y + height); ctx.lineTo(drawX, y + height - cornerSize);
  // Bottom-right corner
  ctx.moveTo(drawX + width - cornerSize, y + height); ctx.lineTo(drawX + width, y + height); ctx.lineTo(drawX + width, y + height - cornerSize);
  ctx.stroke();
  
  // Draw subtle dashed lines between corners
  ctx.shadowBlur = 0;
  ctx.strokeStyle = color + '2b'; // transparent hex
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.strokeRect(drawX, y, width, height);
  ctx.setLineDash([]); // Reset dash
  
  // Subtle glow container fill
  ctx.fillStyle = color + '08';
  ctx.fillRect(drawX, y, width, height);
  
  // Label Tag Background
  ctx.fillStyle = color;
  ctx.font = 'bold 11px Orbitron, sans-serif';
  const scoreText = score ? ` ${Math.round(score * 100)}%` : '';
  const textVal = `${label.toUpperCase()}${scoreText}`;
  const textWidth = ctx.measureText(textVal).width;
  
  // Custom tech polygon tag
  ctx.beginPath();
  ctx.moveTo(drawX, y);
  ctx.lineTo(drawX + textWidth + 16, y);
  ctx.lineTo(drawX + textWidth + 8, y - 18);
  ctx.lineTo(drawX, y - 18);
  ctx.closePath();
  ctx.fill();
  
  // Label Text
  ctx.fillStyle = '#000000';
  ctx.fillText(textVal, drawX + 6, y - 5);
}

// --- Main Processing Frame Loop ---
// --- Mode & Profile State Variables ---
let isWebcamMode = true;
let selectedFaceDetector = 'tiny'; // 'tiny' or 'ssd'
let enrolledProfiles = []; // Array of LabeledFaceDescriptors
let faceMatcher = null;
let currentDetectionFaces = []; // Holds current face results to register from

// Load registered face profiles
function loadEnrolledProfiles() {
  try {
    const data = localStorage.getItem('kvision_enrolled_profiles');
    if (data) {
      const parsed = JSON.parse(data);
      enrolledProfiles = parsed.map(profile => {
        const float32Descriptors = profile.descriptors.map(d => new Float32Array(d));
        return new faceapi.LabeledFaceDescriptors(profile.label, float32Descriptors);
      });
      updateFaceMatcher();
      renderEnrolledDB();
      logToTerminal(`Loaded ${enrolledProfiles.length} face profile(s) from database.`, "cyan");
    }
  } catch (err) {
    console.warn("Failed to load enrolled profiles:", err);
  }
}

// Save face profiles to localStorage
function saveEnrolledProfiles() {
  try {
    const serializable = enrolledProfiles.map(p => ({
      label: p.label,
      descriptors: p.descriptors.map(d => Array.from(d))
    }));
    localStorage.setItem('kvision_enrolled_profiles', JSON.stringify(serializable));
  } catch (err) {
    console.warn("Failed to save enrolled profiles:", err);
  }
}

// Update the faceapi FaceMatcher
function updateFaceMatcher() {
  if (enrolledProfiles.length > 0) {
    faceMatcher = new faceapi.FaceMatcher(enrolledProfiles, 0.6);
  } else {
    faceMatcher = null;
  }
}

// Update Face Enrollment UI database list
function renderEnrolledDB() {
  const dbList = document.getElementById('authorized-db-list');
  const emptyMsg = document.getElementById('empty-db-msg');
  
  dbList.innerHTML = '';
  
  if (enrolledProfiles.length === 0) {
    dbList.appendChild(emptyMsg);
    emptyMsg.style.display = 'block';
    return;
  }
  
  emptyMsg.style.display = 'none';
  
  enrolledProfiles.forEach(profile => {
    const item = document.createElement('div');
    item.className = 'db-item';
    item.innerHTML = `
      <div>
        <span class="db-item-name">${profile.label}</span>
        <span class="db-item-count">(${profile.descriptors.length} signature(s))</span>
      </div>
      <button class="db-item-delete" title="Delete Profile" data-name="${profile.label}">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor">
          <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
        </svg>
      </button>
    `;
    
    item.querySelector('.db-item-delete').addEventListener('click', (e) => {
      const label = e.currentTarget.getAttribute('data-name');
      enrolledProfiles = enrolledProfiles.filter(p => p.label !== label);
      saveEnrolledProfiles();
      updateFaceMatcher();
      renderEnrolledDB();
      logToTerminal(`DELETED BIOMETRIC PROFILE: ${label}`, "muted");
      playSynthBeep(300, 0.1, 'sawtooth');
    });
    
    dbList.appendChild(item);
  });
}

// Register current detected face with a custom name
function enrollCurrentFace() {
  const nameInput = document.getElementById('register-face-name');
  const name = nameInput.value.trim().toUpperCase();
  
  if (!name) {
    logToTerminal("ENROLLMENT ERROR: Please enter a target label name.", "object");
    playSynthBeep(200, 0.2, 'sawtooth');
    return;
  }
  
  if (currentDetectionFaces.length === 0) {
    logToTerminal("ENROLLMENT ERROR: No active face detected in frame.", "object");
    playSynthBeep(200, 0.2, 'sawtooth');
    return;
  }
  
  // Grab descriptor of the first detected face
  const faceToRegister = currentDetectionFaces[0];
  const descriptor = faceToRegister.descriptor;
  
  const existingProfile = enrolledProfiles.find(p => p.label === name);
  if (existingProfile) {
    existingProfile.descriptors.push(descriptor);
    logToTerminal(`BIOMETRIC UPDATED: Appended signature for ${name}.`, "cyan");
  } else {
    const newProfile = new faceapi.LabeledFaceDescriptors(name, [descriptor]);
    enrolledProfiles.push(newProfile);
    logToTerminal(`BIOMETRIC REGISTERED: Enrolled target "${name}".`, "cyan");
  }
  
  nameInput.value = '';
  saveEnrolledProfiles();
  updateFaceMatcher();
  renderEnrolledDB();
  playSynthBeep(880, 0.15, 'sine');
  
  // Hide enrollment widget after registration
  document.getElementById('face-registry-panel').classList.add('hidden');
}

// Show/Hide enrollment widget based on face detections
function updateRegistryWidget() {
  const registryPanel = document.getElementById('face-registry-panel');
  if (settings.filters.human && currentDetectionFaces.length > 0) {
    registryPanel.classList.remove('hidden');
  } else {
    registryPanel.classList.add('hidden');
  }
}

// Common function to draw face and object detections
function drawDetections(faceResults, cocoResults) {
  const activeFilters = settings.filters;
  
  let frameHumans = 0;
  let frameAnimals = 0;
  let frameObjects = 0;
  const currentFrameLabels = new Set();
  
  // 1. Draw COCO Detections
  cocoResults.forEach(det => {
    if (det.score < settings.confidenceThreshold) return;
    
    const { category, label, color } = parseDetectionClass(det.class);
    
    if (category === 'human') {
      if (!activeFilters.human) return;
      frameHumans++;
      currentFrameLabels.add('Human');
      
      const [bx, by, bw, bh] = det.bbox;
      drawBoundingBoxCorners(bx, by, bw, bh, color, 'Human', det.score);
    } else if (category === 'animal') {
      if (!activeFilters.animal) return;
      frameAnimals++;
      currentFrameLabels.add(label);
      
      const [bx, by, bw, bh] = det.bbox;
      drawBoundingBoxCorners(bx, by, bw, bh, color, label, det.score);
    } else if (category === 'object') {
      if (!activeFilters.object) return;
      frameObjects++;
      currentFrameLabels.add(label);
      
      const [bx, by, bw, bh] = det.bbox;
      drawBoundingBoxCorners(bx, by, bw, bh, color, label, det.score);
    }
  });
  
  // 2. Draw Face API Biometrics
  if (activeFilters.human && faceResults && faceResults.length > 0) {
    faceResults.forEach(face => {
      const { x, y, width, height } = face.detection.box;
      const gender = face.gender;
      const age = Math.round(face.age);
      
      // Recognition lookup
      let nameLabel = 'Person';
      
      if (faceMatcher && face.descriptor) {
        const match = faceMatcher.findBestMatch(face.descriptor);
        if (match && match.label !== 'unknown') {
          const matchPercent = Math.round((1 - match.distance) * 100);
          nameLabel = `${match.label} [${matchPercent}% MATCH]`;
        }
      }
      
      let expression = 'Neutral';
      let maxVal = 0;
      Object.entries(face.expressions).forEach(([exp, val]) => {
        if (val > maxVal) {
          maxVal = val;
          expression = exp.charAt(0).toUpperCase() + exp.slice(1);
        }
      });
      
      const detailsLabel = `${gender.toUpperCase()}, ~${age}y [${expression}]`;
      
      // Green color for humans
      const drawX = isWebcamMode ? (canvasEl.width - x - width) : x;
      
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(drawX, y, width, height);
      
      ctx.fillStyle = 'rgba(16, 185, 129, 0.04)';
      ctx.fillRect(drawX, y, width, height);
      
      // Details tag
      ctx.fillStyle = 'rgba(16, 185, 129, 0.85)';
      ctx.font = '9px Orbitron, sans-serif';
      ctx.fillRect(drawX, y + height, width, 16);
      
      ctx.fillStyle = '#000';
      ctx.fillText(detailsLabel, drawX + 4, y + height + 11);
      
      // Top identity tag
      ctx.fillStyle = '#10b981';
      ctx.font = 'bold 11px Orbitron, sans-serif';
      const textWidth = ctx.measureText(nameLabel).width;
      
      ctx.beginPath();
      ctx.moveTo(drawX, y);
      ctx.lineTo(drawX + textWidth + 16, y);
      ctx.lineTo(drawX + textWidth + 8, y - 18);
      ctx.lineTo(drawX, y - 18);
      ctx.closePath();
      ctx.fill();
      
      ctx.fillStyle = '#000';
      ctx.fillText(nameLabel, drawX + 6, y - 5);
      
      // Draw wireframe face mesh
      drawFaceMesh(face.landmarks);
      
      // Count face detections as humans if COCO person isn't detected
      if (cocoResults.filter(d => d.class === 'person').length === 0) {
        frameHumans++;
        currentFrameLabels.add('Human');
      }
    });
  }
  
  // Update Live Stats
  const activeScansTotal = frameHumans + frameAnimals + frameObjects;
  statActiveScans.textContent = activeScansTotal;
  statBreakdown.textContent = `H:${frameHumans} A:${frameAnimals} O:${frameObjects}`;
  
  countHuman.textContent = frameHumans;
  countAnimal.textContent = frameAnimals;
  countObject.textContent = frameObjects;
  
  if (isWebcamMode) {
    currentFrameLabels.forEach(label => {
      if (!lastDetectedLabels.has(label)) {
        totalScannedCount++;
        let logType = 'muted';
        if (label === 'Human') {
          logType = 'person';
          logToTerminal(`TARGET LOCKED: Human detected. Analyzing biometrics...`, logType);
          speakText("Human detected");
          playSynthBeep(880, 0.08, 'sawtooth');
        } else {
          const isAnimal = ANIMAL_CLASSES.includes(label.toLowerCase());
          logType = isAnimal ? 'animal' : 'object';
          logToTerminal(`SCANNER REGISTERED: ${label} loaded in field.`, logType);
          speakText(`${label} detected`);
          playSynthBeep(650, 0.1, 'sine');
        }
      }
    });
    lastDetectedLabels = currentFrameLabels;
  }
  
  // Find highest confidence target for slot display
  let highestConfTarget = null;
  
  cocoResults.forEach(det => {
    if (det.score < settings.confidenceThreshold) return;
    const { category, label } = parseDetectionClass(det.class);
    if (category === 'human' && !activeFilters.human) return;
    if (category === 'animal' && !activeFilters.animal) return;
    if (category === 'object' && !activeFilters.object) return;
    
    const targetLabel = category === 'human' ? 'Person' : label;
    if (highestConfTarget === null || det.score > highestConfTarget.score) {
      highestConfTarget = { label: targetLabel, category, score: det.score };
    }
  });
  
  if (activeFilters.human && faceResults && faceResults.length > 0) {
    faceResults.forEach(face => {
      let label = 'Person (Human)';
      let score = face.detection.score;
      if (faceMatcher && face.descriptor) {
        const match = faceMatcher.findBestMatch(face.descriptor);
        if (match && match.label !== 'unknown') {
          label = match.label;
          score = 1 - match.distance;
        }
      }
      if (highestConfTarget === null || score > highestConfTarget.score) {
        highestConfTarget = { label, category: 'human', score };
      }
    });
  }
  
  if (highestConfTarget) {
    const { label, category, score } = highestConfTarget;
    const scoreText = ` (${Math.round(score * 100)}%)`;
    const fullLabel = `${label.toUpperCase()}${scoreText}`;
    
    let color = '#00f2fe';
    if (category === 'human') color = '#10b981';
    else if (category === 'animal') color = '#f59e0b';
    else if (category === 'object') color = '#d946ef';
    
    activeTargetName.textContent = fullLabel;
    activeTargetName.style.color = color;
    activeTargetName.style.textShadow = `0 0 8px ${color}99`;
    activeTargetCard.style.borderBottomColor = color;
    
    diagnosticsTargetVal.textContent = fullLabel;
    diagnosticsTargetVal.style.color = color;
    diagnosticsTargetVal.style.textShadow = `0 0 6px ${color}66`;
    diagnosticsTargetCard.style.borderLeftColor = color;
  } else {
    const activeScansText = activeScansTotal > 0 ? "SCANNING FIELD..." : "NO TARGETS DETECTED";
    activeTargetName.textContent = activeScansText;
    activeTargetName.style.color = '#00f2fe';
    activeTargetName.style.textShadow = '0 0 8px rgba(0, 242, 254, 0.5)';
    activeTargetCard.style.borderBottomColor = '#00f2fe';
    
    diagnosticsTargetVal.textContent = "SCANNING SYSTEM...";
    diagnosticsTargetVal.style.color = '#00f2fe';
    diagnosticsTargetVal.style.textShadow = '0 0 6px rgba(0, 242, 254, 0.4)';
    diagnosticsTargetCard.style.borderLeftColor = '#00f2fe';
  }
}

// Static image scanning handler
async function runImageDetection() {
  if (!modelsLoaded || isWebcamMode) return;
  
  const imgEl = document.getElementById('uploaded-image-preview');
  if (!imgEl.src) return;
  
  isProcessing = true;
  logToTerminal("Initiating static image analysis...", "cyan");
  playSynthBeep(800, 0.1, 'sine');
  
  // Set canvas dimensions to match natural image size
  canvasEl.width = imgEl.naturalWidth;
  canvasEl.height = imgEl.naturalHeight;
  
  // Fit canvas CSS styles to the displayed dimensions
  const displayWidth = imgEl.clientWidth;
  const displayHeight = imgEl.clientHeight;
  canvasEl.style.width = `${displayWidth}px`;
  canvasEl.style.height = `${displayHeight}px`;
  
  document.getElementById('overlay-resolution').textContent = `${imgEl.naturalWidth}x${imgEl.naturalHeight}`;
  
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  
  try {
    const activeFilters = settings.filters;
    const promises = [];
    
    if (activeFilters.human) {
      let faceOptions;
      if (selectedFaceDetector === 'tiny') {
        faceOptions = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });
      } else {
        faceOptions = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });
      }
      promises.push(
        faceapi.detectAllFaces(imgEl, faceOptions)
          .withFaceLandmarks()
          .withFaceExpressions()
          .withAgeAndGender()
          .withFaceDescriptors()
      );
    } else {
      promises.push(Promise.resolve([]));
    }
    
    if (activeFilters.animal || activeFilters.object || activeFilters.human) {
      promises.push(cocoModel.detect(imgEl));
    } else {
      promises.push(Promise.resolve([]));
    }
    
    const startTime = performance.now();
    const [faceResults, cocoResults] = await Promise.all(promises);
    const latency = Math.round(performance.now() - startTime);
    document.getElementById('overlay-latency').textContent = `${latency} ms`;
    
    currentDetectionFaces = faceResults;
    
    drawDetections(faceResults, cocoResults);
    updateRegistryWidget();
    logToTerminal("Static image scan complete.", "cyan");
    
  } catch (err) {
    console.error("Static image detection error:", err);
    logToTerminal(`ANALYSIS ERROR: ${err.message}`, "object");
  }
  isProcessing = false;
}

// Real-time camera scan frame loop
async function detectionFrameLoop() {
  if (videoEl.paused || videoEl.ended || !modelsLoaded || !isWebcamMode) {
    requestAnimationFrame(detectionFrameLoop);
    return;
  }
  
  const now = performance.now();
  frameCount++;
  const frameTime = now - lastFrameTime;
  lastFrameTime = now;
  
  if (!isProcessing) {
    isProcessing = true;
    const startTime = performance.now();
    
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    
    try {
      const activeFilters = settings.filters;
      const promises = [];
      
      if (activeFilters.human) {
        let faceOptions;
        if (selectedFaceDetector === 'tiny') {
          faceOptions = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });
        } else {
          faceOptions = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });
        }
        promises.push(
          faceapi.detectAllFaces(videoEl, faceOptions)
            .withFaceLandmarks()
            .withFaceExpressions()
            .withAgeAndGender()
            .withFaceDescriptors()
        );
      } else {
        promises.push(Promise.resolve([]));
      }
      
      if (activeFilters.animal || activeFilters.object || activeFilters.human) {
        promises.push(cocoModel.detect(videoEl));
      } else {
        promises.push(Promise.resolve([]));
      }
      
      const [faceResults, cocoResults] = await Promise.all(promises);
      
      const latency = Math.round(performance.now() - startTime);
      document.getElementById('overlay-latency').textContent = `${latency} ms`;
      
      currentDetectionFaces = faceResults;
      
      drawDetections(faceResults, cocoResults);
      updateRegistryWidget();
      
    } catch (err) {
      console.error("Frame detection loop crash:", err);
    }
    
    isProcessing = false;
  }
  
  requestAnimationFrame(detectionFrameLoop);
}

// Capture current frame snapshot
function captureSnapshot() {
  const sourceEl = isWebcamMode ? videoEl : document.getElementById('uploaded-image-preview');
  if (!sourceEl || (isWebcamMode && (!activeStream || videoEl.paused))) {
    logToTerminal("CAPTURE FAILED: Source feed is inactive.", "object");
    return;
  }
  
  playSynthBeep(1200, 0.05, 'sine');
  setTimeout(() => playSynthBeep(400, 0.15, 'sawtooth'), 50);
  
  const shutterFlash = document.createElement('div');
  shutterFlash.className = 'shutter-flash flash-animate';
  document.querySelector('.hud-frame-container').appendChild(shutterFlash);
  setTimeout(() => shutterFlash.remove(), 400);
  
  const captureCanvas = document.createElement('canvas');
  captureCanvas.width = isWebcamMode ? videoEl.videoWidth : sourceEl.naturalWidth;
  captureCanvas.height = isWebcamMode ? videoEl.videoHeight : sourceEl.naturalHeight;
  const captureCtx = captureCanvas.getContext('2d');
  
  captureCtx.drawImage(sourceEl, 0, 0, captureCanvas.width, captureCanvas.height);
  captureCtx.drawImage(canvasEl, 0, 0, captureCanvas.width, captureCanvas.height);
  
  const imgUrl = captureCanvas.toDataURL('image/jpeg');
  const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
  const snapshotId = `snap_${Date.now()}`;
  
  const snapshot = {
    id: snapshotId,
    url: imgUrl,
    time: timestamp
  };
  
  snapshots.unshift(snapshot);
  renderSnapshots();
  logToTerminal(`SNAPSHOT REGISTERED: ID #${snapshotId} logged to gallery.`, "cyan");
}

function renderSnapshots() {
  if (snapshots.length === 0) {
    emptyGalleryMsg.style.display = 'block';
    const thumbs = snapshotGallery.querySelectorAll('.gallery-thumb-wrapper');
    thumbs.forEach(t => t.remove());
    return;
  }
  
  emptyGalleryMsg.style.display = 'none';
  
  const thumbs = snapshotGallery.querySelectorAll('.gallery-thumb-wrapper');
  thumbs.forEach(t => t.remove());
  
  snapshots.forEach(snap => {
    const wrapper = document.createElement('div');
    wrapper.className = 'gallery-thumb-wrapper';
    wrapper.id = snap.id;
    
    wrapper.innerHTML = `
      <img src="${snap.url}" class="gallery-thumb-img" alt="Capture thumbnail">
      <div class="gallery-thumb-overlay">
        <a href="${snap.url}" download="Kathirvel_AI_Snapshot_${snap.time.replace(/:/g, '-')}.jpg" class="gallery-thumb-btn" title="Download Image">
          <svg viewBox="0 0 24 24" width="12" height="12" fill="currentColor">
            <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM17 13l-5 5-5-5h3V9h4v4h3z"/>
          </svg>
        </a>
        <button class="gallery-thumb-btn delete" title="Delete Snapshot">
          <svg viewBox="0 0 24 24" width="12" height="12" fill="currentColor">
            <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
          </svg>
        </button>
      </div>
    `;
    
    wrapper.querySelector('.delete').addEventListener('click', () => {
      snapshots = snapshots.filter(s => s.id !== snap.id);
      renderSnapshots();
      logToTerminal(`SNAPSHOT DELETED: ID #${snap.id} deleted.`, "muted");
      playSynthBeep(300, 0.08, 'sawtooth');
    });
    
    snapshotGallery.appendChild(wrapper);
  });
}

// Bind Control Inputs
function bindControlInputs() {
  const tabWebcam = document.getElementById('tab-webcam');
  const tabImage = document.getElementById('tab-image');
  const videoSourceGroup = document.getElementById('video-source-group');
  const uploadDropzone = document.getElementById('image-upload-dropzone');
  const imagePreview = document.getElementById('uploaded-image-preview');
  
  tabWebcam.addEventListener('click', async () => {
    if (isWebcamMode) return;
    isWebcamMode = true;
    tabWebcam.classList.add('active');
    tabImage.classList.remove('active');
    
    videoSourceGroup.classList.remove('hidden');
    uploadDropzone.classList.add('hidden');
    imagePreview.classList.add('hidden');
    videoEl.classList.remove('hidden');
    
    canvasEl.style.width = '100%';
    canvasEl.style.height = '100%';
    
    await startWebcam();
    logToTerminal("LIVE WEBCAM MODE ACTIVE", "cyan");
  });
  
  tabImage.addEventListener('click', () => {
    if (!isWebcamMode) return;
    isWebcamMode = false;
    tabImage.classList.add('active');
    tabWebcam.classList.remove('active');
    
    videoSourceGroup.classList.add('hidden');
    videoEl.classList.add('hidden');
    imagePreview.classList.remove('hidden');
    
    if (!imagePreview.src) {
      uploadDropzone.classList.remove('hidden');
    }
    
    if (activeStream) {
      activeStream.getTracks().forEach(track => track.stop());
      activeStream = null;
    }
    
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    currentDetectionFaces = [];
    updateRegistryWidget();
    logToTerminal("STATIC IMAGE SCAN MODE ACTIVE", "cyan");
  });

  const detectorSelect = document.getElementById('detector-select');
  detectorSelect.addEventListener('change', async (e) => {
    selectedFaceDetector = e.target.value;
    logToTerminal(`Face detector model set to: ${selectedFaceDetector === 'tiny' ? 'TinyFaceDetector' : 'SsdMobilenetv1'}`, "cyan");
    if (!isWebcamMode) {
      await runImageDetection();
    }
  });

  const fileInput = document.getElementById('image-file-input');
  const browseBtn = document.getElementById('browse-files-btn');
  
  const handleImageFile = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      logToTerminal("FILE ERROR: Invalid format. Please load an image.", "object");
      return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      uploadDropzone.classList.add('hidden');
      
      imagePreview.onload = async () => {
        await runImageDetection();
      };
    };
    reader.readAsDataURL(file);
  };
  
  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });
  
  fileInput.addEventListener('change', (e) => {
    handleImageFile(e.target.files[0]);
  });
  
  uploadDropzone.addEventListener('click', () => {
    fileInput.click();
  });
  
  uploadDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadDropzone.style.borderColor = 'var(--accent-cyan)';
  });
  
  uploadDropzone.addEventListener('dragleave', () => {
    uploadDropzone.style.borderColor = 'rgba(0, 242, 254, 0.3)';
  });
  
  uploadDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadDropzone.style.borderColor = 'rgba(0, 242, 254, 0.3)';
    handleImageFile(e.dataTransfer.files[0]);
  });

  const registerBtn = document.getElementById('register-face-btn');
  registerBtn.addEventListener('click', () => {
    enrollCurrentFace();
  });
  
  const registerInput = document.getElementById('register-face-name');
  registerInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      enrollCurrentFace();
    }
  });
  
  const reRunImageFilters = () => {
    if (!isWebcamMode) runImageDetection();
  };
  
  cameraSelect.addEventListener('change', () => {
    if (isWebcamMode) startWebcam();
  });
  
  filterHuman.addEventListener('change', (e) => {
    settings.filters.human = e.target.checked;
    logToTerminal(`Neural Filter Human detection set to: ${settings.filters.human ? 'ENABLED' : 'DISABLED'}`, "muted");
    reRunImageFilters();
  });
  
  filterAnimal.addEventListener('change', (e) => {
    settings.filters.animal = e.target.checked;
    logToTerminal(`Neural Filter Animal detection set to: ${settings.filters.animal ? 'ENABLED' : 'DISABLED'}`, "muted");
    reRunImageFilters();
  });
  
  filterObject.addEventListener('change', (e) => {
    settings.filters.object = e.target.checked;
    logToTerminal(`Neural Filter Object detection set to: ${settings.filters.object ? 'ENABLED' : 'DISABLED'}`, "muted");
    reRunImageFilters();
  });
  
  confidenceThreshold.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    confidenceVal.textContent = `${val}%`;
    settings.confidenceThreshold = val / 100;
  });
  confidenceThreshold.addEventListener('change', (e) => {
    logToTerminal(`Neural scanning threshold calibrated to: ${e.target.value}%`, "cyan");
    reRunImageFilters();
  });
  
  toggleVoice.addEventListener('change', (e) => {
    settings.voiceEnabled = e.target.checked;
    voiceSettingsContainer.style.opacity = settings.voiceEnabled ? '1' : '0.4';
    voiceSettingsContainer.style.pointerEvents = settings.voiceEnabled ? 'all' : 'none';
    logToTerminal(`TTS Voice announcer system set to: ${settings.voiceEnabled ? 'ENABLED' : 'DISABLED'}`, "muted");
  });
  
  voiceSelect.addEventListener('change', (e) => {
    settings.selectedVoiceName = e.target.value;
    logToTerminal(`TTS Voice profile changed to: ${settings.selectedVoiceName || 'Default System'}`, "muted");
    speakText("Voice profile updated.");
  });
  
  voiceRate.addEventListener('input', (e) => {
    const val = parseInt(e.target.value) / 10;
    voiceRateVal.textContent = `${val.toFixed(1)}x`;
    settings.voiceRate = val;
  });
  
  clearLogsBtn.addEventListener('click', () => {
    terminalLog.innerHTML = '';
    logToTerminal("Event terminal sequence cleared.", "cyan");
    playSynthBeep(700, 0.05, 'sine');
  });
  
  captureBtn.addEventListener('click', () => {
    captureSnapshot();
  });
  
  retryCameraBtn.addEventListener('click', () => {
    startWebcam();
  });
}

// App Entry Point
async function main() {
  logToTerminal("Kathirvel AI Vision core initial sequence started.", "cyan");
  
  bindControlInputs();
  initVoices();
  
  // Load registered profiles
  loadEnrolledProfiles();
  
  await initializeAICore();
  await startWebcam();
  
  videoEl.addEventListener('play', () => {
    logToTerminal("Video renderer frame feed active. Initializing scan loop.", "cyan");
    requestAnimationFrame(detectionFrameLoop);
  });
  
  if (!videoEl.paused) {
    requestAnimationFrame(detectionFrameLoop);
  }
}

window.addEventListener('DOMContentLoaded', main);

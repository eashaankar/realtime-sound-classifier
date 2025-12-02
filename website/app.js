// ====== CONFIG ======
const API_URL   = "http://localhost:5001/predict"; // Must match server.py
const RECORD_MS = 5000;                            // 5 seconds recording

// Class names should match server.py order
const CLASS_NAMES = [
  "baby_cry",
  "boiling_water",
  "door_knock",
  "cat_meow",
  "background",
  "dog_bark",
];

// ====== UI ELEMENTS ======
const statusEl    = document.getElementById("status");
const detectedEl  = document.getElementById("detected");
const startBtn    = document.getElementById("startBtn");
const stopBtn     = document.getElementById("stopBtn");   // unused in this version
const levelBarEl  = document.getElementById("levelBar");
const dingAudio   = document.getElementById("ding");

// Initial UI
statusEl.textContent    = "Idle";
detectedEl.textContent  = "None";
levelBarEl.style.width  = "0%";
stopBtn.disabled        = true; // we don't use Stop in this version

// ====== AUDIO LEVEL METER STATE ======
let audioContext = null;
let analyserNode = null;
let levelRafId   = null;

// Start the level meter using an AnalyserNode on the mic stream
function startLevelMeter(stream) {
  try {
    // Some browsers reuse contexts poorly; always create a fresh one for safety
    audioContext = new (window.AudioContext || window.webkitAudioContext)();

    const source = audioContext.createMediaStreamSource(stream);
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 256; // small FFT for fast updates

    source.connect(analyserNode);

    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function updateLevel() {
      if (!analyserNode) return;

      analyserNode.getByteTimeDomainData(dataArray);

      // Compute a simple RMS over the time-domain samples
      let sumSquares = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = (dataArray[i] - 128) / 128.0; // center to [-1, 1]
        sumSquares += v * v;
      }
      const rms = Math.sqrt(sumSquares / bufferLength); // 0..1 (ish)

      // Map RMS to a % bar width. You can tweak the multiplier for sensitivity.
      const level = Math.min(rms * 3.0, 1.0); // boost a bit, clamp at 1.0
      levelBarEl.style.width = (level * 100).toFixed(1) + "%";

      levelRafId = requestAnimationFrame(updateLevel);
    }

    updateLevel();
  } catch (e) {
    console.warn("[WARN] Could not start level meter:", e);
  }
}

// Stop the meter when we are done recording
function stopLevelMeter() {
  if (levelRafId) {
    cancelAnimationFrame(levelRafId);
    levelRafId = null;
  }
  levelBarEl.style.width = "0%";

  if (audioContext) {
    try {
      audioContext.close();
    } catch (e) {
      // ignore close errors
    }
    audioContext = null;
    analyserNode = null;
  }
}

// ====== UI HELPERS ======

// Highlight the detected class if there are DOM elements with data-class="class_name"
function highlightDetectedClass(predicted) {
  CLASS_NAMES.forEach((label) => {
    const el = document.querySelector(`[data-class="${label}"]`);
    if (!el) return;

    if (label === predicted) {
      el.classList.add("active");
    } else {
      el.classList.remove("active");
    }
  });
}

// Update visible text + ding using server response
function updateUIWithPrediction(data) {
  const predicted = data.predicted_class || "Unknown";

  // Optional: show probability for the predicted class if probabilities_dict exists
  let probPercentText = "";
  if (data.probabilities_dict && data.probabilities_dict[predicted] != null) {
    const p = data.probabilities_dict[predicted];
    probPercentText = ` (${(p * 100).toFixed(1)}%)`;
  }

  statusEl.textContent   = `Detected: ${predicted}${probPercentText}`;
  detectedEl.textContent = predicted;

  // Play ding for any class except "background"
  if (predicted !== "background" && dingAudio) {
    try {
      dingAudio.currentTime = 0;
      dingAudio.play();
    } catch (e) {
      console.warn("[WARN] Could not play ding:", e);
    }
  }

  // Highlight detected label in list (if elements exist)
  highlightDetectedClass(predicted);
}

// ====== CORE: one click → one recording → one prediction ======
async function startOneShotPrediction() {
  if (startBtn.disabled) return; // guard double click

  startBtn.disabled = true;
  statusEl.textContent   = "Requesting microphone…";
  detectedEl.textContent = "None";

  let stream = null;
  let mediaRecorder = null;
  let chunks = [];

  try {
    // 1) Get mic
    stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
      video: false,
    });
    console.log("[DEBUG] Mic stream acquired");

    // Start input level meter using same stream
    startLevelMeter(stream);

    // 2) Set up MediaRecorder
    mediaRecorder = new MediaRecorder(stream);
    console.log("[DEBUG] MediaRecorder created, mimeType:", mediaRecorder.mimeType);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        chunks.push(e.data);
      }
    };

    mediaRecorder.onerror = (event) => {
      console.error("[MEDIARECORDER ERROR]:", event.error);
      statusEl.textContent = "Recording error (see console)";
    };

    mediaRecorder.onstop = async () => {
      console.log("[DEBUG] Recording stopped. Captured chunks:", chunks.length);

      // Stop the level meter when recording stops
      stopLevelMeter();

      // 3) Build blob from chunks
      const blob = new Blob(chunks, { type: "audio/webm" });
      console.log("[DEBUG] Blob size:", blob.size);

      // 4) Send to backend
      const formData = new FormData();
      formData.append("audio_file", blob, "chunk.webm");

      try {
        statusEl.textContent = "Sending to backend…";
        console.log("[DEBUG] Sending POST to", API_URL);

        const res = await fetch(API_URL, {
          method: "POST",
          body: formData,
        });

        const rawText = await res.text();
        console.log("[DEBUG] HTTP status:", res.status);
        console.log("[DEBUG] Raw response text:", rawText);

        if (!res.ok) {
          console.error("[BACKEND ERROR]:", rawText);
          statusEl.textContent = "Backend error (see console)";
          return;
        }

        let data;
        try {
          data = JSON.parse(rawText);
        } catch (e) {
          console.error("[JSON PARSE ERROR]:", e);
          statusEl.textContent = "Bad JSON from server";
          return;
        }

        console.log("=== FINAL PREDICTION ===");
        console.log("Class:", data.predicted_class);
        console.log("Probabilities (array):", data.probabilities);
        console.log("Probabilities (dict):", data.probabilities_dict);
        console.log("========================");

        // 5) Show result in UI (persist until next Start or reload)
        updateUIWithPrediction(data);

      } catch (err) {
        console.error("[FETCH ERROR]:", err);
        statusEl.textContent = "Network error (see console)";
      } finally {
        // 6) Clean up mic
        if (stream) {
          stream.getTracks().forEach((t) => t.stop());
        }
        startBtn.disabled = false; // allow another one-shot run
      }
    };

    // 3) Start recording
    statusEl.textContent = "Recording… (one-shot)";
    mediaRecorder.start();
    console.log("[DEBUG] Recording started for", RECORD_MS, "ms");

    // 4) Stop after fixed time
    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        console.log("[DEBUG] Auto-stopping recorder after RECORD_MS");
        mediaRecorder.stop();
      }
    }, RECORD_MS);

  } catch (err) {
    console.error("[ERROR] startOneShotPrediction:", err);
    statusEl.textContent = "Mic error (see console)";
    startBtn.disabled = false;

    // Make sure we also stop the meter on error
    stopLevelMeter();

    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }
  }
}

// ====== HOOK BUTTON ======
startBtn.addEventListener("click", startOneShotPrediction);

/**
 * app.js — SoilSense AI Frontend Logic
 * =====================================
 * Handles:
 *  - Drag-and-drop + click file selection
 *  - Client-side validation (format, size)
 *  - Loading animation with stage messaging
 *  - fetch() POST to /predict API
 *  - Dynamic result rendering (soil type, confidence bar, plant cards, nutrients)
 *  - Error display
 *  - Print / Save as PDF
 */

"use strict";

// ──────────────────────────────────────────────
// CONFIG
// ──────────────────────────────────────────────
const API_BASE        = "http://localhost:5000";
const PREDICT_URL     = `${API_BASE}/predict`;
const MAX_SIZE_BYTES  = 16 * 1024 * 1024;   // 16 MB
const ALLOWED_TYPES   = ["image/jpeg", "image/png"];

// Soil type → emoji mapping
const SOIL_EMOJIS = {
  Sandy: "🏜️",
  Clay:  "🏔️",
  Loamy: "🌿",
  Silt:  "💧",
};

// Crop → emoji mapping (best-effort)
const CROP_EMOJIS = {
  Tomato:     "🍅", Corn:     "🌽", Spinach:  "🥬", Soybean:   "🫘",
  Sunflower:  "🌻", Rice:     "🌾", Wheat:    "🌾", Sugarcane: "🎋",
  Broccoli:   "🥦", Cabbage:  "🥬", Groundnut:"🥜", Watermelon:"🍉",
  Carrot:     "🥕", Potato:   "🥔", Barley:   "🌾", Jute:      "🌿",
  Maize:      "🌽", Cucumber: "🥒", Pepper:   "🌶️", Mustard:   "🟡",
};

// Loading stage messages (cycled during API call)
const LOADER_STAGES = [
  "Preprocessing image…",
  "Running CNN model…",
  "Classifying soil type…",
  "Fetching crop database…",
  "Generating recommendations…",
];


// ──────────────────────────────────────────────
// DOM REFERENCES
// ──────────────────────────────────────────────
const dropZone        = document.getElementById("dropZone");
const fileInput       = document.getElementById("fileInput");
const browseBtn       = document.getElementById("browseBtn");
const dropContent     = document.getElementById("dropContent");
const previewContent  = document.getElementById("previewContent");
const previewImg      = document.getElementById("previewImg");
const previewName     = document.getElementById("previewName");
const previewSize     = document.getElementById("previewSize");
const analyzeBtn      = document.getElementById("analyzeBtn");
const clearBtn        = document.getElementById("clearBtn");
const errorToast      = document.getElementById("errorToast");
const errorMsg        = document.getElementById("errorMsg");
const toastClose      = document.getElementById("toastClose");
const loadingOverlay  = document.getElementById("loadingOverlay");
const loaderSub       = document.getElementById("loaderSub");
const loaderProgress  = document.getElementById("loaderProgress");
const resultsSection  = document.getElementById("resultsSection");
const demoBanner      = document.getElementById("demoBanner");
const soilEmoji       = document.getElementById("soilEmoji");
const soilName        = document.getElementById("soilName");
const soilDesc        = document.getElementById("soilDesc");
const processingTime  = document.getElementById("processingTime");
const confidenceNum   = document.getElementById("confidenceNum");
const confidenceBar   = document.getElementById("confidenceBar");
const confidenceVerdict = document.getElementById("confidenceVerdict");
const plantsGrid      = document.getElementById("plantsGrid");
const nutrientsGrid   = document.getElementById("nutrientsGrid");
const cropsTableBody  = document.getElementById("cropsTableBody");
const analyzeAnotherBtn = document.getElementById("analyzeAnotherBtn");
const printBtn        = document.getElementById("printBtn");

// Current selected file
let selectedFile = null;
let loaderInterval = null;
let loaderProgressVal = 0;


// ──────────────────────────────────────────────
// FILE SELECTION HELPERS
// ──────────────────────────────────────────────

/** Format bytes to human-readable string */
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(2)} MB`;
}

/** Client-side validation of the File object */
function validateFile(file) {
  if (!ALLOWED_TYPES.includes(file.type)) {
    return `Unsupported format "${file.type || 'unknown'}". Please upload a JPEG or PNG image.`;
  }
  if (file.size > MAX_SIZE_BYTES) {
    return `File too large (${formatBytes(file.size)}). Maximum allowed size is 16 MB.`;
  }
  return null;   // valid
}

/** Show the preview area with the selected image */
function showPreview(file) {
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    previewSize.textContent = formatBytes(file.size);
    dropContent.classList.add("hidden");
    previewContent.classList.remove("hidden");
    hideError();
  };
  reader.readAsDataURL(file);
}

/** Reset drop zone to initial state */
function clearSelection() {
  selectedFile   = null;
  fileInput.value = "";
  previewImg.src = "";
  previewContent.classList.add("hidden");
  dropContent.classList.remove("hidden");
  hideError();
}

/** Show error toast */
function showError(message) {
  errorMsg.textContent = message;
  errorToast.classList.remove("hidden");
  errorToast.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/** Hide error toast */
function hideError() {
  errorToast.classList.add("hidden");
  errorMsg.textContent = "";
}


// ──────────────────────────────────────────────
// EVENT LISTENERS – FILE SELECTION
// ──────────────────────────────────────────────

// Click to open browser
browseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

// Clicking the drop zone also opens browser (but not the preview area)
dropZone.addEventListener("click", () => {
  if (selectedFile) return;   // already has preview
  fileInput.click();
});

// Keyboard accessibility for drop zone
dropZone.addEventListener("keydown", (e) => {
  if ((e.key === "Enter" || e.key === " ") && !selectedFile) {
    e.preventDefault();
    fileInput.click();
  }
});

// File input change
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  const err = validateFile(file);
  if (err) { showError(err); return; }
  showPreview(file);
});

// Drag & drop
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (!file) return;
  const err = validateFile(file);
  if (err) { showError(err); return; }
  showPreview(file);
});

// Clear selection
clearBtn.addEventListener("click", clearSelection);

// Dismiss error toast
toastClose.addEventListener("click", hideError);

// Analyze Another button
analyzeAnotherBtn.addEventListener("click", () => {
  clearSelection();
  resultsSection.classList.add("hidden");
  document.getElementById("upload-section").scrollIntoView({ behavior: "smooth" });
});

// Print / PDF
printBtn.addEventListener("click", () => window.print());


// ──────────────────────────────────────────────
// LOADING ANIMATION
// ──────────────────────────────────────────────

function startLoader() {
  loadingOverlay.classList.remove("hidden");
  loaderProgressVal = 0;
  loaderProgress.style.width = "0%";

  let stageIndex = 0;
  loaderSub.textContent = LOADER_STAGES[0];

  loaderInterval = setInterval(() => {
    // Advance progress bar
    loaderProgressVal = Math.min(loaderProgressVal + Math.random() * 8 + 2, 90);
    loaderProgress.style.width = `${loaderProgressVal}%`;

    // Cycle through stage messages
    stageIndex = (stageIndex + 1) % LOADER_STAGES.length;
    loaderSub.textContent = LOADER_STAGES[stageIndex];
  }, 600);
}

function stopLoader() {
  clearInterval(loaderInterval);
  loaderProgress.style.width = "100%";
  setTimeout(() => {
    loadingOverlay.classList.add("hidden");
    loaderProgress.style.width = "0%";
    loaderProgressVal = 0;
  }, 350);
}


// ──────────────────────────────────────────────
// API CALL
// ──────────────────────────────────────────────

async function analyzeImage() {
  if (!selectedFile) {
    showError("Please select an image first.");
    return;
  }

  hideError();
  startLoader();

  const formData = new FormData();
  formData.append("image", selectedFile);

  let data;

  try {
    const response = await fetch(PREDICT_URL, {
      method: "POST",
      body: formData,
      signal: AbortSignal.timeout(30000),   // 30s timeout
    });

    // Try to parse JSON regardless of HTTP status code (error messages)
    let json;
    try {
      json = await response.json();
    } catch {
      throw new Error(`Server returned an unreadable response (HTTP ${response.status}).`);
    }

    if (!response.ok || json.success === false) {
      throw new Error(json.error || `Server error (${response.status})`);
    }

    data = json;

  } catch (err) {
    stopLoader();

    if (err.name === "TimeoutError") {
      showError("Request timed out. Make sure the backend is running on http://localhost:5000.");
    } else if (err.name === "TypeError" && err.message.includes("fetch")) {
      showError("Cannot connect to backend (http://localhost:5000). Please run: python app.py");
    } else {
      showError(err.message || "An unexpected error occurred.");
    }
    return;
  }

  stopLoader();
  renderResults(data);
}

// Trigger analyze on button click
analyzeBtn.addEventListener("click", analyzeImage);


// ──────────────────────────────────────────────
// RESULT RENDERING
// ──────────────────────────────────────────────

function renderResults(data) {
  // ── Demo banner ──
  if (data.demo_mode) {
    demoBanner.classList.remove("hidden");
  } else {
    demoBanner.classList.add("hidden");
  }

  // ── Soil type ──
  const type = data.soil_type || "Unknown";
  soilEmoji.textContent = SOIL_EMOJIS[type] || "🌍";
  soilName.textContent  = type;
  soilDesc.textContent  = data.soil_description || "";
  if (data.processing_time_ms) {
    processingTime.textContent = `Analysis completed in ${data.processing_time_ms} ms`;
  }

  // ── Confidence ──
  const conf    = parseFloat(data.confidence) || 0;
  const confPct = Math.round(conf * 100);
  confidenceNum.textContent = `${confPct}%`;
  // Delay bar fill for animation after element becomes visible
  requestAnimationFrame(() => {
    setTimeout(() => { confidenceBar.style.width = `${confPct}%`; }, 100);
  });

  // Verdict text
  let verdict = "";
  if (confPct >= 85)      verdict = "✅ High confidence – reliable prediction";
  else if (confPct >= 70) verdict = "🟡 Moderate confidence – consider field testing";
  else                    verdict = "⚠️ Low confidence – result may vary";
  confidenceVerdict.textContent = verdict;

  // ── Recommended plants (top 3) ──
  plantsGrid.innerHTML = "";
  const plants = data.recommended_plants || [];
  plants.slice(0, 3).forEach((name, idx) => {
    const card = document.createElement("div");
    card.className = "plant-card";
    card.style.animationDelay = `${idx * 0.12}s`;
    card.innerHTML = `
      <div class="plant-icon">${CROP_EMOJIS[name] || "🌱"}</div>
      <div class="plant-name">${name}</div>
      <div class="plant-rank">#${idx + 1} Recommendation</div>
    `;
    plantsGrid.appendChild(card);
  });

  // ── Nutrients ──
  nutrientsGrid.innerHTML = "";
  const nutrients = data.soil_nutrients || {};
  const nutrientDefs = [
    { key: "nitrogen",   label: "Nitrogen (N)",   icon: "🌱" },
    { key: "phosphorus", label: "Phosphorus (P)", icon: "⚡" },
    { key: "potassium",  label: "Potassium (K)",  icon: "💧" },
  ];
  nutrientDefs.forEach(({ key, label, icon }) => {
    const level = nutrients[key] || "—";
    const div = document.createElement("div");
    div.className = "nutrient-item";
    div.innerHTML = `
      <div class="nutrient-icon">${icon}</div>
      <div>
        <div class="nutrient-label">${label}</div>
        <span class="nutrient-level level-${level}">${level}</span>
      </div>
    `;
    nutrientsGrid.appendChild(div);
  });

  // ── All crops table ──
  cropsTableBody.innerHTML = "";
  const allCrops = data.all_crops || [];
  const displayCrops = allCrops.length > 0 ? allCrops : plants.map(n => ({ name: n, description: "" }));
  displayCrops.forEach((crop, idx) => {
    const tr = document.createElement("tr");
    const rankClass = idx === 0 ? "rank-badge top1" : idx === 1 ? "rank-badge top2" : idx === 2 ? "rank-badge top3" : "rank-badge";
    tr.innerHTML = `
      <td><span class="${rankClass}">${idx + 1}</span></td>
      <td class="crop-name-cell">${CROP_EMOJIS[crop.name] || "🌱"} ${crop.name}</td>
      <td>${crop.description || "Suitable for this soil type."}</td>
    `;
    cropsTableBody.appendChild(tr);
  });

  // ── Show results section ──
  resultsSection.classList.remove("hidden");
  // Smooth scroll to results
  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 150);
}


// ──────────────────────────────────────────────
// SMOOTH SCROLL for nav links
// ──────────────────────────────────────────────
document.querySelectorAll(".nav-link[href^='#']").forEach(link => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const target = document.querySelector(link.getAttribute("href"));
    if (target) target.scrollIntoView({ behavior: "smooth" });

    // Update active state
    document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
    link.classList.add("active");
  });
});

// ──────────────────────────────────────────────
// INTERSECTION OBSERVER – nav active state
// ──────────────────────────────────────────────
const sections = document.querySelectorAll("section[id]");
const navLinks  = document.querySelectorAll(".nav-link[href^='#']");
const observer  = new IntersectionObserver(
  (entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(link => {
          link.classList.toggle(
            "active",
            link.getAttribute("href") === `#${entry.target.id}`
          );
        });
      }
    });
  },
  { threshold: 0.4, rootMargin: "-80px 0px 0px 0px" }
);
sections.forEach(s => observer.observe(s));

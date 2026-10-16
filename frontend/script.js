const API_URL = "http://127.0.0.1:8000/api/process";

const videoList = document.getElementById("videoList");
const addVideoBtn = document.getElementById("addVideoBtn");
const generateBtn = document.getElementById("generateBtn");
const statusBox = document.getElementById("statusBox");
const outputSection = document.getElementById("outputSection");
const statsBar = document.getElementById("statsBar");
const notesContainer = document.getElementById("notesContainer");
const flashcardsContainer = document.getElementById("flashcardsContainer");

// Add another video input dynamically
addVideoBtn.addEventListener("click", () => {
  const count = videoList.querySelectorAll(".video-item").length + 1;
  const wrapper = document.createElement("div");
  wrapper.className = "video-item";
  wrapper.innerHTML = `
    <div class="video-label-row">
      <label>Lecture Video or Transcript #${count}</label>
      <button type="button" class="remove-video-btn" onclick="removeVideoItem(this)">Remove</button>
    </div>
    <textarea
      class="video-input"
      rows="3"
      placeholder="Paste a YouTube link (e.g., https://youtu.be/...) or paste lecture notes for Video #${count}"
    ></textarea>
  `;
  videoList.appendChild(wrapper);
});

// Remove extra video input boxes and re-number remaining ones
function removeVideoItem(buttonEl) {
  const item = buttonEl.closest(".video-item");
  if (item) {
    item.remove();
    const items = videoList.querySelectorAll(".video-item");
    items.forEach((el, idx) => {
      const label = el.querySelector("label");
      if (label) {
        label.textContent = `Lecture Video or Transcript #${idx + 1}`;
      }
    });
  }
}

// Switch between Notes and Flashcards tabs
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document
      .querySelectorAll(".tab-btn")
      .forEach((b) => b.classList.remove("active"));
    document
      .querySelectorAll(".tab-content")
      .forEach((c) => c.classList.remove("active"));

    btn.classList.add("active");
    const targetId = btn.getAttribute("data-tab");
    document.getElementById(targetId).classList.add("active");
  });
});

// Process videos via FastAPI Backend
generateBtn.addEventListener("click", async () => {
  const topic =
    document.getElementById("topicInput").value.trim() || "Study Guide";
  const inputs = Array.from(document.querySelectorAll(".video-input"))
    .map((el) => el.value.trim())
    .filter((val) => val.length > 0);

  if (inputs.length === 0) {
    showStatus(
      "Please paste at least one YouTube video link or lecture transcript to begin.",
      true,
    );
    return;
  }

  showStatus(
    "Analyzing your lectures and creating your study notes & flashcards...",
    false,
  );
  generateBtn.disabled = true;

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        topic: topic,
        video_inputs: inputs,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Failed to process lectures.");
    }

    renderResults(data);
    statusBox.classList.add("hidden");
  } catch (err) {
    showStatus(`Error: ${err.message}`, true);
  } finally {
    generateBtn.disabled = false;
  }
});

function showStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.style.border = isError
    ? "1.5px solid #ff5500"
    : "1.5px solid #047857";
  statusBox.style.color = isError ? "#c23b00" : "#064e3b";
  statusBox.style.background = isError ? "#fff3ed" : "#ecfdf5";
  statusBox.classList.remove("hidden");
}

function renderResults(data) {
  outputSection.classList.remove("hidden");

  statsBar.innerHTML = `
    <div class="stat-item">
      <span class="stat-label">Study Topic</span>
      <span class="stat-value">${escapeHtml(data.topic)}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Lectures Combined</span>
      <span class="stat-value">${data.video_count}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Key Points Analyzed</span>
      <span class="stat-value">${data.total_sentences_analyzed}</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Flashcards Ready</span>
      <span class="stat-value">${data.flashcards.length}</span>
    </div>
  `;

  // Render Unified Revision Notes
  notesContainer.innerHTML = data.notes
    .map(
      (mod) => `
      <div class="note-module">
        <div class="note-header">
          <h3>${escapeHtml(mod.module_title)}</h3>
          <span class="source-tag">From: ${mod.sources.join(", ")}</span>
        </div>
        <div class="summary-box">
          <strong>Quick Summary:</strong> ${escapeHtml(mod.summary)}
        </div>
        <h4>Key Takeaways to Remember</h4>
        <ul class="key-points-list">
          ${mod.key_points.map((pt) => `<li>${escapeHtml(pt)}</li>`).join("")}
        </ul>
      </div>
    `,
    )
    .join("");

  // Render 3D Interactive Flashcards
  flashcardsContainer.innerHTML = data.flashcards
    .map(
      (card) => `
      <div class="flashcard" onclick="this.classList.toggle('flipped')">
        <div class="flashcard-inner">
          <div class="flashcard-front">
            <div class="card-meta">
              <span>Flashcard #${card.id} &bull; ${escapeHtml(card.difficulty)}</span>
              <span>${escapeHtml(card.source)}</span>
            </div>
            <div class="card-body">
              ${escapeHtml(card.question)}
            </div>
            <div class="card-footer">Click to flip for answer ↻</div>
          </div>
          <div class="flashcard-back">
            <div class="card-meta">
              <span>Answer</span>
              <span>${escapeHtml(card.source)}</span>
            </div>
            <div class="card-body">
              ${escapeHtml(card.answer)}
            </div>
            <div class="card-footer">Click to flip back ↻</div>
          </div>
        </div>
      </div>
    `,
    )
    .join("");

  outputSection.scrollIntoView({ behavior: "smooth" });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

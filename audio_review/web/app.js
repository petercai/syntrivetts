const state = {
  view: "landing",
  activeJobId: null,
  activeChapterId: null,
  chapters: [],
  playlist: null,
  selectedIndex: null,
  continuousPlay: true,
  wavesurfer: null,
  referenceAudioEl: null,
};

const ISSUE_CATEGORIES = ["silence", "noise", "audio_quality", "voice_mismatch", "other"];

async function getJSON(url) {
  const resp = await fetch(url);
  const data = await resp.json();
  if (!resp.ok) throw new Error(describeError(data));
  return data;
}

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(describeError(data));
  return data;
}

function describeError(data) {
  const detail = data && data.detail;
  if (detail && typeof detail === "object" && detail.url) {
    return `${detail.message} -- open ${detail.url} instead.`;
  }
  return (detail && (detail.message || detail)) || "Request failed";
}

function showError(message) {
  const el = document.getElementById("error-banner");
  el.textContent = message;
  el.style.display = "block";
}

function clearError() {
  document.getElementById("error-banner").style.display = "none";
}

function setStatus(message) {
  document.getElementById("status-banner").textContent = message || "";
}

const APP_TITLE = "SyntriveTTS Audio Review";

function setHeaderTitle(bookTitle) {
  document.getElementById("app-title").textContent = bookTitle || APP_TITLE;
  document.title = bookTitle ? `${bookTitle} - ${APP_TITLE}` : APP_TITLE;
}

function showView(name) {
  state.view = name;
  if (name === "landing") setHeaderTitle(null);
  document.getElementById("landing-view").classList.toggle("hidden", name !== "landing");
  document.getElementById("review-view").classList.toggle("hidden", name !== "review");
  document.getElementById("back-to-landing").style.display = name === "review" ? "inline-block" : "none";
}

async function loadJobs() {
  clearError();
  setStatus("Loading...");
  try {
    const data = await getJSON("/api/jobs");
    document.getElementById("repo-path").textContent = data.repo_dir;
    renderJobs(data.jobs);
    setStatus(`${data.jobs.length} job(s) in this repo.`);
  } catch (err) {
    setStatus("");
    showError("Failed to load jobs: " + err.message);
  }
}

function renderJobs(jobs) {
  const rows = document.getElementById("job-rows");
  rows.innerHTML = "";
  for (const job of jobs) {
    const tr = document.createElement("tr");
    tr.className = "clickable";
    tr.innerHTML =
      `<td>${escapeHtml(job.book_title)}</td>` +
      `<td>${escapeHtml(job.process_dir)}</td>` +
      `<td>${escapeHtml(job.status)}</td>` +
      `<td>${escapeHtml(job.created_at || "")}</td>`;
    tr.addEventListener("click", () => openJob(job));
    rows.appendChild(tr);
  }
}

async function switchRepo(path) {
  clearError();
  setStatus("Switching repo...");
  try {
    const data = await postJSON("/api/repo/switch", { repo_dir: path });
    document.getElementById("repo-path").textContent = data.repo_dir;
    renderJobs(data.jobs);
    setStatus(`${data.jobs.length} job(s) in this repo.`);
  } catch (err) {
    setStatus("");
    showError(String(err.message || err));
  }
}

async function openJob(job) {
  const jobId = job.job_id;
  clearError();
  setStatus("Loading chapters...");
  state.activeJobId = jobId;
  state.activeChapterId = null;
  state.chapters = [];
  showView("review");
  setHeaderTitle(job.book_title);
  renderChapterTree(state.chapters);
  clearChapterView();
  try {
    const data = await getJSON(`/api/jobs/${jobId}/chapters`);
    state.chapters = data.chapters;
    renderChapterTree(state.chapters);
    if (state.chapters.length > 0) {
      await selectChapter(state.chapters[0].transcript_chapter_id);
    } else {
      setStatus("No sentence audio yet for this job (nothing under sentence_audio/).");
    }
  } catch (err) {
    setStatus("");
    showError("Failed to load chapters: " + err.message);
  }
}

function renderChapterTree(chapters) {
  const el = document.getElementById("chapter-tree");
  el.innerHTML = "";
  for (const ch of chapters) {
    const row = document.createElement("div");
    row.className = "chapter-row" + (ch.transcript_chapter_id === state.activeChapterId ? " active" : "");
    row.textContent = ch.chapter_name ? `${ch.chapter_basename} (${ch.chapter_name})` : ch.chapter_basename;
    row.title = `${ch.chapter_id} (sequence ${ch.sequence_number || "?"}, ${ch.synthesis_status})`;
    row.addEventListener("click", () => selectChapter(ch.transcript_chapter_id));
    el.appendChild(row);
  }
}

async function selectChapter(chapterId) {
  clearError();
  state.activeChapterId = chapterId;
  renderChapterTree(state.chapters);
  clearChapterView();
  setStatus("Loading chapter...");
  try {
    await postJSON(`/api/jobs/${state.activeJobId}/chapters/${chapterId}/rescan`);
    const playlist = await getJSON(`/api/jobs/${state.activeJobId}/chapters/${chapterId}`);
    state.playlist = playlist;
    renderChapterStatus();
    renderSentenceList();
    const firstPlayable = playlist.sentences.findIndex((s) => s.status === "ok");
    selectSentence(firstPlayable >= 0 ? firstPlayable : 0);
  } catch (err) {
    setStatus("");
    showError("Failed to load chapter: " + err.message);
  }
}

function renderChapterStatus(note) {
  const p = state.playlist;
  if (!p) return;
  const total = p.sentences.length;
  const generated = p.sentences.filter((s) => s.exists).length;
  const pendingRegen = p.sentences.filter((s) => s.status === "pending_regen").length;
  let text = `${p.chapter_basename}: ${generated} of ${total} sentence FLAC file(s) generated`;
  if (pendingRegen > 0) text += `, ${pendingRegen} deleted and awaiting regeneration`;
  setStatus(`${text}.${note ? " " + note : ""}`);
}

function renderSentenceList() {
  const el = document.getElementById("sentence-list");
  el.innerHTML = "";
  if (!state.playlist) return;
  state.playlist.sentences.forEach((s, i) => {
    const row = document.createElement("div");
    row.className = "sentence-row" + (i === state.selectedIndex ? " selected" : "");
    row.dataset.status = s.status;
    row.innerHTML =
      `<span class="status-dot"></span>` +
      `<span class="file-name">${escapeHtml(s.file_name)}</span>` +
      `<span class="preview">${escapeHtml(s.text)}</span>`;
    row.title = `${s.status}: ${s.text}`;
    row.addEventListener("click", () => selectSentence(i));
    el.appendChild(row);
  });
}

function clearChapterView() {
  state.wavesurfer?.stop?.();
  state.wavesurfer?.empty?.();
  state.referenceAudioEl?.pause();
  state.playlist = null;
  state.selectedIndex = null;
  renderSentenceList();
  renderRightPane();
  updateTransportTime();
}

function setTransportEnabled(enabled) {
  document.getElementById("restart-btn").disabled = !enabled;
  document.getElementById("play-pause-btn").disabled = !enabled;
}

function renderRightPane() {
  const sentence = state.playlist && state.selectedIndex != null ? state.playlist.sentences[state.selectedIndex] : null;
  const textEl = document.getElementById("sentence-text");
  const metaEl = document.getElementById("sentence-meta");
  const deleteActions = document.getElementById("delete-actions");
  const undoActions = document.getElementById("undo-actions");
  const refRow = document.getElementById("reference-audio-row");

  if (!sentence) {
    textEl.textContent = "";
    metaEl.textContent = "";
    deleteActions.style.display = "none";
    undoActions.style.display = "none";
    refRow.style.display = "none";
    setTransportEnabled(false);
    return;
  }

  textEl.textContent = sentence.text;
  metaEl.textContent =
    `#${sentence.sentence_index} · voice ${sentence.voice_id} · ${sentence.status}` +
    (sentence.description_tag ? ` · ${sentence.description_tag}` : "") +
    ` · +${sentence.silence_after_seconds.toFixed(2)}s gap`;

  document.getElementById("delete-target").textContent = sentence.file_name;
  document.getElementById("undo-target").textContent = sentence.file_name;
  deleteActions.style.display = sentence.status === "ok" ? "flex" : "none";
  undoActions.style.display = sentence.status === "pending_regen" ? "flex" : "none";

  refRow.style.display = "flex";
  document.getElementById("reference-audio-btn").disabled = !sentence.reference_audio_available;
  setTransportEnabled(sentence.exists);
  document.getElementById("reference-audio-btn").title = sentence.reference_audio_available
    ? "Play the reference voice used for this sentence"
    : "No comparable reference audio for this sentence";
}

function ensureWavesurfer() {
  if (state.wavesurfer) return state.wavesurfer;
  state.wavesurfer = WaveSurfer.create({
    container: "#waveform",
    waveColor: getComputedStyle(document.documentElement).getPropertyValue("--muted"),
    progressColor: getComputedStyle(document.documentElement).getPropertyValue("--accent"),
    height: 128,
    barWidth: 2,
    barGap: 1,
    cursorWidth: 1,
  });
  state.wavesurfer.on("audioprocess", updateTransportTime);
  state.wavesurfer.on("seeking", updateTransportTime);
  state.wavesurfer.on("ready", updateTransportTime);
  state.wavesurfer.on("ready", () => state.wavesurfer.setTime(0));
  state.wavesurfer.on("finish", onSentenceFinished);
  return state.wavesurfer;
}

function updateTransportTime() {
  const ws = state.wavesurfer;
  if (!ws) return;
  const hasAudio = hasPlayableSelection();
  const cur = hasAudio ? ws.getCurrentTime() : 0;
  const dur = hasAudio ? ws.getDuration() : 0;
  document.getElementById("transport-time").textContent = `${formatTime(cur)} / ${formatTime(dur)}`;
}

function formatTime(seconds) {
  if (!isFinite(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function selectSentence(index) {
  if (!state.playlist || index < 0 || index >= state.playlist.sentences.length) return;
  state.selectedIndex = index;
  const sentence = state.playlist.sentences[index];
  renderSentenceList();
  renderRightPane();
  scrollSelectedIntoView();

  const ws = ensureWavesurfer();
  if (sentence.exists) {
    const url = `/api/jobs/${state.activeJobId}/chapters/${state.activeChapterId}/sentences/${sentence.sentence_index}/audio`;
    ws.load(url);
  } else {
    ws.empty?.();
    updateTransportTime();
  }
}

function hasPlayableSelection() {
  return Boolean(state.wavesurfer && state.playlist?.sentences[state.selectedIndex]?.exists);
}

function togglePlayPause() {
  if (hasPlayableSelection()) state.wavesurfer.playPause();
}

function restartCurrentSentence() {
  if (!hasPlayableSelection()) return;
  state.wavesurfer.setTime(0);
  state.wavesurfer.play();
}

function scrollSelectedIntoView() {
  const el = document.querySelectorAll("#sentence-list .sentence-row")[state.selectedIndex];
  el?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function onSentenceFinished() {
  if (!state.continuousPlay || !state.playlist) return;
  const current = state.playlist.sentences[state.selectedIndex];
  const gapMs = Math.round((current?.silence_after_seconds || 0) * 1000);
  const nextIndex = nextPlayableIndex(state.selectedIndex);
  if (nextIndex == null) return;
  setTimeout(() => {
    selectSentence(nextIndex);
    state.wavesurfer.once("ready", () => state.wavesurfer.play());
  }, gapMs);
}

function nextPlayableIndex(fromIndex) {
  if (!state.playlist) return null;
  for (let i = fromIndex + 1; i < state.playlist.sentences.length; i++) {
    if (state.playlist.sentences[i].status === "ok") return i;
  }
  return null;
}

async function deleteSelectedSentence() {
  if (!state.playlist || state.selectedIndex == null) return;
  const sentence = state.playlist.sentences[state.selectedIndex];
  if (sentence.status !== "ok") return;
  const category = document.getElementById("issue-category").value || null;
  clearError();
  const chapter = state.chapters.find((c) => c.transcript_chapter_id === state.activeChapterId);
  const statusBefore = chapter?.synthesis_status;
  try {
    const result = await postJSON("/api/sentences/delete", {
      job_id: state.activeJobId,
      transcript_chapter_id: state.activeChapterId,
      sentence_index: sentence.sentence_index,
      issue_category: category,
    });
    let note = "";
    if (chapter && result.chapter_synthesis_status !== statusBefore) {
      chapter.synthesis_status = result.chapter_synthesis_status;
      renderChapterTree(state.chapters);
      note = `Chapter audio removed and chapter reset to ${result.chapter_synthesis_status}; run tts_batch.py to regenerate it.`;
    }
    await refreshPlaylist(state.selectedIndex + 1, note);
  } catch (err) {
    showError("Delete failed: " + err.message);
  }
}

async function undoSelectedSentence() {
  if (!state.playlist || state.selectedIndex == null) return;
  const sentence = state.playlist.sentences[state.selectedIndex];
  if (sentence.review_event_id == null) return;
  clearError();
  try {
    await postJSON(`/api/sentences/${sentence.review_event_id}/undo`, null);
    await refreshPlaylist(state.selectedIndex);
  } catch (err) {
    showError("Undo failed: " + err.message);
  }
}

async function refreshPlaylist(preferredIndex, note) {
  const playlist = await getJSON(`/api/jobs/${state.activeJobId}/chapters/${state.activeChapterId}`);
  state.playlist = playlist;
  renderChapterStatus(note);
  renderSentenceList();
  const target = Math.min(Math.max(preferredIndex, 0), playlist.sentences.length - 1);
  selectSentence(target);
}

function toggleReferenceAudio() {
  const sentence = state.playlist?.sentences[state.selectedIndex];
  if (!sentence || !sentence.reference_audio_available) return;
  if (!state.referenceAudioEl) {
    state.referenceAudioEl = new Audio();
  }
  const el = state.referenceAudioEl;
  const url = `/api/jobs/${state.activeJobId}/chapters/${state.activeChapterId}/sentences/${sentence.sentence_index}/reference-audio?voice_id=${sentence.voice_id}`;
  if (el.src.endsWith(url) && !el.paused) {
    el.pause();
    return;
  }
  el.src = url;
  el.play();
}

function isTypingTarget(el) {
  return el && (el.tagName === "INPUT" || el.tagName === "SELECT" || el.tagName === "TEXTAREA");
}

document.addEventListener("keydown", (ev) => {
  if (state.view !== "review" || isTypingTarget(ev.target)) return;
  if (ev.code === "Space") {
    ev.preventDefault();
    togglePlayPause();
  } else if (ev.key === "Home") {
    ev.preventDefault();
    restartCurrentSentence();
  } else if (ev.key === "Delete" || ev.key === "Backspace") {
    ev.preventDefault();
    deleteSelectedSentence();
  } else if (ev.key === "ArrowDown" || ev.key === "ArrowRight") {
    ev.preventDefault();
    if (state.selectedIndex != null) selectSentence(Math.min(state.selectedIndex + 1, state.playlist.sentences.length - 1));
  } else if (ev.key === "ArrowUp" || ev.key === "ArrowLeft") {
    ev.preventDefault();
    if (state.selectedIndex != null) selectSentence(Math.max(state.selectedIndex - 1, 0));
  }
});

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function init() {
  document.getElementById("back-to-landing").addEventListener("click", () => {
    showView("landing");
    loadJobs();
  });
  document.getElementById("switch-repo-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const path = document.getElementById("repo-input").value.trim();
    if (path) switchRepo(path);
  });
  document.getElementById("issue-category").innerHTML =
    '<option value="">(uncategorized)</option>' +
    ISSUE_CATEGORIES.map((c) => `<option value="${c}">${c}</option>`).join("");
  document.getElementById("delete-btn").addEventListener("click", deleteSelectedSentence);
  document.getElementById("undo-btn").addEventListener("click", undoSelectedSentence);
  document.getElementById("reference-audio-btn").addEventListener("click", toggleReferenceAudio);
  document.getElementById("restart-btn").addEventListener("click", restartCurrentSentence);
  document.getElementById("play-pause-btn").addEventListener("click", togglePlayPause);
  document.getElementById("continuous-toggle").addEventListener("change", (ev) => {
    state.continuousPlay = ev.target.checked;
  });

  ensureWavesurfer();
  loadJobs();
}

init();

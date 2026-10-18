/* LeafScan: on-device plant disease detection with ONNX Runtime Web.
 *
 * Pipeline (matches the evaluation transform used in training):
 *   photo -> center square (Resize 256 + CenterCrop 224, optionally zoomed)
 *         -> 224x224 RGB, scaled to 0..1, CHW float32
 *         -> MobileNetV3-Large (normalization + softmax are inside the model)
 *         -> class probabilities, mapped to records in the expert knowledge base (data/kb.json)
 */
(() => {
  'use strict';

  const MODEL_URL = 'model/plant_disease_mnv3.onnx';
  const LABELS_URL = 'model/labels.json';
  const KB_URL = 'data/kb.json';
  const SIZE = 224;
  const RESIZE = 256;
  const CONFIDENT = 0.70;   // at or above: show result normally
  const UNSURE = 0.45;      // below: say we are not sure
  const HISTORY_KEY = 'leafscan.history.v1';
  const HISTORY_MAX = 8;

  const $ = (id) => document.getElementById(id);
  const el = {
    dropzone: $('dropzone'), empty: $('emptyState'), previewWrap: $('previewWrap'),
    preview: $('previewCanvas'), cropBox: $('cropBox'), zoomRow: $('zoomRow'), zoom: $('zoom'),
    zoomOut: $('zoomOut'), cameraBtn: $('cameraBtn'), uploadBtn: $('uploadBtn'),
    cameraInput: $('cameraInput'), fileInput: $('fileInput'), status: $('status'), progress: $('progress'),
    resultCard: $('resultCard'), chip: $('resultChip'), time: $('resultTime'), plant: $('resultPlant'),
    name: $('resultName'), meter: $('resultMeter'), bar: $('resultBar'), conf: $('resultConf'),
    warning: $('resultWarning'), details: $('resultDetails'), sci: $('resultSci'), symptoms: $('resultSymptoms'),
    symptomsTitle: $('symptomsTitle'), lookTitle: $('lookTitle'), look: $('resultLook'), manageTitle: $('manageTitle'),
    manage: $('resultManage'), sourcesBox: $('sourcesBox'), sources: $('resultSources'), note: $('resultNote'),
    others: $('resultOthers'), confirmBox: $('confirmBox'), confirmList: $('confirmList'), confirmResult: $('confirmResult'),
    emptySub: $('emptySub'),
    historyCard: $('historyCard'), historyList: $('historyList'), clearHistory: $('clearHistory'),
    cropList: $('cropList'), installBtn: $('installBtn'), offlineNote: $('offlineNote'),
    dialog: $('cameraDialog'), video: $('cameraVideo'), shutter: $('cameraShutter'),
    cameraCancel: $('cameraCancel'), cameraSwitch: $('cameraSwitch'),
  };

  let session = null;
  let labels = [];
  let kb = { hosts: {}, conditions: [] };
  let byId = new Map();     // knowledge-base id -> condition record
  let entries = [];         // per model output index: { host, record|null, healthy }
  let source = null;        // { image: CanvasImageSource, w, h }
  let zoomTimer = 0;
  let busy = false;

  // ---------- Model loading ----------

  ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

  function setStatus(text) { el.status.textContent = text; }
  function setProgress(fraction) {
    el.progress.querySelector('span').style.width = `${Math.min(100, Math.round(fraction * 100))}%`;
  }

  async function fetchWithProgress(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
    const total = Number(res.headers.get('Content-Length')) || 0;
    if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress(received / total);
    }
    const bytes = new Uint8Array(received);
    let offset = 0;
    for (const c of chunks) { bytes.set(c, offset); offset += c.length; }
    return bytes;
  }

  async function init() {
    renderHistory();
    try {
      const [labelsRes, kbRes] = await Promise.all([fetch(LABELS_URL), fetch(KB_URL)]);
      labels = await labelsRes.json();
      kb = await kbRes.json();
      byId = new Map(kb.conditions.map((c) => [c.id, c]));
      entries = labels.map(mapLabel);
      renderCrops();
      renderHistory();

      setStatus('Downloading the AI model (first time only)…');
      const bytes = await fetchWithProgress(MODEL_URL, setProgress);
      setStatus('Starting the AI model…');
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      // Warm-up so the first real scan is fast
      await session.run({ image: new ort.Tensor('float32', new Float32Array(3 * SIZE * SIZE), [1, 3, SIZE, SIZE]) });

      el.progress.classList.add('done');
      el.cameraBtn.disabled = false;
      el.uploadBtn.disabled = false;
      setStatus('Ready. Take or upload a photo of a leaf.');
    } catch (err) {
      console.error(err);
      setStatus('Could not load the AI model. Check your connection and reload the page.');
    }
  }

  // ---------- Image input ----------

  async function loadFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      setStatus('That file is not an image. Please choose a photo.');
      return;
    }
    const url = URL.createObjectURL(file);
    try {
      // createImageBitmap also works while the page is in the background (e.g. returning from the
      // phone's camera app) and applies the photo's EXIF rotation; fall back to <img> where it is missing.
      let image = null;
      if (window.createImageBitmap) {
        try { image = await createImageBitmap(file, { imageOrientation: 'from-image' }); } catch { image = null; }
      }
      if (!image) {
        const img = new Image();
        await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = reject; img.src = url; });
        image = img;
      }
      setSource(image, image.naturalWidth || image.width, image.naturalHeight || image.height);
    } catch (err) {
      console.error(err);
      setStatus('Could not open that image. Try a JPG or PNG photo.');
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  function setSource(image, w, h) {
    source = { image, w, h, id: Date.now() };
    el.zoom.value = '1';
    el.zoomOut.textContent = '1.0×';
    drawPreview();
    analyze();
  }

  function cropRect(zoom) {
    const side = Math.min(source.w, source.h) * (SIZE / RESIZE) / zoom;
    return { x: (source.w - side) / 2, y: (source.h - side) / 2, side };
  }

  function drawPreview() {
    el.empty.hidden = true;
    el.previewWrap.hidden = false;
    el.zoomRow.hidden = false;
    const maxSide = 1024;
    const scale = Math.min(1, maxSide / Math.max(source.w, source.h));
    el.preview.width = Math.round(source.w * scale);
    el.preview.height = Math.round(source.h * scale);
    const ctx = el.preview.getContext('2d');
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(source.image, 0, 0, el.preview.width, el.preview.height);
    positionCropBox();
  }

  function positionCropBox() {
    if (!source) return;
    // Fit the photo inside the square preview, scaling small images up and large ones down.
    const box = el.previewWrap.getBoundingClientRect();
    const fit = Math.min(box.width / source.w, box.height / source.h);
    el.preview.style.width = `${source.w * fit}px`;
    el.preview.style.height = `${source.h * fit}px`;
    const wrap = el.previewWrap.getBoundingClientRect();
    const cv = el.preview.getBoundingClientRect();
    const k = cv.width / source.w;
    const r = cropRect(Number(el.zoom.value));
    Object.assign(el.cropBox.style, {
      left: `${cv.left - wrap.left + r.x * k}px`,
      top: `${cv.top - wrap.top + r.y * k}px`,
      width: `${r.side * k}px`,
      height: `${r.side * k}px`,
    });
  }

  // Downscale in halving steps so large phone photos are smoothed like PIL's antialiased resize.
  function cropTo224(zoom) {
    const r = cropRect(zoom);
    let src = source.image, sx = r.x, sy = r.y, side = r.side;
    while (side > SIZE * 2) {
      const half = Math.round(side / 2);
      const step = document.createElement('canvas');
      step.width = step.height = half;
      const sctx = step.getContext('2d');
      sctx.imageSmoothingQuality = 'high';
      sctx.drawImage(src, sx, sy, side, side, 0, 0, half, half);
      src = step; sx = 0; sy = 0; side = half;
    }
    const out = document.createElement('canvas');
    out.width = out.height = SIZE;
    const ctx = out.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(src, sx, sy, side, side, 0, 0, SIZE, SIZE);
    return out;
  }

  function toTensor(canvas) {
    const { data } = canvas.getContext('2d').getImageData(0, 0, SIZE, SIZE);
    const plane = SIZE * SIZE;
    const chw = new Float32Array(3 * plane);
    for (let i = 0, p = 0; p < plane; i += 4, p++) {
      chw[p] = data[i] / 255;
      chw[p + plane] = data[i + 1] / 255;
      chw[p + 2 * plane] = data[i + 2] / 255;
    }
    return new ort.Tensor('float32', chw, [1, 3, SIZE, SIZE]);
  }

  // ---------- Inference and results ----------

  async function analyze() {
    if (!session || !source) return;
    if (busy) { clearTimeout(zoomTimer); zoomTimer = setTimeout(analyze, 80); return; }
    busy = true;
    setStatus('Analyzing…');
    try {
      const canvas = cropTo224(Number(el.zoom.value));
      const t0 = performance.now();
      const output = await session.run({ image: toTensor(canvas) });
      const ms = performance.now() - t0;
      const probs = Array.from(output.probabilities.data);
      const ranked = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
      showResult(ranked, ms);
      saveHistory(canvas, ranked[0]);
      setStatus('Done. Use the zoom slider if the leaf is small in the photo.');
    } catch (err) {
      console.error(err);
      setStatus('Something went wrong while analyzing. Please try another photo.');
    } finally {
      busy = false;
    }
  }

  // ---------- Knowledge base ----------

  // PlantVillage folder names (used by the current model) -> knowledge-base host keys.
  const PV_HOST = {
    'Apple': 'apple', 'Blueberry': 'blueberry', 'Cherry_(including_sour)': 'cherry', 'Corn_(maize)': 'corn',
    'Grape': 'grape', 'Orange': 'citrus', 'Peach': 'peach', 'Pepper,_bell': 'bell_pepper', 'Potato': 'potato',
    'Raspberry': 'raspberry', 'Soybean': 'soybean', 'Squash': 'squash', 'Strawberry': 'strawberry', 'Tomato': 'tomato',
  };

  // A model label is either a PlantVillage name ("Tomato___Early_blight") or a knowledge-base id ("tomato/early-blight").
  function mapLabel(label) {
    const id = label.id;
    if (id.includes('___')) {
      const host = PV_HOST[id.split('___')[0]];
      if (id.endsWith('___healthy')) return { host, record: null, healthy: true };
      return { host, record: kb.conditions.find((c) => c.pv === id) || null, healthy: false };
    }
    const [host, slug] = id.split('/');
    return { host, record: slug === 'healthy' ? null : byId.get(id) || null, healthy: slug === 'healthy' };
  }

  function describe(i) {
    const e = entries[i] || {};
    const h = kb.hosts[e.host] || { en: labels[i]?.plant || 'Unknown plant' };
    const r = e.record;
    return {
      host: e.host, plant: h.en, plantFil: h.fil, hostSci: h.scientific, hostRank: h.rank, healthy: !!e.healthy,
      name: e.healthy ? 'Healthy' : (r ? r.name : (labels[i]?.condition || '').replace(/_/g, ' ')),
      type: e.healthy ? 'Healthy' : (r ? r.type : 'Unknown'), record: r,
    };
  }

  function kindOf(d) {
    if (d.healthy) return 'healthy';
    if (d.type.startsWith('Pest')) return 'pest';
    return 'disease';
  }

  function pct(p) { return `${(p * 100).toFixed(p >= 0.995 ? 0 : 1)}%`; }

  function italic(text) {
    const i = document.createElement('i');
    i.textContent = text;
    return i;
  }

  // "Solanum lycopersicum · Alternaria linariae (fungus)"; pathovars and formae speciales stay upright.
  function sciLine(d) {
    const parts = [];
    if (d.hostSci) parts.push(italic(d.hostSci), d.hostRank === 'genus' ? ' spp.' : '');
    const r = d.record;
    if (r && r.agent) {
      parts.push(' · ');
      const m = r.agent.match(/^(.*?)( (?:pv\.|f\. sp\.|subsp\.) )(.*)$/);
      if (r.type === 'Virus') parts.push(r.agent);
      else if (m) parts.push(italic(m[1]), m[2], italic(m[3]));
      else parts.push(italic(r.agent));
      parts.push(` (${r.type.toLowerCase()})`);
    } else if (r) {
      parts.push(` · ${r.type}`);
    }
    return parts;
  }

  function showResult(ranked, ms) {
    const top = ranked[0];
    const d = describe(top.i);
    const unsure = top.p < UNSURE;
    const kind = unsure ? 'unsure' : kindOf(d);
    const chipText = { healthy: 'Healthy', disease: 'Disease detected', pest: 'Pest detected', unsure: 'Not sure' }[kind];

    el.resultCard.hidden = false;
    el.chip.className = `chip ${kind}`;
    el.chip.textContent = chipText;
    el.time.textContent = `Analyzed on this device in ${Math.round(ms)} ms`;
    el.meter.className = `meter ${kind}`;
    el.bar.style.width = `${Math.round(top.p * 100)}%`;
    el.meter.setAttribute('aria-valuenow', String(Math.round(top.p * 100)));

    const plantLabel = d.plantFil ? `${d.plant} · ${d.plantFil}` : d.plant;
    el.plant.textContent = unsure ? `Best guess: ${plantLabel}` : plantLabel;
    el.name.textContent = unsure ? `${d.name}?` : d.name;
    el.sci.replaceChildren(...sciLine(d));
    el.conf.textContent = unsure ? `Confidence: ${pct(top.p)}, too low to trust` : `Confidence: ${pct(top.p)}`;
    el.warning.hidden = top.p >= CONFIDENT;
    el.warning.textContent = unsure
      ? 'The photo may not show one of the supported plants, may not be a leaf, or may be blurry or too dark. Retake it with one leaf on a plain background in good light, or zoom in on the leaf.'
      : 'Moderate confidence. Retake the photo closer to the leaf, or check the other possibilities.';
    el.details.hidden = unsure;
    if (!unsure) renderDetails(d);
    renderConfirm(ranked, top.p);
    if (!el.confirmBox.hidden && !unsure) {
      el.warning.textContent = 'Moderate confidence. Answer the questions below to narrow it down, or retake the photo closer to the leaf.';
    }

    el.others.replaceChildren(...ranked.slice(1, 4).map(({ p, i }) => {
      const o = describe(i);
      const li = document.createElement('li');
      li.append(`${o.plant}: ${o.name} `);
      const span = document.createElement('span');
      span.textContent = `(${pct(p)})`;
      li.append(span);
      return li;
    }));

    if (window.matchMedia('(max-width: 900px)').matches) {
      el.resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  function renderDetails(d) {
    const r = d.record;
    if (d.healthy || !r) {
      el.symptomsTitle.textContent = 'What this means';
      el.symptoms.textContent = d.healthy
        ? `No disease was detected on this ${d.plant.toLowerCase()} leaf.`
        : 'Details for this result are not in the knowledge base yet.';
      el.lookTitle.hidden = el.look.hidden = true;
      el.manageTitle.textContent = 'Keep it healthy';
      fillList(el.manage, [
        'Check your plants every few days, especially older lower leaves.',
        'Water at the base of the plant in the morning so leaves stay dry.',
        'Space plants for good air flow and remove weeds and fallen leaves.',
      ]);
      el.sourcesBox.hidden = true;
      return;
    }
    el.symptomsTitle.textContent = 'What it looks like';
    el.symptoms.textContent = r.symptoms;
    el.lookTitle.hidden = el.look.hidden = false;
    fillList(el.look, r.look_for.map((q) => q.replace(/\?$/, '')));
    el.manageTitle.textContent = 'What you can do';
    fillList(el.manage, r.management);
    el.sourcesBox.hidden = false;
    el.sources.replaceChildren(...r.sources.map((s) => {
      const li = document.createElement('li');
      if (s.url) {
        const a = document.createElement('a');
        a.href = s.url;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        a.textContent = s.title;
        li.append(a);
      } else {
        li.append(s.title);
      }
      const meta = document.createElement('div');
      meta.className = 'src-meta';
      meta.textContent = [s.publisher, s.kind, s.year, s.license].filter(Boolean).join(' · ');
      li.append(meta);
      return li;
    }));
    const notes = [r.review_note, r.name_note && `Name: ${r.name_note}`, kb.note].filter(Boolean);
    el.note.textContent = notes.join(' ');
  }

  // When the photo is not conclusive, ask about visible signs listed in the expert records of the top
  // candidates and re-rank them: each "yes" doubles a candidate's weight, each "no" halves it.
  let confirmState = null;

  function renderConfirm(ranked, topP) {
    // The pool keeps every likely candidate, including "healthy" (which has no questions), so that
    // answering "no" to all disease signs can still lead to a healthy result.
    const pool = ranked.slice(0, 3).filter(({ p }) => p >= 0.08).map(({ p, i }) => ({ p, i, d: describe(i) }));
    const cands = pool.filter((c) => c.d.record && c.d.record.look_for && c.d.record.look_for.length);
    if (topP >= CONFIDENT || pool.length < 2 || cands.length < 1) {
      el.confirmBox.hidden = true;
      confirmState = null;
      return;
    }
    confirmState = { pool, cands, answers: {} };
    el.confirmBox.hidden = false;
    el.confirmResult.hidden = true;
    const items = [];
    for (const c of cands) {
      c.d.record.look_for.forEach((q, n) => {
        const key = `${c.i}:${n}`;
        const li = document.createElement('li');
        const label = document.createElement('span');
        label.className = 'confirm-q';
        label.textContent = `${q} `;
        const small = document.createElement('small');
        small.textContent = `(sign of ${c.d.name.toLowerCase()})`;
        label.append(small);
        const seg = document.createElement('div');
        seg.className = 'seg';
        seg.setAttribute('role', 'group');
        seg.setAttribute('aria-label', q);
        for (const [val, text] of [['yes', 'Yes'], ['no', 'No'], ['skip', 'Not sure']]) {
          const b = document.createElement('button');
          b.type = 'button';
          b.textContent = text;
          b.setAttribute('aria-pressed', 'false');
          b.addEventListener('click', () => {
            confirmState.answers[key] = val;
            seg.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
            applyConfirm();
          });
          seg.append(b);
        }
        li.append(label, seg);
        items.push(li);
      });
    }
    el.confirmList.replaceChildren(...items);
  }

  function applyConfirm() {
    if (!confirmState) return;
    const scored = confirmState.pool.map((c) => {
      let w = c.p;
      (c.d.record?.look_for || []).forEach((_, n) => {
        const a = confirmState.answers[`${c.i}:${n}`];
        if (a === 'yes') w *= 2;
        if (a === 'no') w *= 0.5;
      });
      return { ...c, w };
    });
    const total = scored.reduce((sum, c) => sum + c.w, 0);
    scored.sort((a, b) => b.w - a.w);
    const best = scored[0];
    el.confirmResult.hidden = false;
    el.confirmResult.textContent = `Best match after your answers: ${best.d.plant}: ${best.d.name} ` +
      `(${pct(best.w / total)} among these candidates). This combines the photo with your answers; confirm with a technician.`;
    el.name.textContent = best.d.name;
    el.sci.replaceChildren(...sciLine(best.d));
    renderDetails(best.d);
    el.details.hidden = false;
  }

  function fillList(ul, items) {
    ul.replaceChildren(...items.map((t) => {
      const li = document.createElement('li');
      li.textContent = t;
      return li;
    }));
  }

  function renderCrops() {
    const crops = new Map();
    entries.forEach((e) => {
      if (!e.host) return;
      crops.set(e.host, (crops.get(e.host) || 0) + (e.healthy ? 0 : 1));
    });
    const problems = [...crops.values()].reduce((a, b) => a + b, 0);
    el.emptySub.textContent = `Recognizes ${problems} problems on ${crops.size} plants. The photo never leaves your device.`;
    const sorted = [...crops].sort((a, b) => (kb.hosts[a[0]]?.en || '').localeCompare(kb.hosts[b[0]]?.en || ''));
    el.cropList.replaceChildren(...sorted.map(([host, n]) => {
      const h = kb.hosts[host] || { en: host };
      const li = document.createElement('li');
      li.append(`${h.en} `);
      if (h.scientific) li.append(italic(h.scientific), h.rank === 'genus' ? ' spp. ' : ' ');
      const span = document.createElement('span');
      span.textContent = n ? `· ${n} ${n === 1 ? 'problem' : 'problems'}` : '· healthy only';
      li.append(span);
      return li;
    }));
  }

  // ---------- History (per-device convenience only) ----------

  function readHistory() {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; }
  }

  function saveHistory(canvas, top) {
    try {
      const thumb = document.createElement('canvas');
      thumb.width = thumb.height = 88;
      thumb.getContext('2d').drawImage(canvas, 0, 0, 88, 88);
      const items = readHistory();
      const entry = { t: Date.now(), id: labels[top.i].id, p: top.p, sid: source.id, img: thumb.toDataURL('image/jpeg', 0.7) };
      // Zoom changes re-analyze the same photo: replace its entry instead of stacking duplicates.
      if (items[0] && items[0].sid === source.id) items.shift();
      items.unshift(entry);
      localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, HISTORY_MAX)));
    } catch { /* storage unavailable: history is optional */ }
    renderHistory();
  }

  function renderHistory() {
    const items = readHistory();
    el.historyCard.hidden = items.length === 0;
    el.historyList.replaceChildren(...items.map((h) => {
      const idx = labels.findIndex((l) => l.id === h.id);
      const d = idx >= 0 ? describe(idx) : { plant: h.id.split(/___|\//)[0], name: h.id.split(/___|\//)[1] || '' };
      const li = document.createElement('li');
      const img = document.createElement('img');
      img.src = h.img; img.alt = ''; img.width = 44; img.height = 44;
      const text = document.createElement('div');
      text.className = 'h-text';
      const name = document.createElement('div');
      name.className = 'h-name';
      name.textContent = h.p < UNSURE ? `Not sure (maybe ${d.plant}: ${d.name})` : `${d.plant}: ${d.name}`;
      const meta = document.createElement('div');
      meta.className = 'h-meta';
      meta.textContent = `${pct(h.p)} · ${new Date(h.t).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}`;
      text.append(name, meta);
      li.append(img, text);
      return li;
    }));
  }

  el.clearHistory.addEventListener('click', () => {
    try { localStorage.removeItem(HISTORY_KEY); } catch { /* ignore */ }
    renderHistory();
  });

  // ---------- Camera ----------

  let stream = null;
  let facing = 'environment';

  async function openLiveCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false,
      });
      el.video.srcObject = stream;
      if (!el.dialog.open) el.dialog.showModal();
    } catch (err) {
      console.warn('Live camera unavailable, using the file picker instead', err);
      closeLiveCamera();
      el.cameraInput.click();
    }
  }

  function closeLiveCamera() {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    el.video.srcObject = null;
    if (el.dialog.open) el.dialog.close();
  }

  el.cameraBtn.addEventListener('click', () => {
    const phoneLike = window.matchMedia('(pointer: coarse)').matches;
    if (phoneLike || !navigator.mediaDevices?.getUserMedia || !window.isSecureContext) {
      el.cameraInput.click(); // native camera app on phones and tablets
    } else {
      openLiveCamera();
    }
  });

  el.shutter.addEventListener('click', () => {
    const w = el.video.videoWidth, h = el.video.videoHeight;
    if (!w || !h) return;
    const shot = document.createElement('canvas');
    shot.width = w; shot.height = h;
    shot.getContext('2d').drawImage(el.video, 0, 0, w, h);
    closeLiveCamera();
    setSource(shot, w, h);
  });
  el.cameraCancel.addEventListener('click', closeLiveCamera);
  el.dialog.addEventListener('cancel', closeLiveCamera);
  el.cameraSwitch.addEventListener('click', () => {
    facing = facing === 'environment' ? 'user' : 'environment';
    if (stream) stream.getTracks().forEach((t) => t.stop());
    openLiveCamera();
  });

  // ---------- Other inputs ----------

  el.uploadBtn.addEventListener('click', () => el.fileInput.click());
  for (const input of [el.fileInput, el.cameraInput]) {
    input.addEventListener('change', () => {
      if (input.files[0]) loadFile(input.files[0]);
      input.value = '';
    });
  }

  el.dropzone.addEventListener('dragover', (e) => { e.preventDefault(); el.dropzone.classList.add('dragging'); });
  el.dropzone.addEventListener('dragleave', () => el.dropzone.classList.remove('dragging'));
  el.dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    el.dropzone.classList.remove('dragging');
    if (session && e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
  });
  el.dropzone.addEventListener('click', () => { if (session && !source) el.fileInput.click(); });
  el.dropzone.addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && session) { e.preventDefault(); el.fileInput.click(); }
  });
  document.addEventListener('paste', (e) => {
    const file = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith('image/'));
    if (file && session) loadFile(file);
  });

  el.zoom.addEventListener('input', () => {
    el.zoomOut.textContent = `${Number(el.zoom.value).toFixed(1)}×`;
    positionCropBox();
    clearTimeout(zoomTimer);
    zoomTimer = setTimeout(analyze, 150);
  });
  window.addEventListener('resize', positionCropBox);

  // ---------- Install and offline ----------

  let installEvent = null;
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    installEvent = e;
    el.installBtn.hidden = false;
  });
  el.installBtn.addEventListener('click', async () => {
    if (!installEvent) return;
    installEvent.prompt();
    await installEvent.userChoice;
    installEvent = null;
    el.installBtn.hidden = true;
  });

  const isIOS = /iphone|ipad|ipod/i.test(navigator.userAgent) ||
    (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  const standalone = window.matchMedia('(display-mode: standalone)').matches || navigator.standalone;

  if ('serviceWorker' in navigator && window.isSecureContext) {
    navigator.serviceWorker.register('sw.js').then(() => navigator.serviceWorker.ready).then(() => {
      el.offlineNote.textContent = 'Saved on this device: LeafScan works without internet after the first visit.' +
        (isIOS && !standalone ? ' On iPhone or iPad, tap Share → Add to Home Screen to install it.' : '');
    }).catch((err) => console.warn('Service worker failed', err));
  }

  init();
})();

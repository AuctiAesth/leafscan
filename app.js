/* LeafScan: on-device plant disease detection with ONNX Runtime Web.
 *
 * Pipeline (matches the evaluation transform used in training):
 *   photo -> the square under the frame (by default the center square of Resize 256 + CenterCrop 224;
 *            the user can move and zoom the photo underneath the frame)
 *         -> 224x224 RGB, scaled to 0..1, CHW float32
 *         -> MobileNetV3-Large (normalization + softmax are inside the model)
 *         -> class probabilities (averaged over leaves when the user adds more than one),
 *            mapped to records in the expert knowledge base (data/kb.json)
 *
 * The photo is checked once when it is loaded. Moving or zooming it never re-checks on a timer: the result
 * is marked as out of date until the user taps "Check this leaf" (or, if they turn it on, until they let go).
 * Results are updated in place and the page never scrolls by itself.
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
  // Out of 10 answers in each band, how many were right on 3,100 held-out field photos
  // (92.6%, 69.4% and 39.4%; training/calibrate_app.py -> results/app_calibration.json).
  const RIGHT_IN_10 = { confident: 9, moderate: 7, unsure: 4 };
  // Photo checks on the 224x224 crop, set from the same field photos: blurred copies score below 20 while
  // fewer than 1% of real field photos do; crops darker than 50 (of 255) lose accuracy.
  const BLURRY = 20;
  const DARK = 50;
  const ZOOM_STEP = 1.25;
  const MAX_LEAVES = 3;
  const AUTO_DELAY = 350;   // ms after the user lets go, when automatic checking is on
  const HISTORY_KEY = 'leafscan.history.v1';
  const AUTO_KEY = 'leafscan.autocheck.v1';
  const HISTORY_MAX = 8;

  const $ = (id) => document.getElementById(id);
  const el = {
    stage: $('stage'), empty: $('emptyState'), emptySub: $('emptySub'), preview: $('previewCanvas'), frame: $('frame'),
    tools: $('tools'), zoomOutBtn: $('zoomOutBtn'), zoomInBtn: $('zoomInBtn'), zoomLevel: $('zoomLevel'),
    resetBtn: $('resetBtn'), retakeBtn: $('retakeBtn'), pickBtn: $('pickBtn'), hint: $('hint'),
    addBanner: $('addBanner'), addText: $('addText'), addCancel: $('addCancel'),
    actions: $('actions'), cameraBtn: $('cameraBtn'), uploadBtn: $('uploadBtn'),
    cameraInput: $('cameraInput'), fileInput: $('fileInput'), status: $('status'), progress: $('progress'),
    summaryCard: $('summaryCard'), summaryBody: $('summaryBody'), chip: $('resultChip'), basis: $('resultBasis'),
    plant: $('resultPlant'), name: $('resultName'), sci: $('resultSci'), meter: $('resultMeter'), bar: $('resultBar'),
    conf: $('resultConf'), confWord: $('confWord'), delta: $('resultDelta'), message: $('resultMessage'),
    staleBar: $('staleBar'), checkBtn: $('checkBtn'), leafStrip: $('leafStrip'), addLeafBtn: $('addLeafBtn'),
    autoCheck: $('autoCheck'),
    detailsCard: $('detailsCard'), details: $('resultDetails'), symptomsTitle: $('symptomsTitle'), symptoms: $('resultSymptoms'),
    lookTitle: $('lookTitle'), look: $('resultLook'), manageTitle: $('manageTitle'), manage: $('resultManage'),
    chemNote: $('chemNote'), sourcesBox: $('sourcesBox'), sources: $('resultSources'), note: $('resultNote'),
    others: $('resultOthers'), confirmBox: $('confirmBox'), confirmList: $('confirmList'), confirmResult: $('confirmResult'),
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
  let source = null;        // current photo: { image: CanvasImageSource, w, h, id }
  const view = { cx: 0, cy: 0, zoom: 1 };  // the square under the frame, in photo pixels
  let checked = null;       // the square behind the result on screen: { cx, cy, zoom, id }
  let leaves = [];          // leaves of the current plant: { id, probs, thumb, quality }
  let plantId = 0;          // groups the history entry of one plant
  let adding = false;       // the next photo is another leaf of the same plant
  let shown = null;         // the last result shown for this plant, for the "changed by" line
  let busy = false;
  let again = false;
  let autoTimer = 0;

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
    try { el.autoCheck.checked = localStorage.getItem(AUTO_KEY) === '1'; } catch { /* ignore */ }
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
      setStatus(source ? '' : 'Ready. Take a photo of one leaf, or choose one from your gallery.');
      if (source) analyze();
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
    if (source && source.image !== image && typeof ImageBitmap !== 'undefined' && source.image instanceof ImageBitmap) {
      source.image.close(); // free the previous photo's memory
    }
    if (!adding) {
      leaves = [];
      plantId = Date.now();
      shown = null;
    }
    setAdding(false);
    source = { image, w, h, id: `${Date.now()}-${Math.random()}` };
    checked = null;
    resetView();
    drawPreview();
    el.empty.hidden = true;
    el.preview.hidden = el.frame.hidden = el.tools.hidden = el.hint.hidden = false;
    el.actions.hidden = true;
    el.stage.classList.add('has-photo');
    el.stage.setAttribute('aria-label',
      'Leaf photo. Drag or pinch to move the leaf into the square, or use the arrow and plus or minus keys. Press Enter to check.');
    setStatus(session ? '' : 'The photo is ready. Waiting for the AI model to finish loading…');
    scheduleRender();
    updateStale();
    analyze();
  }

  function drawPreview() {
    const maxSide = navigator.deviceMemory && navigator.deviceMemory <= 2 ? 1280 : 2048;
    const scale = Math.min(1, maxSide / Math.max(source.w, source.h));
    el.preview.width = Math.max(1, Math.round(source.w * scale));
    el.preview.height = Math.max(1, Math.round(source.h * scale));
    el.preview.style.width = `${el.preview.width}px`;
    el.preview.style.height = `${el.preview.height}px`;
    const ctx = el.preview.getContext('2d');
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(source.image, 0, 0, el.preview.width, el.preview.height);
  }

  // ---------- The square under the frame ----------

  // Side of the square at zoom 1: the CenterCrop 224 of Resize 256 used in training.
  const baseSide = () => Math.min(source.w, source.h) * (SIZE / RESIZE);
  // Stop zooming in before the square has fewer than 112 photo pixels across.
  const maxZoom = () => Math.max(1, Math.min(5, baseSide() / 112));

  function resetView() {
    view.zoom = 1;
    view.cx = source.w / 2;
    view.cy = source.h / 2;
  }

  function clampView() {
    view.zoom = Math.min(maxZoom(), Math.max(1, view.zoom));
    const half = baseSide() / view.zoom / 2;
    view.cx = Math.min(source.w - half, Math.max(half, view.cx));
    view.cy = Math.min(source.h - half, Math.max(half, view.cy));
  }

  function cropRect(v) {
    const side = baseSide() / v.zoom;
    return { x: v.cx - side / 2, y: v.cy - side / 2, side };
  }

  // Frame position inside the stage (CSS px) and screen pixels per photo pixel.
  function frameBox() {
    const F = el.frame.offsetWidth;
    return { x: el.frame.offsetLeft, y: el.frame.offsetTop, F, k: F / (baseSide() / view.zoom) };
  }

  function toPhoto(px, py) {
    const f = frameBox();
    return { x: view.cx + (px - f.x - f.F / 2) / f.k, y: view.cy + (py - f.y - f.F / 2) / f.k };
  }

  // Keep the photo point (ax, ay) under the stage point (px, py) at the current zoom.
  function anchor(a, px, py) {
    const f = frameBox();
    view.cx = a.x - (px - f.x - f.F / 2) / f.k;
    view.cy = a.y - (py - f.y - f.F / 2) / f.k;
  }

  let renderReq = 0;
  let glideTimer = 0;
  function scheduleRender(glide = false) {
    if (!source) return;
    el.preview.classList.toggle('glide', glide);
    clearTimeout(glideTimer);
    if (glide) glideTimer = setTimeout(() => el.preview.classList.remove('glide'), 260);
    if (!renderReq) renderReq = requestAnimationFrame(applyTransform);
  }

  function applyTransform() {
    renderReq = 0;
    if (!source || !el.frame.offsetWidth) return;
    const f = frameBox();
    const s = f.k * source.w / el.preview.width;
    const tx = f.x + f.F / 2 - view.cx * f.k;
    const ty = f.y + f.F / 2 - view.cy * f.k;
    el.preview.style.transform = `translate(${tx}px, ${ty}px) scale(${s})`;
    el.zoomLevel.textContent = `${view.zoom.toFixed(1)}×`;
    el.zoomOutBtn.disabled = view.zoom <= 1.001;
    el.zoomInBtn.disabled = view.zoom >= maxZoom() - 0.001;
  }

  function viewChanged(glide = false) {
    clampView();
    scheduleRender(glide);
    updateStale();
  }

  function isStale() {
    if (!checked || !source) return false;
    if (checked.id !== source.id) return true;
    const tol = baseSide() / view.zoom * 0.01;
    return Math.abs(checked.zoom - view.zoom) > 0.01 || Math.hypot(checked.cx - view.cx, checked.cy - view.cy) > tol;
  }

  function updateStale() {
    const stale = isStale();
    el.summaryBody.classList.toggle('stale', stale);
    el.detailsCard.classList.toggle('details-stale', stale);
    el.staleBar.hidden = !stale;
    el.message.style.visibility = stale ? 'hidden' : '';
  }

  // The user let go of the photo (or finished a button press): check again only if they asked for it.
  function settle() {
    clearTimeout(autoTimer);
    if (el.autoCheck.checked && isStale()) autoTimer = setTimeout(analyze, AUTO_DELAY);
  }

  // ---------- Gestures: drag to move, pinch or wheel to zoom, tap to center ----------

  const pointers = new Map();
  let pinch = null;
  let tap = null;

  function localPoint(e) {
    const r = el.stage.getBoundingClientRect();
    return { x: e.clientX - r.left - el.stage.clientLeft, y: e.clientY - r.top - el.stage.clientTop };
  }

  function startPinch() {
    const [a, b] = [...pointers.values()];
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    pinch = { dist: Math.hypot(a.x - b.x, a.y - b.y) || 1, zoom: view.zoom, anchor: toPhoto(mid.x, mid.y) };
  }

  el.stage.addEventListener('pointerdown', (e) => {
    if (!source || (e.pointerType === 'mouse' && e.button !== 0)) return;
    e.preventDefault();
    el.stage.focus({ preventScroll: true });
    try { el.stage.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    clearTimeout(autoTimer);
    el.preview.classList.remove('glide');
    const pt = localPoint(e);
    pointers.set(e.pointerId, pt);
    el.stage.classList.add('grabbing');
    if (pointers.size === 1) {
      tap = { x: pt.x, y: pt.y, t: performance.now() };
    } else {
      tap = null;
      startPinch();
    }
  });

  el.stage.addEventListener('pointermove', (e) => {
    if (!pointers.has(e.pointerId)) return;
    const pt = localPoint(e);
    let prev = pointers.get(e.pointerId);
    pointers.set(e.pointerId, pt);
    if (tap) {
      if (Math.hypot(pt.x - tap.x, pt.y - tap.y) < 8) return; // still a tap
      prev = tap;                                             // a drag: move from where the finger started
      tap = null;
    }
    if (pointers.size >= 2 && pinch) {
      const [a, b] = [...pointers.values()];
      view.zoom = Math.min(maxZoom(), Math.max(1, pinch.zoom * Math.hypot(a.x - b.x, a.y - b.y) / pinch.dist));
      anchor(pinch.anchor, (a.x + b.x) / 2, (a.y + b.y) / 2);
    } else if (pointers.size === 1) {
      const { k } = frameBox();
      view.cx -= (pt.x - prev.x) / k;
      view.cy -= (pt.y - prev.y) / k;
    }
    viewChanged();
  });

  function endPointer(e) {
    if (!pointers.has(e.pointerId)) return;
    pointers.delete(e.pointerId);
    if (tap && e.type === 'pointerup' && performance.now() - tap.t < 500) {
      const p = toPhoto(tap.x, tap.y); // tap: bring that spot to the middle of the square
      view.cx = p.x;
      view.cy = p.y;
      viewChanged(true);
    }
    tap = null;
    if (pointers.size >= 2) startPinch();
    else pinch = null;
    if (pointers.size === 0) {
      el.stage.classList.remove('grabbing');
      settle();
    }
  }
  el.stage.addEventListener('pointerup', endPointer);
  el.stage.addEventListener('pointercancel', endPointer);

  let wheelTimer = 0;
  el.stage.addEventListener('wheel', (e) => {
    if (!source) return;
    e.preventDefault();
    clearTimeout(autoTimer);
    el.preview.classList.remove('glide');
    const pt = localPoint(e);
    const a = toPhoto(pt.x, pt.y);
    const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
    view.zoom = Math.min(maxZoom(), Math.max(1, view.zoom * Math.exp(-dy * 0.002)));
    anchor(a, pt.x, pt.y);
    viewChanged();
    clearTimeout(wheelTimer);
    wheelTimer = setTimeout(settle, 250);
  }, { passive: false });

  el.stage.addEventListener('keydown', (e) => {
    if (!source) {
      if ((e.key === 'Enter' || e.key === ' ') && session) { e.preventDefault(); el.fileInput.click(); }
      return;
    }
    const step = baseSide() / view.zoom * 0.1;
    switch (e.key) {
      case 'ArrowLeft': view.cx -= step; break;
      case 'ArrowRight': view.cx += step; break;
      case 'ArrowUp': view.cy -= step; break;
      case 'ArrowDown': view.cy += step; break;
      case '+': case '=': view.zoom *= ZOOM_STEP; break;
      case '-': case '_': view.zoom /= ZOOM_STEP; break;
      case '0': resetView(); break;
      case 'Enter': e.preventDefault(); analyze(); return;
      default: return;
    }
    e.preventDefault();
    viewChanged(true);
    settle();
  });

  function zoomBy(f) {
    if (!source) return;
    view.zoom *= f;
    viewChanged(true);
    settle();
  }
  el.zoomInBtn.addEventListener('click', () => zoomBy(ZOOM_STEP));
  el.zoomOutBtn.addEventListener('click', () => zoomBy(1 / ZOOM_STEP));
  el.resetBtn.addEventListener('click', () => {
    if (!source) return;
    resetView();
    viewChanged(true);
    settle();
  });

  if (window.ResizeObserver) new ResizeObserver(() => scheduleRender()).observe(el.stage);
  else window.addEventListener('resize', () => scheduleRender());

  // ---------- Preprocessing ----------

  // Downscale in halving steps so large phone photos are smoothed like PIL's antialiased resize.
  function cropTo224(v) {
    const r = cropRect(v);
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

  // Sharpness = variance of the 4-neighbour Laplacian of the luma; brightness = mean luma (0..255).
  // Same measures as training/calibrate_app.py.
  function checkQuality(canvas) {
    const { data } = canvas.getContext('2d').getImageData(0, 0, SIZE, SIZE);
    const y = new Float32Array(SIZE * SIZE);
    let sum = 0;
    for (let i = 0, p = 0; p < y.length; i += 4, p++) {
      y[p] = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
      sum += y[p];
    }
    let n = 0, m = 0, m2 = 0;
    for (let r = 1; r < SIZE - 1; r++) {
      for (let c = 1, p = r * SIZE + 1; c < SIZE - 1; c++, p++) {
        const lap = y[p - SIZE] + y[p + SIZE] + y[p - 1] + y[p + 1] - 4 * y[p];
        n++; m += lap; m2 += lap * lap;
      }
    }
    const sharpness = m2 / n - (m / n) ** 2;
    const brightness = sum / y.length;
    return { sharpness, brightness, blurry: sharpness < BLURRY, dark: brightness < DARK };
  }

  // ---------- Checking ----------

  function setChecking(on) {
    el.checkBtn.disabled = on;
    el.checkBtn.textContent = on ? 'Checking…' : 'Check this leaf';
  }

  async function analyze() {
    clearTimeout(autoTimer);
    if (!session || !source) return;
    if (busy) { again = true; return; }
    busy = true;
    again = false;
    const v = { cx: view.cx, cy: view.cy, zoom: view.zoom, id: source.id };
    setChecking(true);
    try {
      const canvas = cropTo224(v);
      const quality = checkQuality(canvas);
      const t0 = performance.now();
      const output = await session.run({ image: toTensor(canvas) });
      const ms = performance.now() - t0;
      if (!source || source.id !== v.id) return; // a new photo replaced this one meanwhile
      const probs = Float32Array.from(output.probabilities.data);
      const leaf = leaves.find((l) => l.id === v.id);
      const thumb = thumbnail(canvas, 80);
      if (leaf) Object.assign(leaf, { probs, thumb, quality });
      else leaves.push({ id: v.id, probs, thumb, quality });
      checked = v;
      const ranked = rank(combine());
      showResult(ranked, ms, quality);
      saveHistory(thumbnail(canvas, 88), ranked[0]);
    } catch (err) {
      console.error(err);
      setStatus('Something went wrong while checking. Please try another photo.');
    } finally {
      busy = false;
      setChecking(false);
      updateStale();
      if (again && isStale()) analyze();
    }
  }

  function combine() {
    if (leaves.length === 1) return leaves[0].probs;
    const avg = new Float32Array(leaves[0].probs.length);
    for (const l of leaves) l.probs.forEach((p, i) => { avg[i] += p / leaves.length; });
    return avg;
  }

  function rank(probs) {
    return Array.from(probs, (p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
  }

  function thumbnail(canvas, side) {
    const t = document.createElement('canvas');
    t.width = t.height = side;
    t.getContext('2d').drawImage(canvas, 0, 0, side, side);
    return t.toDataURL('image/jpeg', 0.7);
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
      name: e.healthy ? 'Healthy' : (r ? r.name : (labels[i]?.condition || '').replace(/[_-]/g, ' ')),
      type: e.healthy ? 'Healthy' : (r ? r.type : 'Unknown'), record: r,
    };
  }

  function kindOf(d) {
    if (d.healthy) return 'healthy';
    if (d.type.startsWith('Pest')) return 'pest';
    return 'disease';
  }

  function bandOf(p) {
    if (p >= CONFIDENT) return 'confident';
    if (p >= UNSURE) return 'moderate';
    return 'unsure';
  }

  function pct(p) { return `${(p * 100).toFixed(p >= 0.995 ? 0 : 1)}%`; }
  function pct0(p) { return `${Math.round(p * 100)}%`; }

  function italic(text) {
    const i = document.createElement('i');
    i.textContent = text;
    return i;
  }

  function make(tag, className, text) {
    const n = document.createElement(tag);
    if (className) n.className = className;
    if (text != null) n.textContent = text;
    return n;
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
      const type = r.type.toLowerCase();
      parts.push(type.includes('(') ? `, ${type}` : ` (${type})`);
    } else if (r) {
      parts.push(` · ${r.type}`);
    }
    return parts;
  }

  // "Early blight", or "early blight on potato" when the other candidate is on a different plant.
  function shortName(d, other) {
    const name = d.healthy ? 'healthy' : d.name.toLowerCase();
    return other && other.host !== d.host ? `${name} (${d.plant.toLowerCase()})` : name;
  }

  const CHIP = {
    confident: { healthy: 'No disease seen', disease: 'Disease found', pest: 'Pest found' },
    moderate: { healthy: 'Probably healthy', disease: 'Possible disease', pest: 'Possible pest' },
  };
  const CONF_WORD = { confident: 'High confidence', moderate: 'Medium confidence', unsure: 'Low confidence' };

  function showResult(ranked, ms, quality) {
    const top = ranked[0];
    const d = describe(top.i);
    const band = bandOf(top.p);
    const kind = kindOf(d);
    const unsure = band === 'unsure';

    el.summaryCard.hidden = false;
    el.detailsCard.hidden = false;
    el.chip.className = `chip ${unsure ? 'unsure' : kind}`;
    el.chip.textContent = unsure ? 'Not sure' : CHIP[band][kind];
    el.basis.textContent = leaves.length > 1
      ? `${leaves.length} leaves combined`
      : (ms != null ? `Checked in ${Math.round(ms)} ms` : '');

    const plantLabel = d.plantFil ? `${d.plant} · ${d.plantFil}` : d.plant;
    el.plant.textContent = unsure ? `Best guess: ${plantLabel}` : plantLabel;
    el.name.textContent = unsure ? `${d.name}?` : d.name;
    el.sci.replaceChildren(...sciLine(d));
    el.meter.className = `meter ${unsure ? 'unsure' : kind}`;
    el.bar.style.transform = `scaleX(${top.p.toFixed(3)})`;
    el.meter.setAttribute('aria-valuenow', String(Math.round(top.p * 100)));
    el.conf.textContent = pct0(top.p);
    el.confWord.textContent = CONF_WORD[band];

    // What changed since the last check of this plant, so the effect of moving the square is visible.
    el.delta.className = 'delta';
    el.delta.textContent = '';
    if (shown) {
      if (shown.i !== top.i) {
        el.delta.classList.add('changed');
        el.delta.textContent = `was ${shortName(describe(shown.i))} ${pct0(shown.p)}`;
      } else {
        const diff = Math.round(top.p * 100) - Math.round(shown.p * 100);
        el.delta.textContent = diff === 0 ? 'no change' : `${diff > 0 ? '▲' : '▼'} ${Math.abs(diff)} since last check`;
        if (diff) el.delta.classList.add(diff > 0 ? 'up' : 'down');
      }
    }
    shown = { i: top.i, p: top.p };

    renderMessage(band, d, ranked, quality);
    el.details.hidden = unsure;
    el.sourcesBox.hidden = unsure;
    if (!unsure) renderDetails(d, band);
    renderConfirm(ranked, top.p);
    renderOthers(ranked);
    renderLeaves();

    el.addLeafBtn.hidden = leaves.length >= MAX_LEAVES;
    el.addLeafBtn.classList.toggle('suggest', band !== 'confident');
  }

  function renderMessage(band, d, ranked, quality) {
    const parts = [];
    if (quality && (quality.blurry || quality.dark)) {
      parts.push(make('p', 'quality', quality.blurry
        ? 'The photo looks blurry. Hold the phone still, tap the leaf on the screen to focus, and retake it.'
        : 'The photo looks too dark. Retake it in shade or near a window, without dark shadows.'));
    }
    let text;
    if (band === 'confident') {
      text = d.healthy
        ? 'No disease seen in this photo. Early signs can be missed, so check the plant again in 3 to 5 days.'
        : `On farm photos, results this sure were right about ${RIGHT_IN_10.confident} in 10 times. Compare the signs below before you act.`;
    } else if (band === 'moderate') {
      const alt = ranked[1] ? describe(ranked[1].i) : null;
      text = `Results this sure were right about ${RIGHT_IN_10.moderate} in 10 times.` +
        (alt ? ` It could also be ${shortName(alt, d)}.` : '') +
        ' Answer the questions below, or add another leaf.';
    } else {
      text = `LeafScan is not sure: guesses like this were right only about ${RIGHT_IN_10.unsure} in 10 times. ` +
        'This does not mean the plant is healthy. Move the square onto one leaf, retake it in even light, or add another leaf.';
    }
    parts.push(make('p', null, text));
    el.message.className = `message ${band}`;
    el.message.replaceChildren(...parts);
  }

  // Chemical control lines from the knowledge base (fungicides, sprays, copper, ...).
  const CHEMICAL = /\b(fungicides?|insecticides?|miticides?|acaricides?|bactericides?|nematicides?|pesticides?|copper|sprays?|spraying)\b/i;

  function renderDetails(d, band) {
    const r = d.record;
    el.chemNote.hidden = true;
    if (d.healthy || !r) {
      el.symptomsTitle.textContent = 'What this means';
      el.symptoms.textContent = d.healthy
        ? `No disease was seen on this ${d.plant.toLowerCase()} leaf. Early signs can be missed, so keep checking.`
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
    // Free, non-chemical steps first; chemical options last, and only when the result is confident.
    el.manageTitle.textContent = 'What you can do';
    const plain = r.management.filter((m) => !CHEMICAL.test(m));
    const chem = r.management.filter((m) => CHEMICAL.test(m));
    fillList(el.manage, band === 'confident' ? [...plain, ...chem] : plain);
    if (chem.length) {
      el.chemNote.hidden = false;
      el.chemNote.textContent = band === 'confident'
        ? 'Before using any pesticide, check the diagnosis with an agricultural technician or a plant clinic. Use only FPA-registered products and follow the label dose and the waiting time before harvest.'
        : 'Chemical options are hidden until the diagnosis is confirmed. Get a second opinion (see "Still not sure" below) before using any pesticide.';
    }
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
      li.append(make('div', 'src-meta', [s.publisher, s.kind, s.year, s.license].filter(Boolean).join(' · ')));
      return li;
    }));
    const notes = [r.review_note, r.name_note && `Name: ${r.name_note}`, kb.note].filter(Boolean);
    el.note.textContent = notes.join(' ');
  }

  function renderOthers(ranked) {
    el.others.replaceChildren(...ranked.slice(1, 4).map(({ p, i }) => {
      const o = describe(i);
      const li = document.createElement('li');
      const bar = make('div', 'o-bar');
      const fill = document.createElement('span');
      fill.style.transform = `scaleX(${p.toFixed(3)})`;
      bar.append(fill);
      li.append(make('span', null, `${o.plant}: ${o.name}`), make('span', 'o-pct', pct(p)), bar);
      return li;
    }));
  }

  function renderLeaves() {
    el.leafStrip.hidden = leaves.length < 2;
    if (leaves.length < 2) { el.leafStrip.replaceChildren(); return; }
    el.leafStrip.replaceChildren(...leaves.map((l, n) => {
      const top = rank(l.probs)[0];
      const d = describe(top.i);
      const li = document.createElement('li');
      li.classList.toggle('current', l.id === source.id);
      const img = document.createElement('img');
      img.src = l.thumb; img.alt = '';
      const text = make('div', 'l-text');
      text.append(make('b', null, `Leaf ${n + 1}${l.id === source.id ? ' (shown)' : ''}`),
        make('span', null, `${d.healthy ? 'Healthy' : d.name} ${pct0(top.p)}`));
      li.append(img, text);
      if (l.id !== source.id) {
        const rm = make('button', null, '×');
        rm.type = 'button';
        rm.setAttribute('aria-label', `Remove leaf ${n + 1}`);
        rm.addEventListener('click', () => {
          leaves = leaves.filter((x) => x !== l);
          const cur = leaves.find((x) => x.id === source.id);
          showResult(rank(combine()), null, cur && cur.quality);
          updateStale();
        });
        li.append(rm);
      }
      return li;
    }));
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
        const label = make('span', 'confirm-q', `${q} `);
        label.append(make('small', null, `(sign of ${c.d.name.toLowerCase()})`));
        const seg = make('div', 'seg');
        seg.setAttribute('role', 'group');
        seg.setAttribute('aria-label', q);
        for (const [val, text] of [['yes', 'Yes'], ['no', 'No'], ['skip', 'Not sure']]) {
          const b = make('button', null, text);
          b.type = 'button';
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
    renderDetails(best.d, 'moderate');
    el.details.hidden = false;
    el.sourcesBox.hidden = !best.d.record;
  }

  function fillList(ul, items) {
    ul.replaceChildren(...items.map((t) => make('li', null, t)));
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
      li.append(make('span', null, n ? `· ${n} ${n === 1 ? 'problem' : 'problems'}` : '· healthy only'));
      return li;
    }));
  }

  // ---------- History (per-device convenience only) ----------

  function readHistory() {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; }
  }

  function saveHistory(img, top) {
    try {
      const items = readHistory();
      const entry = { t: Date.now(), id: labels[top.i].id, p: top.p, sid: plantId, n: leaves.length, img };
      // Re-checks and extra leaves of the same plant replace its entry instead of stacking duplicates.
      if (items[0] && items[0].sid === plantId) items.shift();
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
      const text = make('div', 'h-text');
      text.append(
        make('div', 'h-name', h.p < UNSURE ? `Not sure (maybe ${d.plant}: ${d.name})` : `${d.plant}: ${d.name}`),
        make('div', 'h-meta', [pct(h.p), h.n > 1 ? `${h.n} leaves` : null,
          new Date(h.t).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })].filter(Boolean).join(' · ')),
      );
      li.append(img, text);
      return li;
    }));
  }

  el.clearHistory.addEventListener('click', () => {
    try { localStorage.removeItem(HISTORY_KEY); } catch { /* ignore */ }
    renderHistory();
  });

  // ---------- Adding leaves and starting over ----------

  function setAdding(on) {
    adding = on;
    el.addBanner.hidden = !on;
    el.actions.hidden = !on && !!source;
    if (on) {
      el.addText.replaceChildren(make('b', null, `Leaf ${leaves.length + 1} of up to ${MAX_LEAVES}:`),
        ' take or choose a photo of another leaf from the same plant. The results are combined.');
    }
  }

  el.addLeafBtn.addEventListener('click', () => {
    setAdding(true);
    el.cameraBtn.focus({ preventScroll: true });
  });
  el.addCancel.addEventListener('click', () => setAdding(false));

  el.checkBtn.addEventListener('click', () => analyze());
  el.autoCheck.addEventListener('change', () => {
    try { localStorage.setItem(AUTO_KEY, el.autoCheck.checked ? '1' : '0'); } catch { /* ignore */ }
    if (el.autoCheck.checked && isStale()) analyze();
  });

  // The camera and gallery buttons under the photo start a new plant, unless "Add another leaf" is open.
  function newPhoto(input) {
    if (!adding && leaves.length > 1 &&
        !window.confirm(`Start over with a new plant? The ${leaves.length} leaves checked so far will be cleared.`)) return;
    input();
  }

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

  function takePhoto() {
    const phoneLike = window.matchMedia('(pointer: coarse)').matches;
    if (phoneLike || !navigator.mediaDevices?.getUserMedia || !window.isSecureContext) {
      el.cameraInput.click(); // native camera app on phones and tablets
    } else {
      openLiveCamera();
    }
  }
  const pickPhoto = () => el.fileInput.click();

  el.cameraBtn.addEventListener('click', takePhoto);
  el.uploadBtn.addEventListener('click', pickPhoto);
  el.retakeBtn.addEventListener('click', () => newPhoto(takePhoto));
  el.pickBtn.addEventListener('click', () => newPhoto(pickPhoto));

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

  for (const input of [el.fileInput, el.cameraInput]) {
    input.addEventListener('change', () => {
      if (input.files[0]) loadFile(input.files[0]);
      input.value = '';
    });
  }

  el.stage.addEventListener('dragover', (e) => { e.preventDefault(); el.stage.classList.add('dragging'); });
  el.stage.addEventListener('dragleave', () => el.stage.classList.remove('dragging'));
  el.stage.addEventListener('drop', (e) => {
    e.preventDefault();
    el.stage.classList.remove('dragging');
    if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
  });
  el.stage.addEventListener('click', () => {
    if (source || el.cameraBtn.disabled) return;
    if (window.matchMedia('(pointer: coarse)').matches) el.cameraInput.click();
    else el.fileInput.click();
  });
  document.addEventListener('paste', (e) => {
    const file = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith('image/'));
    if (file) loadFile(file);
  });

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

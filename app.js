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
 * Results are updated in place and the page never scrolls by itself. The model runs in a Web Worker
 * (worker.js), with the page itself as a fallback.
 */
(() => {
  'use strict';

  const LABELS_URL = 'model/labels.json';
  const KB_URL = 'data/kb.json';
  const SIZE = 224;
  const RESIZE = 256;
  const CONFIDENT = 0.70;   // at or above: show result normally
  const UNSURE = 0.45;      // below: say we are not sure
  // Cut-offs are chosen on the 2,209 validation field photos; every rate is reported on the 3,100 test field
  // photos (public datasets, not Philippine farms), neither used in training. training/calibrate_app.py ->
  // results/app_calibration.json, for the app's model v2-int8 (the paper's v1-study figures are in results/v1-study).
  // Out of 10 answers in each band, how many were right: 92.3% [90.9-93.6], 70.2% [66.6-73.5], 40.1% [37.0-43.3].
  const RIGHT_IN_10 = { confident: 9, moderate: 7, unsure: 4 };
  // The same when the user chooses the plant and only that plant's problems are ranked: 94.1%, 58.7%, 29.6%.
  // Choosing the plant raises overall accuracy on these photos from 71.9% to 87.7%.
  const RIGHT_IN_10_PLANT = { confident: 9, moderate: 6, unsure: 3 };
  // Below this top probability the guess was right only 28.9% of the time (about 3 in 10), and the plant 41.1%.
  const NOT_RECOGNIZED = 0.25;
  const OTHER_PLANT = 'other';
  const MAX_QUESTIONS = 4;
  // Photo checks on the 224x224 crop. Blur: at least 95% of artificially blurred crops score below 18, and
  // only 2% of real ones. Darkness barely hurts this model, so it warns only on very dark crops, where a
  // person could not check the spots either.
  const BLURRY = 18;
  const DARK = 30;
  const ZOOM_STEP = 1.25;
  const MAX_LEAVES = 3;
  const AUTO_DELAY = 350;   // ms after the user lets go, when automatic checking is on
  const HISTORY_KEY = 'leafscan.history.v1';
  const AUTO_KEY = 'leafscan.autocheck.v1';
  const FEEDBACK_KEY = 'leafscan.feedback.v1';
  const HISTORY_MAX = 8;
  const FEEDBACK_MAX = 500;

  const $ = (id) => document.getElementById(id);
  const el = {
    stage: $('stage'), empty: $('emptyState'), emptySub: $('emptySub'), preview: $('previewCanvas'), frame: $('frame'),
    tools: $('tools'), zoomOutBtn: $('zoomOutBtn'), zoomInBtn: $('zoomInBtn'), zoomLevel: $('zoomLevel'),
    resetBtn: $('resetBtn'), retakeBtn: $('retakeBtn'), pickBtn: $('pickBtn'), hint: $('hint'),
    addBanner: $('addBanner'), addText: $('addText'), addCancel: $('addCancel'),
    actions: $('actions'), cameraBtn: $('cameraBtn'), uploadBtn: $('uploadBtn'),
    cameraInput: $('cameraInput'), fileInput: $('fileInput'), status: $('status'), progress: $('progress'),
    summaryCard: $('summaryCard'), summaryBody: $('summaryBody'), chip: $('resultChip'), basis: $('resultBasis'),
    plantSelect: $('plantSelect'),
    plant: $('resultPlant'), name: $('resultName'), sci: $('resultSci'), meter: $('resultMeter'), bar: $('resultBar'),
    conf: $('resultConf'), confWord: $('confWord'), delta: $('resultDelta'), message: $('resultMessage'),
    staleBar: $('staleBar'), checkBtn: $('checkBtn'), leafStrip: $('leafStrip'), addLeafBtn: $('addLeafBtn'),
    techBtn: $('techBtn'), autoCheck: $('autoCheck'),
    detailsCard: $('detailsCard'), details: $('resultDetails'), symptomsTitle: $('symptomsTitle'), symptoms: $('resultSymptoms'),
    lookTitle: $('lookTitle'), look: $('resultLook'), manageTitle: $('manageTitle'), manage: $('resultManage'),
    chemNote: $('chemNote'), sourcesBox: $('sourcesBox'), sources: $('resultSources'), note: $('resultNote'),
    othersTitle: $('othersTitle'), others: $('resultOthers'),
    confirmBox: $('confirmBox'), confirmCount: $('confirmCount'), confirmStep: $('confirmStep'), confirmQ: $('confirmQ'),
    confirmWhy: $('confirmWhy'), confirmBack: $('confirmBack'), confirmDone: $('confirmDone'),
    confirmResult: $('confirmResult'), confirmRestart: $('confirmRestart'),
    feedbackSeg: $('feedbackSeg'), unlistedBtn: $('unlistedBtn'), feedbackNote: $('feedbackNote'),
    techDialog: $('techDialog'), techBody: $('techBody'), techShare: $('techShare'), techClose: $('techClose'),
    techNote: $('techNote'),
    historyCard: $('historyCard'), historyList: $('historyList'), clearHistory: $('clearHistory'), tally: $('tally'),
    cropList: $('cropList'), installBtn: $('installBtn'), offlineNote: $('offlineNote'), retryBtn: $('retryBtn'),
    studyBox: $('studyBox'), studyForm: $('studyForm'), studySave: $('studySave'), studyNote: $('studyNote'),
    studyLog: $('studyLog'), studyCount: $('studyCount'), studyExport: $('studyExport'), studyList: $('studyList'),
    studyExportNote: $('studyExportNote'), studyClear: $('studyClear'), studyToggle: $('studyToggle'),
    brandBtn: $('brandBtn'), navrow: $('navrow'), tabs: [...document.querySelectorAll('#navrow [data-view]')],
    intro: $('intro'), introFacts: $('introFacts'), historyCount: $('historyCount'), historyEmpty: $('historyEmpty'),
    plantFilter: $('plantFilter'), plantsLede: $('plantsLede'), plantsEmpty: $('plantsEmpty'),
    dialog: $('cameraDialog'), video: $('cameraVideo'), shutter: $('cameraShutter'),
    cameraCancel: $('cameraCancel'), cameraSwitch: $('cameraSwitch'),
  };

  let runner = null;        // runs the model: { run(Float32Array) -> Promise<{ probs, ms }>, where }
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
  let plantLock = null;     // plant chosen by the user (kept for later photos), or OTHER_PLANT
  let current = null;       // the result on screen: { ranked, top, d, outcome, band }
  let lastMs = null;        // time of the last model run
  let modelVersion = '';    // from model/model.json, recorded with study-log entries
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

  // The model is downloaded in 2 MB parts (training/split_model.py). Each part is checked and stored as it
  // arrives in a cache named after the model version, which app updates never delete: a dropped connection
  // costs at most one part, and updating the app costs no model download at all. model.json gives the exact
  // size, so the progress bar is true (the server's gzip Content-Length is not).
  const MODEL_INFO_URL = 'model/model.json';
  const MODEL_CACHE_PREFIX = 'leafscan-model-';

  async function sha256hex(buf) {
    const hash = await crypto.subtle.digest('SHA-256', buf);
    return Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, '0')).join('');
  }

  async function partIsIntact(buf, part) {
    if (buf.length !== part.bytes) return false;
    return !(self.crypto && crypto.subtle) || (await sha256hex(buf)) === part.sha256;
  }

  // One part, retried with a growing pause, so a short drop in the connection does not stop the download.
  async function fetchPart(url, part, onBytes) {
    let lastErr = null;
    for (let attempt = 0; attempt < 4; attempt++) {
      if (attempt) await new Promise((r) => setTimeout(r, 1500 * attempt));
      try {
        const res = await fetch(url, { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
        let buf;
        if (res.body) {
          buf = new Uint8Array(part.bytes);
          let got = 0;
          const reader = res.body.getReader();
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            if (got + value.length > buf.length) throw new Error(`${url} is larger than expected`);
            buf.set(value, got);
            got += value.length;
            onBytes(got);
          }
          if (got !== part.bytes) throw new Error(`${url} ended early`);
        } else {
          buf = new Uint8Array(await res.arrayBuffer());
        }
        if (await partIsIntact(buf, part)) return buf;
        throw new Error(`${url} is damaged`);
      } catch (err) {
        lastErr = err;
      }
    }
    throw lastErr;
  }

  // The ONNX Runtime core (about 14 MB) is fetched once by the page and stored in its own cache, then handed
  // to the worker, so it is never downloaded twice (the browser's HTTP cache may not keep a file this big).
  async function loadRuntimeBytes(info, onProgress) {
    const rt = info.runtime;
    let cache = null;
    try { if (self.caches) cache = await caches.open(rt.cache); } catch { cache = null; }
    if (cache) {
      const hit = await cache.match(rt.file);
      if (hit) {
        const buf = new Uint8Array(await hit.arrayBuffer());
        if (await partIsIntact(buf, rt)) return { wasm: buf, cached: true };
        await cache.delete(rt.file);
      }
    }
    const wasm = await fetchPart(rt.file, rt, onProgress);
    if (cache) {
      try { await cache.put(rt.file, new Response(wasm, { headers: { 'Content-Type': 'application/wasm' } })); } catch { /* storage full */ }
    }
    return { wasm, cached: !!cache };
  }

  async function loadModelBytes(info, onProgress) {
    const cacheName = MODEL_CACHE_PREFIX + info.version;
    let cache = null;
    try { if (self.caches) cache = await caches.open(cacheName); } catch { cache = null; }
    const bytes = new Uint8Array(info.bytes);
    let offset = 0;
    onProgress(0);
    for (const part of info.parts) {
      const url = `model/${info.version}/${part.file}`;
      let buf = null;
      if (cache) {
        const hit = await cache.match(url);
        if (hit) buf = new Uint8Array(await hit.arrayBuffer());
        if (buf && !(await partIsIntact(buf, part))) { await cache.delete(url); buf = null; }
      }
      if (!buf) {
        buf = await fetchPart(url, part, (n) => onProgress(offset + n));
        if (cache) {
          try { await cache.put(url, new Response(buf, { headers: { 'Content-Type': 'application/octet-stream' } })); } catch { /* storage full: carry on without saving */ }
        }
      }
      bytes.set(buf, offset);
      offset += buf.length;
      onProgress(offset);
    }
    // Parts of older model versions are no longer needed.
    if (self.caches) {
      for (const key of await caches.keys()) {
        if (key.startsWith(MODEL_CACHE_PREFIX) && key !== cacheName) caches.delete(key);
      }
    }
    return { bytes, cached: !!cache };
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
      renderPlantSelect();
      renderHistory();

    } catch (err) {
      console.error(err);
      setStatus('Could not load LeafScan. Check your connection and reload the page.');
      return;
    }
    await startModel();
  }

  async function startModel() {
    el.retryBtn.hidden = true;
    el.progress.classList.remove('done');
    let have = 0, total = 0;
    try {
      setStatus('Getting the AI model ready…');
      const info = await (await fetch(MODEL_INFO_URL, { cache: 'no-cache' })).json();
      total = info.bytes + info.runtime.bytes;
      modelVersion = info.version;
      const progress = (done) => {
        have = done;
        setProgress(done / total);
        if (done < total) setStatus(`Downloading the AI model (first time only): ${Math.floor((done / total) * 100)}%`);
      };
      let { wasm, cached: wasmCached } = await loadRuntimeBytes(info, progress);
      let { bytes, cached } = await loadModelBytes(info, (n) => progress(info.runtime.bytes + n));
      setStatus('Starting the AI model…');
      try {
        // Hand both to the worker without copying; if the worker fails, read them again from the cache.
        runner = await workerRunner(bytes, wasm, cached && wasmCached);
      } catch (err) {
        console.warn('Model worker unavailable, running the model on the page instead', err);
        if (cached && wasmCached) {
          ({ wasm } = await loadRuntimeBytes(info, () => {}));
          ({ bytes } = await loadModelBytes(info, () => {}));
        }
        runner = await pageRunner(bytes, wasm);
      }
      el.progress.classList.add('done');
      el.cameraBtn.disabled = false;
      el.uploadBtn.disabled = false;
      setStatus(source ? '' : 'Ready. Take a photo of one leaf, or choose one from your gallery.');
      if (source) analyze();
      registerOffline();
    } catch (err) {
      console.error(err);
      setStatus(total
        ? `The download stopped at ${Math.floor((have / total) * 100)}%. What arrived is saved: connect and tap Try again to continue.`
        : 'Could not reach LeafScan to download the AI model. Connect and tap Try again.');
      el.retryBtn.hidden = false;
    }
  }

  // The model runs in a Web Worker (worker.js) so a slow check never blocks dragging and pinching.
  function workerRunner(bytes, wasm, transfer) {
    return new Promise((resolve, reject) => {
      let worker;
      try { worker = new Worker('worker.js'); } catch (err) { reject(err); return; }
      const pending = new Map();
      let seq = 0;
      const fail = (err) => { clearTimeout(timer); worker.terminate(); reject(err); };
      const timer = setTimeout(() => fail(new Error('the model worker did not start in time')), 90000);
      worker.onerror = (e) => { e.preventDefault(); fail(new Error(e.message || 'model worker error')); };
      worker.onmessage = (e) => {
        const msg = e.data;
        if (msg.type === 'ready') {
          clearTimeout(timer);
          worker.onerror = (ev) => console.error('Model worker error', ev.message);
          resolve({
            where: 'worker',
            run(data) {
              return new Promise((res, rej) => {
                const id = ++seq;
                pending.set(id, { res, rej });
                worker.postMessage({ type: 'run', id, data }, [data.buffer]);
              });
            },
          });
        } else if (msg.type === 'failed') {
          fail(new Error(msg.message));
        } else {
          const p = pending.get(msg.id);
          pending.delete(msg.id);
          if (!p) return;
          if (msg.type === 'result') p.res({ probs: msg.probs, ms: msg.ms });
          else p.rej(new Error(msg.message));
        }
      };
      // Transferred (not copied) when the parts are cached, so the model sits in memory only once.
      worker.postMessage({ type: 'load', bytes, wasm }, transfer ? [bytes.buffer, wasm.buffer] : []);
    });
  }

  async function pageRunner(bytes, wasm) {
    ort.env.wasm.wasmBinary = wasm;
    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    const tensor = (data) => new ort.Tensor('float32', data, [1, 3, SIZE, SIZE]);
    await session.run({ image: tensor(new Float32Array(3 * SIZE * SIZE)) }); // warm-up
    return {
      where: 'page',
      async run(data) {
        const t0 = performance.now();
        const out = await session.run({ image: tensor(data) });
        return { probs: Float32Array.from(out.probabilities.data), ms: performance.now() - t0 };
      },
    };
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
    el.intro.hidden = true;
    showView('scan');
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
    setStatus(runner ? '' : 'The photo is ready. Waiting for the AI model to finish loading…');
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
      if ((e.key === 'Enter' || e.key === ' ') && runner) { e.preventDefault(); el.fileInput.click(); }
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

  function toCHW(canvas) {
    const { data } = canvas.getContext('2d').getImageData(0, 0, SIZE, SIZE);
    const plane = SIZE * SIZE;
    const chw = new Float32Array(3 * plane);
    for (let i = 0, p = 0; p < plane; i += 4, p++) {
      chw[p] = data[i] / 255;
      chw[p + plane] = data[i + 1] / 255;
      chw[p + 2 * plane] = data[i + 2] / 255;
    }
    return chw;
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
    if (!runner || !source) return;
    if (busy) { again = true; return; }
    busy = true;
    again = false;
    const v = { cx: view.cx, cy: view.cy, zoom: view.zoom, id: source.id };
    setChecking(true);
    try {
      const canvas = cropTo224(v);
      const quality = checkQuality(canvas);
      const { probs, ms } = await runner.run(toCHW(canvas));
      if (!source || source.id !== v.id) return; // a new photo replaced this one meanwhile
      lastMs = ms;
      const leaf = leaves.find((l) => l.id === v.id);
      const shots = { thumb: thumbnail(canvas, 88), crop: canvas.toDataURL('image/jpeg', 0.85) };
      if (leaf) Object.assign(leaf, { probs, quality }, shots);
      else leaves.push({ id: v.id, probs, quality, ...shots });
      checked = v;
      showResult();
      saveHistory();
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

  // All classes ranked, and the ranking shown: when the user chose the plant, only that plant's classes,
  // renormalized so they add up to 100%.
  function ranking() {
    const all = rank(combine());
    if (!plantLock || plantLock === OTHER_PLANT) return { ranked: all, all };
    const mine = all.filter(({ i }) => entries[i].host === plantLock);
    const total = mine.reduce((s, x) => s + x.p, 0) || 1;
    return { ranked: mine.map(({ p, i }) => ({ p: p / total, i })), all };
  }

  const hostHasProblems = (host) => entries.some((e) => e.host === host && !e.healthy);
  const plantName = (host) => kb.hosts[host]?.en || host;

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

  // Outcomes: confident | moderate | unsure (by top probability), unknown (very low, plant not chosen),
  // uncovered (the user says the plant is not in LeafScan's list), healthy-only (the chosen plant has no
  // disease classes, only healthy leaves).
  function showResult() {
    if (!leaves.length) return;
    const { ranked, all } = ranking();
    const top = ranked[0];
    const d = describe(top.i);
    const kind = kindOf(d);
    const band = bandOf(top.p);
    let outcome = band;
    if (plantLock === OTHER_PLANT) outcome = 'uncovered';
    else if (plantLock && !hostHasProblems(plantLock)) outcome = 'healthy-only';
    else if (!plantLock && top.p < NOT_RECOGNIZED) outcome = 'unknown';
    const special = outcome === 'uncovered' || outcome === 'healthy-only';
    const doubtful = special || outcome === 'unsure' || outcome === 'unknown';
    current = { ranked, all, top, d, band, outcome };

    el.summaryCard.hidden = false;
    el.detailsCard.hidden = false;
    renderPlantSelect();
    el.chip.className = `chip ${doubtful ? 'unsure' : kind}`;
    el.summaryCard.dataset.kind = doubtful ? 'unsure' : kind;
    el.chip.textContent = {
      uncovered: 'Not covered', 'healthy-only': 'Not covered', unknown: 'Not recognized', unsure: 'Not sure',
    }[outcome] || CHIP[band][kind];

    const plantLabel = d.plantFil ? `${d.plant} · ${d.plantFil}` : d.plant;
    if (outcome === 'uncovered') {
      el.plant.textContent = 'Your plant is not in LeafScan\'s list';
      el.name.textContent = 'LeafScan cannot check this plant';
      el.sci.replaceChildren();
    } else if (outcome === 'healthy-only') {
      el.plant.textContent = `${plantLabel} (your choice)`;
      el.name.textContent = `${d.plant} diseases are not covered yet`;
      el.sci.replaceChildren(...sciLine({ ...d, record: null }));
    } else if (outcome === 'unknown') {
      el.plant.textContent = `Closest match: ${plantLabel}`;
      el.name.textContent = 'Not recognized';
      el.sci.replaceChildren(`Closest guess: ${d.healthy ? 'a healthy leaf' : d.name.toLowerCase()}`);
    } else {
      el.plant.textContent = `${outcome === 'unsure' ? 'Best guess: ' : ''}${plantLabel}${plantLock ? ' (your choice)' : ''}`;
      el.name.textContent = outcome === 'unsure' ? `${d.name}?` : d.name;
      el.sci.replaceChildren(...sciLine(d));
    }

    el.meter.className = `meter ${doubtful ? 'unsure' : kind}`;
    el.bar.style.transform = `scaleX(${special ? 0 : top.p.toFixed(3)})`;
    el.meter.setAttribute('aria-valuenow', String(special ? 0 : Math.round(top.p * 100)));
    el.conf.textContent = special ? '–' : pct0(top.p);
    el.confWord.textContent = special ? 'No result' : (outcome === 'unknown' ? 'Very low confidence' : CONF_WORD[band]);
    const basis = [leaves.length > 1 ? `${leaves.length} leaves` : null,
      plantLock && !special ? `among ${d.plant.toLowerCase()} problems` : null].filter(Boolean).join(', ');
    el.basis.textContent = basis ? ` · ${basis}` : '';

    // What changed since the last check of this plant, so the effect of moving the square is visible.
    el.delta.className = 'delta';
    el.delta.textContent = '';
    if (shown && !special) {
      if (shown.i !== top.i) {
        el.delta.classList.add('changed');
        el.delta.textContent = `was ${shortName(describe(shown.i))} ${pct0(shown.p)}`;
      } else {
        const diff = Math.round(top.p * 100) - Math.round(shown.p * 100);
        el.delta.textContent = diff === 0 ? 'no change' : `${diff > 0 ? '▲' : '▼'} ${Math.abs(diff)} since last check`;
        if (diff) el.delta.classList.add(diff > 0 ? 'up' : 'down');
      }
    }
    if (!special) shown = { i: top.i, p: top.p };

    renderMessage(outcome, d, ranked, all);
    const detailed = outcome === 'confident' || outcome === 'moderate';
    el.details.hidden = !detailed;
    el.sourcesBox.hidden = !detailed;
    if (detailed) renderDetails(d, band);
    renderQuiz(ranked, outcome);
    renderOthers(ranked, outcome);
    renderLeaves();
    renderFeedbackState();

    el.addLeafBtn.hidden = leaves.length >= MAX_LEAVES || special;
    renderStudyBox();
    el.addLeafBtn.classList.toggle('suggest', outcome !== 'confident');
  }

  function renderMessage(outcome, d, ranked, all) {
    const parts = [];
    const q = leaves.find((l) => source && l.id === source.id)?.quality;
    if (q && (q.blurry || q.dark)) {
      parts.push(make('p', 'quality', q.blurry
        ? 'The photo looks blurry. Hold the phone still, tap the leaf on the screen to focus, and retake it.'
        : 'The photo looks too dark. Retake it in shade or near a window, without dark shadows.'));
    }
    // The chosen plant does not match what LeafScan clearly sees.
    const seen = all[0];
    if (plantLock && plantLock !== OTHER_PLANT && seen.p >= CONFIDENT && entries[seen.i].host !== plantLock) {
      parts.push(make('p', 'quality',
        `To LeafScan this looks more like ${plantName(entries[seen.i].host).toLowerCase()} (${pct0(seen.p)}). Is the plant choice right?`));
    }
    const rate = plantLock ? RIGHT_IN_10_PLANT : RIGHT_IN_10;
    const among = plantLock ? ` among ${d.plant.toLowerCase()} problems` : '';
    let text;
    if (outcome === 'uncovered') {
      text = 'LeafScan knows only the plants under "Supported plants". For this plant, ask a plant clinic or an agriculture teacher (see "Still not sure" below).';
    } else if (outcome === 'healthy-only') {
      text = `LeafScan has learned only healthy ${d.plant.toLowerCase()} leaves, so it cannot check them for diseases. Ask a plant clinic or an agriculture teacher (see "Still not sure" below).`;
    } else if (outcome === 'unknown') {
      text = 'This may not be a leaf, or it may be a plant or problem LeafScan does not know. When LeafScan is this unsure, its guess was right only about 3 in 10 times. Move the square onto one leaf, or choose the plant above if LeafScan knows it.';
    } else if (outcome === 'confident') {
      text = d.healthy
        ? 'No disease seen in this photo. Early signs can be missed, so check the plant again in 3 to 5 days.'
        : `On test photos, results this sure${among} were right about ${rate.confident} in 10 times. Compare the signs below before you act.`;
    } else if (outcome === 'moderate') {
      const alt = ranked[1] ? describe(ranked[1].i) : null;
      text = `Results this sure${among} were right about ${rate.moderate} in 10 times.` +
        (alt ? ` It could also be ${shortName(alt, d)}.` : '') +
        (plantLock ? ' Answer the questions below, or add another leaf.'
          : ' Answer the questions below, add another leaf, or choose the plant above if it is wrong.');
    } else {
      text = `LeafScan is not sure: guesses like this were right only about ${rate.unsure} in 10 times. ` +
        'This does not mean the plant is healthy. Move the square onto one leaf, retake it in even light, or add another leaf.' +
        (plantLock ? '' : ' If the plant is wrong, choose the right one above.');
    }
    parts.push(make('p', null, text));
    el.message.className = `message ${outcome === 'confident' || outcome === 'moderate' ? outcome : 'unsure'}`;
    el.message.replaceChildren(...parts);
  }

  // The plant picker: "Wrong plant?" until the user chooses one; the choice is kept for later photos.
  function renderPlantSelect() {
    const sel = el.plantSelect;
    if (!sel.options.length) {
      const hosts = [...new Set(entries.map((e) => e.host).filter(Boolean))]
        .sort((a, b) => plantName(a).localeCompare(plantName(b)));
      const opt = (value, text) => { const o = document.createElement('option'); o.value = value; o.textContent = text; return o; };
      const placeholder = opt('', 'Wrong plant?');
      placeholder.hidden = true;
      const group = document.createElement('optgroup');
      group.label = 'Plants LeafScan knows';
      group.append(...hosts.map((h) => {
        const info = kb.hosts[h] || {};
        return opt(h, `${plantName(h)}${info.fil ? ` (${info.fil})` : ''}${hostHasProblems(h) ? '' : ': healthy leaves only'}`);
      }));
      sel.append(placeholder, opt('auto', 'Let LeafScan guess the plant'), group, opt(OTHER_PLANT, 'My plant is not in this list'));
    }
    sel.value = plantLock || '';
    sel.querySelector('option[value="auto"]').hidden = !plantLock;
    sel.classList.toggle('locked', !!plantLock);
  }

  el.plantSelect.addEventListener('change', () => {
    const v = el.plantSelect.value;
    plantLock = v === 'auto' || v === '' ? null : v;
    clearTimeout(autoTimer);
    if (leaves.length) {
      showResult();
      saveHistory();
    } else {
      renderPlantSelect();
    }
  });

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

  function renderOthers(ranked, outcome) {
    const special = outcome === 'uncovered' || outcome === 'healthy-only';
    el.othersTitle.hidden = el.others.hidden = special;
    if (special) return;
    el.othersTitle.textContent = outcome === 'unknown' ? 'Closest matches' : 'Other possibilities';
    const list = outcome === 'unknown' ? ranked.slice(0, 3) : ranked.slice(1, 4);
    el.others.replaceChildren(...list.map(({ p, i }) => {
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
          showResult();
          saveHistory();
          updateStale();
        });
        li.append(rm);
      }
      return li;
    }));
  }

  // ---------- Symptom questions ----------
  // When the photo is not conclusive, ask one question at a time about visible signs listed in the expert
  // records of the top candidates, and re-rank them: each "yes" doubles a candidate's weight, each "no"
  // halves it, "not sure" changes nothing. At most MAX_QUESTIONS, alternating between the leading candidates,
  // stopping early once one candidate clearly leads.
  let quiz = null; // { pool, queue: [{ key, q, c }], pos, answers: Map, done }

  function renderQuiz(ranked, outcome) {
    // The pool keeps every likely candidate, including "healthy" (which has no questions), so that
    // answering "no" to all disease signs can still lead to a healthy result.
    const pool = ranked.slice(0, 3).filter(({ p }) => p >= 0.08).map(({ p, i }) => ({ p, i, d: describe(i) }));
    const cands = pool.filter((c) => c.d.record?.look_for?.length);
    if ((outcome !== 'moderate' && outcome !== 'unsure') || pool.length < 2 || !cands.length) {
      el.confirmBox.hidden = true;
      quiz = null;
      return;
    }
    const queue = [];
    for (let n = 0; queue.length < MAX_QUESTIONS && cands.some((c) => c.d.record.look_for[n]); n++) {
      for (const c of cands) {
        const q = c.d.record.look_for[n];
        if (q && queue.length < MAX_QUESTIONS) queue.push({ key: `${c.i}:${n}`, q, c });
      }
    }
    quiz = { pool, queue, pos: 0, answers: new Map(), done: false };
    el.confirmBox.hidden = false;
    showQuestion();
  }

  function showQuestion() {
    const { queue, pos, done } = quiz;
    el.confirmStep.hidden = done;
    el.confirmDone.hidden = !done;
    el.confirmCount.textContent = done ? '' : `Question ${pos + 1} of ${queue.length}`;
    if (done) return;
    const item = queue[pos];
    el.confirmQ.textContent = item.q;
    el.confirmWhy.textContent = `A sign of ${item.c.d.name.toLowerCase()}${item.c.d.host !== current.d.host ? ` on ${item.c.d.plant.toLowerCase()}` : ''}.`;
    el.confirmBack.hidden = pos === 0;
  }

  function quizScores() {
    const scored = quiz.pool.map((c) => {
      let w = c.p;
      for (const item of quiz.queue) {
        if (item.c.i !== c.i) continue;
        const a = quiz.answers.get(item.key);
        if (a === 'yes') w *= 2;
        if (a === 'no') w *= 0.5;
      }
      return { ...c, w };
    });
    const total = scored.reduce((sum, c) => sum + c.w, 0);
    return scored.map((c) => ({ ...c, share: c.w / total })).sort((a, b) => b.w - a.w);
  }

  function answerQuestion(a) {
    if (!quiz || quiz.done) return;
    quiz.answers.set(quiz.queue[quiz.pos].key, a);
    quiz.pos++;
    const scored = quizScores();
    const answered = [...quiz.answers.values()].filter((x) => x !== 'skip').length;
    quiz.done = quiz.pos >= quiz.queue.length || (answered >= 2 && scored[0].share >= 0.85);
    applyQuiz(scored);
    showQuestion();
  }

  function applyQuiz(scored) {
    const answered = [...quiz.answers.values()].some((x) => x !== 'skip');
    const best = scored[0];
    if (!answered) {
      el.confirmResult.textContent = 'No yes or no answers, so the result from the photo stays the same.';
      return;
    }
    const what = best.d.healthy ? `a healthy ${best.d.plant.toLowerCase()} leaf` : shortName(best.d, current.d);
    el.confirmResult.textContent = `After your answers, the best match is ${what} (${pct0(best.share)} among these possibilities). ` +
      'This combines the photo with your answers; confirm with a technician or plant clinic.';
    el.name.textContent = best.d.name;
    el.sci.replaceChildren(...sciLine(best.d));
    renderDetails(best.d, 'moderate');
    el.details.hidden = false;
    el.sourcesBox.hidden = !best.d.record;
  }

  el.confirmStep.querySelectorAll('[data-a]').forEach((b) => b.addEventListener('click', () => answerQuestion(b.dataset.a)));
  el.confirmBack.addEventListener('click', () => {
    if (!quiz || quiz.pos === 0) return;
    quiz.pos--;
    quiz.answers.delete(quiz.queue[quiz.pos].key);
    quiz.done = false;
    applyQuiz(quizScores());
    showQuestion();
  });
  el.confirmRestart.addEventListener('click', () => showResult());

  function fillList(ul, items) {
    ul.replaceChildren(...items.map((t) => make('li', null, t)));
  }

  // The catalogue of supported plants: one row per plant with its local and scientific names and the
  // problems LeafScan recognizes on it, filterable by any of those words.
  function renderCrops() {
    const byHost = new Map();
    entries.forEach((e, i) => {
      if (!e.host) return;
      if (!byHost.has(e.host)) byHost.set(e.host, []);
      if (!e.healthy) byHost.get(e.host).push(describe(i).name);
    });
    const hosts = [...byHost.keys()].sort((a, b) => plantName(a).localeCompare(plantName(b)));
    const problems = [...byHost.values()].reduce((sum, names) => sum + names.length, 0);
    el.emptySub.textContent = `Recognizes ${problems} problems on ${hosts.length} plants. The photo never leaves your device.`;
    el.introFacts.textContent = `${problems} problems on ${hosts.length} plants · about 7 in 10 right on test field photos · photos never leave your device`;
    el.plantsLede.textContent = `${hosts.length} plants and ${problems} problems, plus healthy leaves of each plant. ` +
      'Scientific names are checked against the GBIF Backbone Taxonomy.';
    el.cropList.replaceChildren(...hosts.map((host) => {
      const h = kb.hosts[host] || { en: host };
      const names = byHost.get(host).sort((a, b) => a.localeCompare(b));
      const li = document.createElement('li');
      li.dataset.search = [h.en, h.fil, h.scientific, ...names].filter(Boolean).join(' ').toLowerCase();
      const head = make('div', 'pr-head');
      head.append(make('span', 'pr-name', h.en));
      if (h.fil) head.append(make('span', 'pr-fil', h.fil));
      head.append(make('span', 'pr-n', names.length ? `${names.length} ${names.length === 1 ? 'problem' : 'problems'}` : 'healthy leaves only'));
      const sci = make('p', 'pr-sci');
      if (h.scientific) sci.append(italic(h.scientific), h.rank === 'genus' ? ' spp.' : '');
      li.append(head, sci, make('p', 'pr-list', names.join(' · ')));
      return li;
    }));
  }

  el.plantFilter.addEventListener('input', () => {
    const q = el.plantFilter.value.trim().toLowerCase();
    let visible = 0;
    for (const li of el.cropList.children) {
      const hit = !q || li.dataset.search.includes(q);
      li.hidden = !hit;
      if (hit) visible++;
    }
    el.plantsEmpty.hidden = visible > 0;
  });

  // ---------- Sections (Scan, History, Plants, Guide) ----------
  // Switched in place, without changing the URL: on iPhone home-screen apps a URL change can reset the
  // camera permission (WebKit bug 215884).

  let section = 'scan';

  function showView(name, { focusTab = false } = {}) {
    if (name === section) return;
    section = name;
    for (const b of el.tabs) {
      const on = b.dataset.view === name;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', String(on));
      b.tabIndex = on ? 0 : -1;
      $(`view-${b.dataset.view}`).hidden = !on;
      if (on && focusTab) b.focus();
    }
    window.scrollTo(0, 0); // the user chose another section: start it at the top
    if (name === 'history') renderStudyLog();
    if (name === 'scan') scheduleRender();
  }

  el.tabs.forEach((b) => b.addEventListener('click', () => showView(b.dataset.view)));
  el.navrow.addEventListener('keydown', (e) => {
    const i = el.tabs.findIndex((b) => b.dataset.view === section);
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: el.tabs.length - 1 }[e.key];
    if (next == null) return;
    e.preventDefault();
    showView(el.tabs[(next + el.tabs.length) % el.tabs.length].dataset.view, { focusTab: true });
  });
  el.brandBtn.addEventListener('click', () => showView('scan'));

  // ---------- History and feedback (stored only on this device) ----------

  const FB_LABEL = {
    right: 'marked right', wrong: 'marked wrong', unknown: 'not sure yet', unlisted: 'problem not in LeafScan\'s list',
  };

  function readHistory() {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; }
  }

  function readFeedback() {
    try { return JSON.parse(localStorage.getItem(FEEDBACK_KEY)) || []; } catch { return []; }
  }

  function saveHistory() {
    if (!current) return;
    try {
      const items = readHistory();
      // Re-checks, extra leaves and plant changes replace the plant's entry instead of stacking duplicates.
      const prev = items[0] && items[0].sid === plantId ? items.shift() : null;
      const leaf = leaves.find((l) => source && l.id === source.id) || leaves[leaves.length - 1];
      const id = labels[current.top.i].id;
      items.unshift({
        t: Date.now(), sid: plantId, id, p: current.top.p, band: current.outcome, lock: plantLock,
        n: leaves.length, img: leaf.thumb, fb: prev && prev.id === id ? prev.fb : undefined,
      });
      localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, HISTORY_MAX)));
    } catch { /* storage unavailable: history is optional */ }
    renderHistory();
  }

  function historyName(h) {
    if (h.band === 'uncovered') return 'Plant not in LeafScan\'s list';
    const idx = labels.findIndex((l) => l.id === h.id);
    const d = idx >= 0 ? describe(idx) : { plant: h.id.split(/___|\//)[0], name: h.id.split(/___|\//)[1] || '' };
    if (h.band === 'healthy-only') return `${d.plant}: diseases not covered`;
    if (h.band === 'unknown') return `Not recognized (closest: ${d.plant})`;
    const unsure = h.band ? h.band === 'unsure' : h.p < UNSURE;
    return unsure ? `Not sure (maybe ${d.plant}: ${d.name})` : `${d.plant}: ${d.name}`;
  }

  function renderHistory() {
    const items = readHistory();
    el.historyCard.hidden = items.length === 0;
    el.historyEmpty.hidden = items.length > 0;
    el.historyCount.hidden = items.length === 0;
    el.historyCount.textContent = String(items.length);
    el.historyList.replaceChildren(...items.map((h) => {
      const name = historyName(h);
      const li = document.createElement('li');
      const img = document.createElement('img');
      img.src = h.img; img.alt = ''; img.width = 44; img.height = 44;
      const text = make('div', 'h-text');
      const special = h.band === 'uncovered' || h.band === 'healthy-only';
      text.append(
        make('div', 'h-name', name),
        make('div', 'h-meta', [special ? null : pct(h.p), h.n > 1 ? `${h.n} leaves` : null,
          new Date(h.t).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }), FB_LABEL[h.fb]]
          .filter(Boolean).join(' · ')),
      );
      li.append(img, text);
      // Feedback can be given later, once the real cause is known.
      if (!special && (!h.fb || h.fb === 'unknown')) {
        const box = make('div', 'h-fb');
        for (const [a, sym, label] of [['right', '✓', 'Mark right'], ['wrong', '✗', 'Mark wrong']]) {
          const b = make('button', a, sym);
          b.type = 'button';
          b.title = label;
          b.setAttribute('aria-label', `${label}: ${name}`);
          b.addEventListener('click', () => setFeedback(h.sid, a));
          box.append(b);
        }
        li.append(box);
      }
      return li;
    }));
    renderTally();
  }

  // "Your checks so far": how often results in each confidence band turned out right for this user.
  function renderTally() {
    const list = readFeedback();
    const parts = [['confident', 'high confidence'], ['moderate', 'medium'], ['unsure', 'not sure'], ['unknown', 'not recognized']]
      .map(([band, label]) => {
        const xs = list.filter((f) => f.band === band && (f.a === 'right' || f.a === 'wrong'));
        return xs.length ? `${label} ${xs.filter((f) => f.a === 'right').length} of ${xs.length} right` : null;
      }).filter(Boolean);
    const unlisted = list.filter((f) => f.a === 'unlisted').length;
    if (unlisted) parts.push(`${unlisted} not in LeafScan's list`);
    el.tally.hidden = !parts.length;
    el.tally.textContent = parts.length ? `Your checks so far: ${parts.join(' · ')}.` : '';
  }

  function setFeedback(sid, a) {
    try {
      const items = readHistory();
      const h = items.find((x) => x.sid === sid);
      if (h) {
        h.fb = a;
        localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
      }
      const info = h || (current && sid === plantId
        ? { id: labels[current.top.i].id, p: current.top.p, band: current.outcome, lock: plantLock } : {});
      const list = readFeedback().filter((f) => f.sid !== sid);
      list.unshift({ sid, t: Date.now(), a, id: info.id, p: info.p, band: info.band || bandOf(info.p || 0), lock: info.lock || null });
      localStorage.setItem(FEEDBACK_KEY, JSON.stringify(list.slice(0, FEEDBACK_MAX)));
    } catch {
      el.feedbackNote.textContent = 'Could not save on this device.';
      return;
    }
    renderHistory();
    renderFeedbackState();
  }

  function renderFeedbackState() {
    const f = readFeedback().find((x) => x.sid === plantId);
    el.feedbackSeg.querySelectorAll('[data-a]').forEach((b) => b.setAttribute('aria-pressed', String(!!f && f.a === b.dataset.a)));
    el.feedbackNote.textContent = !f ? '' : {
      right: 'Saved as right. Thank you.',
      wrong: 'Saved as wrong. Please get a second opinion before treating the plant (see "Still not sure" below).',
      unknown: 'Saved. You can answer later under History.',
      unlisted: 'Saved. A plant clinic or an agriculture teacher can identify it (see "Still not sure" below).',
    }[f.a];
  }

  el.feedbackSeg.querySelectorAll('[data-a]').forEach((b) => b.addEventListener('click', () => {
    if (current) setFeedback(plantId, b.dataset.a);
  }));
  el.unlistedBtn.addEventListener('click', () => { if (current) setFeedback(plantId, 'unlisted'); });

  el.clearHistory.addEventListener('click', () => {
    try {
      localStorage.removeItem(HISTORY_KEY);
      localStorage.removeItem(FEEDBACK_KEY);
    } catch { /* ignore */ }
    renderHistory();
    renderFeedbackState();
  });

  // ---------- Summary to show or send to a technician, teacher or plant clinic ----------

  let techFiles = [];

  function techContent() {
    const c = current;
    const d = c.d;
    const special = c.outcome === 'uncovered' || c.outcome === 'healthy-only';
    const facts = [];
    const add = (label, value) => facts.push([label, value]);
    const hostSci = d.hostSci ? ` (${d.hostSci}${d.hostRank === 'genus' ? ' spp.' : ''})` : '';
    add('Plant', c.outcome === 'uncovered' ? 'Not in LeafScan\'s list (chosen by the user)'
      : `${d.plant}${hostSci}, ${plantLock ? 'chosen by the user' : 'guessed by LeafScan'}`);
    if (special) {
      add('LeafScan result', 'None: LeafScan does not cover diseases of this plant');
    } else {
      const name = c.outcome === 'unknown' ? `Not recognized (closest guess: ${d.healthy ? 'healthy' : d.name})`
        : (d.healthy ? 'Healthy' : d.name);
      const word = c.outcome === 'unknown' ? 'very low confidence' : CONF_WORD[c.band].toLowerCase();
      add('LeafScan result', `${name}${d.record?.agent ? ` (${d.record.agent})` : ''}, ${pct0(c.top.p)}, ${word}`);
      const others = c.ranked.slice(1, 4).filter(({ p }) => p >= 0.01).map(({ p, i }) => {
        const o = describe(i);
        return `${o.plant}: ${o.healthy ? 'healthy' : o.name.toLowerCase()} ${pct(p)}`;
      });
      if (others.length) add('Other possibilities', others);
    }
    if (quiz && quiz.answers.size) {
      const said = { yes: 'Yes', no: 'No', skip: 'Not sure' };
      add('Answers to symptom questions', quiz.queue.filter((it) => quiz.answers.has(it.key))
        .map((it) => `${it.q} ${said[quiz.answers.get(it.key)]}`));
    }
    add('Photo check', leaves.map((l, n) => `Leaf ${n + 1}: ${l.quality.blurry ? 'blurry' : l.quality.dark ? 'too dark' : 'sharp, good light'}`));
    add('How it was checked', `${leaves.length} leaf photo${leaves.length > 1 ? 's' : ''} checked on the user's device` +
      `${lastMs != null ? ` in ${Math.round(lastMs)} ms` : ''} by LeafScan, a student prototype (MobileNetV3 model, ` +
      'about 72% right on 3,100 test field photos from public datasets, not yet Philippine photos).');
    const fb = readFeedback().find((x) => x.sid === plantId);
    if (fb) add('User\'s feedback', FB_LABEL[fb.a]);
    const date = new Date().toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
    const header = `LeafScan result, ${date}. This is a student research prototype and can be wrong; please confirm the diagnosis.`;
    const text = [header, ...facts.map(([label, value]) => `${label}: ${Array.isArray(value) ? value.join('; ') : value}`)].join('\n');
    return { header, facts, text };
  }

  async function openTech() {
    if (!current) return;
    const { header, facts } = techContent();
    const pics = make('div', 'tech-leaves');
    leaves.forEach((l, n) => {
      const fig = document.createElement('figure');
      const img = document.createElement('img');
      img.src = l.crop;
      img.alt = `Leaf ${n + 1}, the square LeafScan checked`;
      fig.append(img, make('figcaption', null, `Leaf ${n + 1}`));
      pics.append(fig);
    });
    const dl = make('dl', 'tech-facts');
    for (const [label, value] of facts) {
      const row = document.createElement('div');
      const dd = document.createElement('dd');
      if (Array.isArray(value)) {
        const ul = document.createElement('ul');
        ul.append(...value.map((v) => make('li', null, v)));
        dd.append(ul);
      } else {
        dd.textContent = value;
      }
      row.append(make('dt', null, label), dd);
      dl.append(row);
    }
    el.techBody.replaceChildren(make('p', 'small muted', header), pics, dl);
    el.techNote.textContent = '';
    el.techDialog.showModal();
    // Prepare the leaf photos now, so sharing starts right away when the button is tapped.
    techFiles = [];
    try {
      techFiles = await Promise.all(leaves.map(async (l, n) =>
        new File([await (await fetch(l.crop)).blob()], `leafscan-leaf-${n + 1}.jpg`, { type: 'image/jpeg' })));
    } catch { techFiles = []; }
  }

  el.techBtn.addEventListener('click', openTech);
  el.techClose.addEventListener('click', () => el.techDialog.close());
  el.techDialog.addEventListener('click', (e) => { if (e.target === el.techDialog) el.techDialog.close(); });
  el.techShare.addEventListener('click', async () => {
    if (!current) return;
    const { text } = techContent();
    if (navigator.share) {
      const data = { title: 'LeafScan result', text };
      if (techFiles.length && navigator.canShare && navigator.canShare({ files: techFiles })) data.files = techFiles;
      try {
        await navigator.share(data);
        el.techNote.textContent = 'Shared.';
        return;
      } catch (err) {
        if (err && err.name === 'AbortError') return;
      }
    }
    try {
      await navigator.clipboard.writeText(text);
      el.techNote.textContent = 'Copied. Paste it into a message to your technician, teacher or plant clinic, and attach the leaf photos.';
    } catch {
      el.techNote.textContent = 'Sharing is not available here. Show this screen instead.';
    }
  });

  // ---------- Study mode (research team): log each photo with the study's photo_log.csv fields ----------
  // Entries and photos are kept in IndexedDB on this phone only, until the team exports them as a ZIP with the
  // photos and a photo_log.csv in the study's columns (field-photos/photo_log.csv).

  const STUDY_KEY = 'leafscan.study.v1';
  const STUDY_DEFAULTS_KEY = 'leafscan.study.defaults.v1';
  const STUDY_COLUMNS = ['photo_file', 'date', 'barangay', 'municipality', 'province', 'plant (Filipino / English)',
    'healthy or sick', 'what you think it is', 'confirmed by (technician / office)', 'confirmed disease',
    'garden or farm', 'background (natural / plain paper)', 'light (sun / shade / cloudy)', 'phone model', 'notes',
    'leafscan_result', 'leafscan_confidence', 'leafscan_plant_chosen_by_user', 'leafscan_model'];
  const REMEMBERED = ['barangay', 'municipality', 'province', 'site', 'background', 'light', 'phone'];
  let studySavedFor = null; // the photo whose entry has been saved, so it is not saved twice

  const studyOn = () => { try { return localStorage.getItem(STUDY_KEY) === '1'; } catch { return false; } };

  function studyDb() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open('leafscan-study', 1);
      req.onupgradeneeded = () => req.result.createObjectStore('entries', { keyPath: 'id', autoIncrement: true });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async function studyTx(mode, fn) {
    const db = await studyDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction('entries', mode);
      const out = fn(tx.objectStore('entries'));
      tx.oncomplete = () => { db.close(); resolve(out && 'result' in out ? out.result : undefined); };
      tx.onerror = () => { db.close(); reject(tx.error); };
    });
  }

  const studyAll = () => studyTx('readonly', (s) => s.getAll());

  function photoBlob(max = 1024) {
    const scale = Math.min(1, max / Math.max(source.w, source.h));
    const c = document.createElement('canvas');
    c.width = Math.round(source.w * scale);
    c.height = Math.round(source.h * scale);
    c.getContext('2d').drawImage(source.image, 0, 0, c.width, c.height);
    return new Promise((resolve) => c.toBlob(resolve, 'image/jpeg', 0.88));
  }

  function renderStudyBox() {
    const on = studyOn() && !!current && !!source;
    el.studyBox.hidden = !on;
    if (!on) return;
    const f = el.studyForm.elements;
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(STUDY_DEFAULTS_KEY)) || {}; } catch { saved = {}; }
    for (const k of REMEMBERED) if (saved[k] && !f[k].value) f[k].value = saved[k];
    if (studySavedFor !== source.id) {
      const h = current.outcome === 'uncovered' ? null : kb.hosts[current.d.host];
      f.plant.value = h ? [h.fil, h.en].filter(Boolean).join(' / ') : '';
      f.healthy.value = current.d.healthy && current.outcome !== 'unknown' ? 'healthy' : 'sick';
      f.guess.value = '';
      f.notes.value = '';
      el.studySave.disabled = false;
      el.studySave.textContent = 'Save to study log';
      el.studyNote.textContent = '';
    }
  }

  el.studyForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!current || !source || studySavedFor === source.id) return;
    const f = el.studyForm.elements;
    const val = (k) => f[k].value.trim();
    try {
      localStorage.setItem(STUDY_DEFAULTS_KEY, JSON.stringify(Object.fromEntries(REMEMBERED.map((k) => [k, val(k)]))));
    } catch { /* optional */ }
    el.studySave.disabled = true;
    try {
      const entries = await studyAll();
      const plantEn = (val('plant').split('/').pop() || 'leaf').trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'leaf';
      const n = entries.filter((x) => x.photoFile.startsWith(plantEn + '_')).length + 1;
      const now = new Date();
      const pad = (x, w = 2) => String(x).padStart(w, '0');
      const leaf = leaves.find((l) => l.id === source.id);
      const entry = {
        created: now.getTime(),
        photoFile: `${plantEn}_${pad(n, 3)}.jpg`,
        date: `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`,
        barangay: val('barangay'), municipality: val('municipality'), province: val('province'),
        plant: val('plant'), healthy: f.healthy.value, guess: val('guess'), confirmedBy: '', confirmedDisease: '',
        site: f.site.value, background: f.background.value, light: f.light.value, phone: val('phone'), notes: val('notes'),
        result: current.outcome === 'uncovered' ? 'plant not covered' : labels[current.top.i].id,
        confidence: current.outcome === 'uncovered' ? '' : current.top.p.toFixed(3),
        plantChosen: plantLock || '', model: modelVersion,
        thumb: leaf ? leaf.thumb : '', photo: await photoBlob(),
      };
      await studyTx('readwrite', (s) => s.add(entry));
      studySavedFor = source.id;
      el.studySave.textContent = 'Saved';
      el.studyNote.textContent = `Saved as ${entry.photoFile} (${entries.length + 1} in the study log).`;
      renderStudyLog();
    } catch (err) {
      console.error(err);
      el.studySave.disabled = false;
      el.studyNote.textContent = 'Could not save on this phone (storage may be full).';
    }
  });

  async function renderStudyLog() {
    let entries = [];
    try { entries = await studyAll(); } catch { entries = []; }
    el.studyLog.hidden = !studyOn() && entries.length === 0;
    el.studyCount.textContent = entries.length ? `(${entries.length})` : '';
    el.studyExport.disabled = entries.length === 0;
    el.studyList.replaceChildren(...entries.slice().reverse().map((x) => {
      const li = document.createElement('li');
      const img = document.createElement('img');
      img.src = x.thumb || '';
      img.alt = '';
      const idx = labels.findIndex((l) => l.id === x.result);
      const res = idx >= 0 ? describe(idx) : null;
      li.append(img,
        make('div', 's-title', `${x.photoFile} · ${x.plant || 'plant not given'}`),
        make('div', 's-meta', [x.date, [x.barangay, x.municipality].filter(Boolean).join(', '),
          res ? `LeafScan: ${res.healthy ? 'healthy' : res.name} ${pct0(Number(x.confidence))}` : x.result].filter(Boolean).join(' · ')));
      const confirm = make('div', 's-confirm');
      for (const [key, label] of [['confirmedBy', 'Confirmed by'], ['confirmedDisease', 'Confirmed disease']]) {
        const input = document.createElement('input');
        input.value = x[key] || '';
        input.placeholder = label;
        input.setAttribute('aria-label', `${label} for ${x.photoFile}`);
        input.addEventListener('change', () => {
          x[key] = input.value.trim();
          studyTx('readwrite', (s) => s.put(x)).catch((err) => console.error(err));
        });
        confirm.append(input);
      }
      li.append(confirm);
      return li;
    }));
  }

  // A minimal ZIP writer (files stored without compression, which JPEG photos don't need).
  const CRC_TABLE = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      t[n] = c >>> 0;
    }
    return t;
  })();

  function crc32(bytes) {
    let c = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }

  function zipStore(files) {
    const enc = new TextEncoder();
    const parts = [], central = [];
    let offset = 0;
    const d = new Date();
    const time = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
    const date = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
    for (const f of files) {
      const name = enc.encode(f.name);
      const crc = crc32(f.data);
      const local = new DataView(new ArrayBuffer(30));
      [[0, 0x04034b50, 4], [4, 20, 2], [6, 0x0800, 2], [8, 0, 2], [10, time, 2], [12, date, 2], [14, crc, 4],
        [18, f.data.length, 4], [22, f.data.length, 4], [26, name.length, 2], [28, 0, 2]]
        .forEach(([o, v, s]) => (s === 4 ? local.setUint32(o, v, true) : local.setUint16(o, v, true)));
      parts.push(local.buffer, name, f.data);
      const cen = new DataView(new ArrayBuffer(46));
      [[0, 0x02014b50, 4], [4, 20, 2], [6, 20, 2], [8, 0x0800, 2], [10, 0, 2], [12, time, 2], [14, date, 2], [16, crc, 4],
        [20, f.data.length, 4], [24, f.data.length, 4], [28, name.length, 2], [30, 0, 2], [32, 0, 2], [34, 0, 2],
        [36, 0, 2], [38, 0, 4], [42, offset, 4]]
        .forEach(([o, v, s]) => (s === 4 ? cen.setUint32(o, v, true) : cen.setUint16(o, v, true)));
      central.push(cen.buffer, name);
      offset += 30 + name.length + f.data.length;
    }
    const size = central.reduce((s, b) => s + b.byteLength, 0);
    const end = new DataView(new ArrayBuffer(22));
    [[0, 0x06054b50, 4], [8, files.length, 2], [10, files.length, 2], [12, size, 4], [16, offset, 4]]
      .forEach(([o, v, s]) => (s === 4 ? end.setUint32(o, v, true) : end.setUint16(o, v, true)));
    return new Blob([...parts, ...central, end.buffer], { type: 'application/zip' });
  }

  const csvCell = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));

  el.studyExport.addEventListener('click', async () => {
    const entries = await studyAll();
    if (!entries.length) return;
    el.studyExportNote.textContent = 'Preparing the export…';
    const rows = entries.map((x) => [x.photoFile, x.date, x.barangay, x.municipality, x.province, x.plant, x.healthy,
      x.guess, x.confirmedBy, x.confirmedDisease, x.site, x.background, x.light, x.phone, x.notes,
      x.result, x.confidence, x.plantChosen, x.model]);
    const csv = [STUDY_COLUMNS, ...rows].map((r) => r.map(csvCell).join(',')).join('\r\n') + '\r\n';
    const files = [{ name: 'photo_log.csv', data: new TextEncoder().encode('﻿' + csv) }];
    for (const x of entries) files.push({ name: `photos/${x.photoFile}`, data: new Uint8Array(await x.photo.arrayBuffer()) });
    const stamp = new Date().toISOString().slice(0, 10);
    const zip = new File([zipStore(files)], `leafscan-study-${stamp}.zip`, { type: 'application/zip' });
    try {
      if (navigator.canShare && navigator.canShare({ files: [zip] })) {
        await navigator.share({ files: [zip], title: 'LeafScan study log' });
        el.studyExportNote.textContent = `Shared ${entries.length} photos and photo_log.csv.`;
        return;
      }
    } catch (err) {
      if (err && err.name === 'AbortError') { el.studyExportNote.textContent = ''; return; }
    }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(zip);
    a.download = zip.name;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 30000);
    el.studyExportNote.textContent = `Saved ${zip.name} (${entries.length} photos and photo_log.csv).`;
  });

  el.studyClear.addEventListener('click', async () => {
    if (!window.confirm('Delete every photo and entry in the study log on this phone? Export first if you still need them.')) return;
    await studyTx('readwrite', (s) => s.clear());
    el.studyExportNote.textContent = 'The study log on this phone is empty.';
    renderStudyLog();
  });

  el.studyToggle.checked = studyOn();
  el.studyToggle.addEventListener('change', () => {
    try { localStorage.setItem(STUDY_KEY, el.studyToggle.checked ? '1' : '0'); } catch { /* optional */ }
    renderStudyBox();
    renderStudyLog();
  });
  renderStudyLog();

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

  // Registered only once the model is ready: the service worker then saves the app and the runtime from the
  // browser's own cache instead of downloading them a second time while the model is still coming in.
  function registerOffline() {
    if (!('serviceWorker' in navigator) || !window.isSecureContext) return;
    navigator.serviceWorker.register('sw.js').then(() => navigator.serviceWorker.ready).then(() => {
      el.offlineNote.textContent = 'Saved on this device: LeafScan works without internet after the first visit.' +
        (isIOS && !standalone ? ' On iPhone or iPad, tap Share → Add to Home Screen to install it.' : '');
    }).catch((err) => console.warn('Service worker failed', err));
  }

  el.retryBtn.addEventListener('click', () => startModel());

  init();
})();

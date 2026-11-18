// Offline support, in separate stores so that updating the app never makes people download the big files again:
//   leafscan-app-<n>        the small app files; replaced on every app update
//   leafscan-runtime-<ver>  ONNX Runtime (core about 14 MB, stored by the page); kept until the runtime changes
//   leafscan-model-<ver>    the model's parts; written by the page itself (app.js), never touched here
//   leafscan-lesion-<ver>   the diseased-spots model's parts, downloaded only when asked; same rules
// The page registers this worker only after the model has loaded. The small app files are always checked with
// the server ('no-cache': an unchanged file costs a short 304 reply, not a second download), because GitHub Pages
// lets browsers keep files for 10 minutes: without the check, an update published in those minutes could store the
// new page with the old app.js.
const APP_CACHE = 'leafscan-app-26';
const RUNTIME_CACHE = 'leafscan-runtime-1.29.0';
const APP_FILES = [
  './',
  'index.html',
  'label.html',
  'label.js',
  'styles.css',
  'app.js',
  'worker.js',
  'manifest.webmanifest',
  'fonts/spectral-600-latin.woff2',
  'data/kb.json',
  'data/field_codes.json',
  'model/labels.json',
  'model/model.json',
  'model/lesion.json',
  'model/second.json',
  'model/shortlist.json',
  'icons/icon.svg',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'icons/icon-maskable-512.png',
];
// The 14 MB runtime core (vendor/ort-wasm-simd-threaded.wasm) is stored in RUNTIME_CACHE by the page itself.
const RUNTIME_FILES = [
  'vendor/ort.wasm.min.js',
  'vendor/ort-wasm-simd-threaded.mjs',
];
const KEEP = (key) => key === APP_CACHE || key === RUNTIME_CACHE || key.startsWith('leafscan-model-') || key.startsWith('leafscan-lesion-') || key.startsWith('leafscan-second-');
const MODEL_PART = /\/model\/[^/]+\/part-\d+\.bin$/;
const CACHE_FIRST = /\.(wasm|mjs|png|svg|woff2)$|\/vendor\//;

self.addEventListener('install', (event) => {
  event.waitUntil(Promise.all([
    caches.open(APP_CACHE).then((cache) => cache.addAll(APP_FILES.map((f) => new Request(f, { cache: 'no-cache' })))),
    // Runtime files are fetched only if this version of them is not stored yet.
    caches.open(RUNTIME_CACHE).then(async (cache) => {
      const missing = [];
      for (const f of RUNTIME_FILES) if (!(await cache.match(f))) missing.push(f);
      if (missing.length) await cache.addAll(missing);
    }),
  ]).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('leafscan-') && !KEEP(k)).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

function fetchAndStore(request, cacheName, check = false) {
  // A page navigation cannot be copied with new options, so it is fetched again by its address.
  const fresh = !check ? request : request.mode === 'navigate' ? new Request(request.url, { cache: 'no-cache', redirect: 'manual' })
    : new Request(request, { cache: 'no-cache' });
  return fetch(fresh).then((res) => {
    if (res.ok && res.type === 'basic') {
      const copy = res.clone();
      caches.open(cacheName).then((cache) => cache.put(request, copy));
    }
    return res;
  });
}

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;

  if (MODEL_PART.test(url.pathname)) {
    // The page stores model parts itself; just answer from any cache, else the network.
    event.respondWith(caches.match(request, { ignoreSearch: true }).then((hit) => hit || fetch(request)));
  } else if (CACHE_FIRST.test(url.pathname)) {
    const store = url.pathname.includes('/vendor/') ? RUNTIME_CACHE : APP_CACHE;
    event.respondWith(caches.match(request, { ignoreSearch: true }).then((hit) => hit || fetchAndStore(request, store)));
  } else {
    event.respondWith(fetchAndStore(request, APP_CACHE, true).catch(() =>
      caches.match(request, { ignoreSearch: true }).then((hit) => hit || caches.match('index.html'))));
  }
});

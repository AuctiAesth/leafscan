// Offline support, in three stores so that updating the app never makes people download the big files again:
//   leafscan-app-<n>        the small app files; replaced on every app update
//   leafscan-runtime-<ver>  ONNX Runtime (core about 14 MB, stored by the page); kept until the runtime changes
//   leafscan-model-<ver>    the model's parts; written by the page itself (app.js), never touched here
// The page registers this worker only after the model has loaded, so these files come from the browser's
// own HTTP cache rather than being downloaded a second time.
const APP_CACHE = 'leafscan-app-10';
const RUNTIME_CACHE = 'leafscan-runtime-1.29.0';
const APP_FILES = [
  './',
  'index.html',
  'styles.css',
  'app.js',
  'worker.js',
  'manifest.webmanifest',
  'fonts/spectral-600-latin.woff2',
  'data/kb.json',
  'model/labels.json',
  'model/model.json',
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
const KEEP = (key) => key === APP_CACHE || key === RUNTIME_CACHE || key.startsWith('leafscan-model-');
const MODEL_PART = /\/model\/[^/]+\/part-\d+\.bin$/;
const CACHE_FIRST = /\.(wasm|mjs|png|svg|woff2)$|\/vendor\//;

self.addEventListener('install', (event) => {
  event.waitUntil(Promise.all([
    caches.open(APP_CACHE).then((cache) => cache.addAll(APP_FILES)),
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

function fetchAndStore(request, cacheName) {
  return fetch(request).then((res) => {
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
    event.respondWith(fetchAndStore(request, APP_CACHE).catch(() =>
      caches.match(request, { ignoreSearch: true }).then((hit) => hit || caches.match('index.html'))));
  }
});

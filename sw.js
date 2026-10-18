// Offline support: cache the app, the ONNX Runtime files and the model on first visit.
// Large files that never change (model, runtime) are served from the cache first;
// the small app files are fetched fresh when online so updates arrive, with the cache as fallback.
const CACHE = 'leafscan-v3';
const FILES = [
  './',
  'index.html',
  'styles.css',
  'app.js',
  'manifest.webmanifest',
  'data/kb.json',
  'model/labels.json',
  'model/plant_disease_mnv3.onnx',
  'vendor/ort.wasm.min.js',
  'vendor/ort-wasm-simd-threaded.mjs',
  'vendor/ort-wasm-simd-threaded.wasm',
  'icons/icon.svg',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'icons/icon-maskable-512.png',
];
const CACHE_FIRST = /\.(onnx|wasm|mjs|png|svg)$|\/vendor\//;

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

function fetchAndStore(request) {
  return fetch(request).then((res) => {
    if (res.ok && res.type === 'basic') {
      const copy = res.clone();
      caches.open(CACHE).then((cache) => cache.put(request, copy));
    }
    return res;
  });
}

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;

  if (CACHE_FIRST.test(url.pathname)) {
    event.respondWith(caches.match(request, { ignoreSearch: true }).then((hit) => hit || fetchAndStore(request)));
  } else {
    event.respondWith(fetchAndStore(request).catch(() =>
      caches.match(request, { ignoreSearch: true }).then((hit) => hit || caches.match('index.html'))));
  }
});

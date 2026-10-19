/* LeafScan model worker: runs the ONNX model off the main thread, so moving and zooming the photo stays
 * smooth while a check is running on a slow phone. app.js falls back to the main thread if this fails. */
importScripts('vendor/ort.wasm.min.js');

ort.env.wasm.wasmPaths = new URL('vendor/', self.location.href).href;
ort.env.wasm.numThreads = 1;

const SIZE = 224;
let session = null;

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'load') {
    try {
      if (msg.wasm) ort.env.wasm.wasmBinary = msg.wasm; // already downloaded and stored by the page
      session = await ort.InferenceSession.create(msg.bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      // Warm-up so the first real check is fast
      await session.run({ image: new ort.Tensor('float32', new Float32Array(3 * SIZE * SIZE), [1, 3, SIZE, SIZE]) });
      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'failed', message: String(err && err.message || err) });
    }
  } else if (msg.type === 'run') {
    try {
      const t0 = performance.now();
      const out = await session.run({ image: new ort.Tensor('float32', msg.data, [1, 3, SIZE, SIZE]) });
      const probs = Float32Array.from(out.probabilities.data);
      self.postMessage({ type: 'result', id: msg.id, probs, ms: performance.now() - t0 }, [probs.buffer]);
    } catch (err) {
      self.postMessage({ type: 'error', id: msg.id, message: String(err && err.message || err) });
    }
  }
};

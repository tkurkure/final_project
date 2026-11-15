// Runs the stance model off the main thread, so the page stays responsive while it reads.
//
// Messages in:  {type: "predict", id, sentences, local}
// Messages out: {type: "download", loaded, total}   model download progress, in bytes
//               {type: "batch", id, done, total}    sentences read so far
//               {type: "result", id, probs}          one [hawkish, dovish, neutral, irrelevant] row per sentence
//               {type: "error", id, message}
import { AutoTokenizer, AutoModelForSequenceClassification, env }
  from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const HUB_MODEL = "f006tsj/central-bank-stance-roberta";
const MAX_LENGTH = 128;     // matches training (config.DEFAULT_HPARAMS["max_length"])
const BATCH = 8;

let modelPromise = null;

function loadModel(local) {
  if (modelPromise) return modelPromise;
  if (local) {      // ?local: serve the export from ./web_model/ instead of the Hub
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = new URL("./", self.location.href).href;
  }
  const id = local ? "web_model" : HUB_MODEL;
  const files = {};
  const progress_callback = info => {
    if (info.status !== "progress" || !info.total) return;
    files[info.file] = info;
    const sum = key => Object.values(files).reduce((a, f) => a + f[key], 0);
    self.postMessage({ type: "download", loaded: sum("loaded"), total: sum("total") });
  };
  modelPromise = Promise.all([
    AutoTokenizer.from_pretrained(id, { progress_callback }),
    // fp32 on the CPU (wasm) backend: the same numbers the PyTorch model gives.
    AutoModelForSequenceClassification.from_pretrained(id, { dtype: "fp32", device: "wasm", progress_callback }),
  ]).catch(err => { modelPromise = null; throw err; });
  return modelPromise;
}

function softmax(row) {
  const m = Math.max(...row);
  const ex = row.map(x => Math.exp(x - m));
  const z = ex.reduce((a, b) => a + b, 0);
  return ex.map(x => x / z);
}

self.onmessage = async ({ data: { type, id, sentences, local } }) => {
  if (type !== "predict") return;
  try {
    const [tokenizer, model] = await loadModel(local);
    const probs = [];
    for (let i = 0; i < sentences.length; i += BATCH) {
      const inputs = tokenizer(sentences.slice(i, i + BATCH),
                               { padding: true, truncation: true, max_length: MAX_LENGTH });
      const { logits } = await model(inputs);
      const [n, k] = logits.dims;
      for (let r = 0; r < n; r++) probs.push(softmax(Array.from(logits.data.slice(r * k, r * k + k))));
      self.postMessage({ type: "batch", id, done: probs.length, total: sentences.length });
    }
    self.postMessage({ type: "result", id, probs });
  } catch (err) {
    self.postMessage({ type: "error", id, message: String(err?.message || err) });
  }
};

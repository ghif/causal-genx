const apiBaseUrl = window.CAUSAL_GENX_CONFIG.apiBaseUrl.replace(/\/$/, "");
const generation = {
  digit: () => Number(document.querySelector('input[name="digit"]:checked').value),
  thickness: document.querySelector("#generation-thickness"),
  intensity: document.querySelector("#generation-intensity"),
  styleSeed: document.querySelector("#style-seed"),
};
const counterfactual = {
  image: null,
  thickness: document.querySelector("#cf-thickness"),
  intensity: document.querySelector("#cf-intensity"),
  digit: null,
};
const apiStatus = document.querySelector("#api-status");
const buttons = [...document.querySelectorAll("button")];
let generationIntensityAbortController;
let counterfactualIntensityAbortController;

for (let digit = 0; digit < 10; digit += 1) {
  const wrapper = document.createElement("div");
  wrapper.innerHTML = `<input id="digit-${digit}" type="radio" name="digit" value="${digit}" ${digit === 3 ? "checked" : ""}><label for="digit-${digit}">${digit}</label>`;
  document.querySelector("#digit-control").append(wrapper);
}

function setStatus(id, message, kind = "") {
  const status = document.querySelector(id);
  status.textContent = message;
  status.className = `status ${kind}`;
}

function updateOutput(input, digits) {
  document.querySelector(`#${input.id}-value`).value = Number(input.value).toFixed(digits);
}

function imageFormData() {
  const data = new FormData();
  data.append("image", counterfactual.image);
  return data;
}

async function request(path, body) {
  const response = await fetch(`${apiBaseUrl}${path}`, { method: "POST", body });
  const content = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
  if (!response.ok) throw new Error(content.detail || `HTTP ${response.status}`);
  return content;
}

function showImage(id, base64) { document.querySelector(id).src = `data:image/png;base64,${base64}`; }

function showFactors(id, parents, extra = {}) {
  const values = { ...parents.physical, ...extra };
  for (const key of ["thickness", "intensity"]) values[key] = Number(values[key]).toFixed(3);
  document.querySelector(id).textContent = JSON.stringify(values, null, 2);
}

function setBusy(isBusy) {
  buttons.forEach((button) => { button.disabled = isBusy; });
  if (!isBusy && counterfactual.digit === null) document.querySelector("#render-counterfactual").disabled = true;
}

async function generateFromParents() {
  const data = new FormData();
  data.append("digit", generation.digit());
  data.append("thickness", generation.thickness.value);
  data.append("intensity", generation.intensity.value);
  data.append("style_seed", generation.styleSeed.value);
  setBusy(true); setStatus("#generation-status", "Generating image from parent nodes…");
  try {
    const result = await request("/v1/generate", data);
    showImage("#generated-image", result.image_png_base64);
    showFactors("#generated-factors", result.generated_parents, { style_seed: result.style_seed, latency_ms: result.latency_ms });
    setStatus("#generation-status", `Generated in ${result.latency_ms} ms.`, "ready");
  } catch (error) { setStatus("#generation-status", error.message, "error"); } finally { setBusy(false); }
}

async function syncGenerationIntensity() {
  generationIntensityAbortController?.abort();
  generationIntensityAbortController = new AbortController();
  const data = new FormData();
  data.append("thickness", generation.thickness.value);
  setStatus("#generation-status", "Deriving intensity from thickness…");
  try {
    const response = await fetch(`${apiBaseUrl}/v1/linked-intensity`, {
      method: "POST", body: data, signal: generationIntensityAbortController.signal,
    });
    const result = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
    generation.intensity.value = result.intensity_physical;
    updateOutput(generation.intensity, 1);
    setStatus("#generation-status", "Intensity derived from thickness. Generate an image when ready.", "ready");
  } catch (error) {
    if (error.name !== "AbortError") setStatus("#generation-status", error.message, "error");
  }
}

async function syncCounterfactualIntensity() {
  if (!counterfactual.image || counterfactual.digit === null) return;
  counterfactualIntensityAbortController?.abort();
  counterfactualIntensityAbortController = new AbortController();
  const data = imageFormData();
  data.append("thickness", counterfactual.thickness.value);
  setStatus("#counterfactual-status", "Deriving intensity from thickness and the factual image…");
  try {
    const response = await fetch(`${apiBaseUrl}/v1/linked-intensity`, {
      method: "POST", body: data, signal: counterfactualIntensityAbortController.signal,
    });
    const result = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
    counterfactual.intensity.value = result.intensity_physical;
    updateOutput(counterfactual.intensity, 1);
    setStatus("#counterfactual-status", "Intensity updated from thickness. You can refine either intervention before rendering.", "ready");
  } catch (error) {
    if (error.name !== "AbortError") setStatus("#counterfactual-status", error.message, "error");
  }
}

async function analyzeFactualImage() {
  if (!counterfactual.image) { setStatus("#counterfactual-status", "Choose a factual sample first.", "error"); return; }
  setBusy(true); setStatus("#counterfactual-status", "Inferring factual parent nodes…");
  try {
    const result = await request("/v1/predict-parents", imageFormData());
    const factual = result.factual_parents.physical;
    counterfactual.digit = factual.digit;
    counterfactual.thickness.value = factual.thickness;
    counterfactual.intensity.value = factual.intensity;
    counterfactual.thickness.disabled = false; counterfactual.intensity.disabled = false;
    document.querySelector("#render-counterfactual").disabled = false;
    document.querySelector("#digit-identity").textContent = `Digit ${factual.digit} is inferred from the factual image and will be preserved.`;
    updateOutput(counterfactual.thickness, 2); updateOutput(counterfactual.intensity, 1);
    showFactors("#factual-factors", result.factual_parents, { latency_ms: result.latency_ms });
    setStatus("#counterfactual-status", `Factual image analyzed in ${result.latency_ms} ms. Adjust the interventions, then render.`, "ready");
  } catch (error) { setStatus("#counterfactual-status", error.message, "error"); } finally { setBusy(false); }
}

async function renderCounterfactual() {
  if (!counterfactual.image || counterfactual.digit === null) { setStatus("#counterfactual-status", "Analyze the factual sample before rendering.", "error"); return; }
  const data = imageFormData();
  data.append("digit", counterfactual.digit);
  data.append("thickness", counterfactual.thickness.value);
  data.append("intensity", counterfactual.intensity.value);
  setBusy(true); setStatus("#counterfactual-status", "Rendering thickness and intensity intervention…");
  try {
    const result = await request("/v1/render-counterfactual", data);
    showImage("#seed-preview", result.seed_image_png_base64);
    showImage("#counterfactual-image", result.image_png_base64);
    showFactors("#factual-factors", result.factual_parents);
    showFactors("#counterfactual-factors", result.counterfactual_parents, { latency_ms: result.latency_ms });
    setStatus("#counterfactual-status", `Counterfactual rendered in ${result.latency_ms} ms.`, "ready");
  } catch (error) { setStatus("#counterfactual-status", error.message, "error"); } finally { setBusy(false); }
}

function activateTab(tabId) {
  for (const tab of document.querySelectorAll("[role=tab]")) {
    const selected = tab.id === tabId;
    tab.setAttribute("aria-selected", String(selected));
    document.querySelector(`#${tab.getAttribute("aria-controls")}`).hidden = !selected;
  }
}

async function checkApi() {
  try {
    const response = await fetch(`${apiBaseUrl}/readyz`);
    if (!response.ok) throw new Error();
    apiStatus.textContent = `API ready at ${apiBaseUrl}`; apiStatus.classList.add("ready");
    await syncGenerationIntensity();
  } catch { apiStatus.textContent = `API unavailable at ${apiBaseUrl}`; apiStatus.classList.add("error"); }
}

generation.styleSeed.addEventListener("input", () => updateOutput(generation.styleSeed, 0));
generation.thickness.addEventListener("input", () => updateOutput(generation.thickness, 2));
generation.thickness.addEventListener("change", syncGenerationIntensity);
counterfactual.thickness.addEventListener("input", () => updateOutput(counterfactual.thickness, 2));
counterfactual.intensity.addEventListener("input", () => updateOutput(counterfactual.intensity, 1));
counterfactual.thickness.addEventListener("change", syncCounterfactualIntensity);
function resetCounterfactualSelection() {
  counterfactual.digit = null;
  counterfactual.thickness.disabled = true; counterfactual.intensity.disabled = true;
  document.querySelector("#render-counterfactual").disabled = true;
  document.querySelector("#digit-identity").textContent = "Analyze a sample to preserve its inferred digit identity.";
}

async function selectSample(sample) {
  const source = sample.dataset.sampleSrc;
  setStatus("#counterfactual-status", "Loading factual sample…");
  try {
    const response = await fetch(source);
    if (!response.ok) throw new Error("Unable to load the selected sample.");
    const imageBlob = await response.blob();
    counterfactual.image = new File([imageBlob], source.split("/").pop(), { type: imageBlob.type || "image/png" });
    resetCounterfactualSelection();
    document.querySelectorAll(".sample-card").forEach((card) => card.classList.toggle("selected", card === sample));
    document.querySelector("#seed-preview").src = source;
    document.querySelector("#sample-dropzone").textContent = `${sample.dataset.sampleLabel} selected. Analyze to infer its factual parents.`;
    setStatus("#counterfactual-status", "Factual sample selected. Analyze it to begin.", "ready");
  } catch (error) { setStatus("#counterfactual-status", error.message, "error"); }
}

const sampleDropzone = document.querySelector("#sample-dropzone");
document.querySelectorAll(".sample-card").forEach((sample) => {
  sample.addEventListener("click", () => selectSample(sample));
  sample.addEventListener("dragstart", (event) => {
    event.dataTransfer.setData("text/plain", sample.dataset.sampleSrc);
    event.dataTransfer.effectAllowed = "copy";
  });
});
sampleDropzone.addEventListener("dragover", (event) => { event.preventDefault(); sampleDropzone.classList.add("drag-over"); });
sampleDropzone.addEventListener("dragleave", () => sampleDropzone.classList.remove("drag-over"));
sampleDropzone.addEventListener("drop", (event) => {
  event.preventDefault(); sampleDropzone.classList.remove("drag-over");
  const source = event.dataTransfer.getData("text/plain");
  const sample = [...document.querySelectorAll(".sample-card")].find((card) => card.dataset.sampleSrc === source);
  if (sample) selectSample(sample);
});
document.querySelector("#generation-tab").addEventListener("click", () => activateTab("generation-tab"));
document.querySelector("#counterfactual-tab").addEventListener("click", () => activateTab("counterfactual-tab"));
document.querySelector("#generate").addEventListener("click", generateFromParents);
document.querySelector("#load-factors").addEventListener("click", analyzeFactualImage);
document.querySelector("#render-counterfactual").addEventListener("click", renderCounterfactual);
updateOutput(generation.styleSeed, 0); updateOutput(generation.thickness, 2); updateOutput(generation.intensity, 1);
updateOutput(counterfactual.thickness, 2); updateOutput(counterfactual.intensity, 1); checkApi();

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import "./style.css";

const $ = (selector) => document.querySelector(selector);
const tokenInput = $("#token");
const unlockButton = $("#unlock");
const serviceState = $("#service-state");
const fileInput = $("#files");
const dropzone = $("#dropzone");
const fileSummary = $("#file-summary");
const reconstructButton = $("#reconstruct");
const jobState = $("#job-state");
const progress = $("#progress");
const emptyState = $("#empty-state");
const confidence = $("#confidence");
const confidenceValue = $("#confidence-value");
const download = $("#download");
const metricValues = [...document.querySelectorAll("#metrics dd")];

let unlocked = false;
let selectedFiles = [];
let downloadUrl = "";

function authHeaders() {
  return { Authorization: `Bearer ${tokenInput.value.trim()}` };
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...authHeaders(), ...(options.headers || {}) },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail || detail; } catch { /* response was not JSON */ }
    throw new Error(detail);
  }
  return response;
}

function uploadJob(form) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/jobs");
    request.setRequestHeader("Authorization", `Bearer ${tokenInput.value.trim()}`);
    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) return;
      const percent = Math.round((event.loaded / event.total) * 100);
      progress.style.width = `${Math.max(3, Math.round(percent * 0.24))}%`;
      setMessage(`Uploading video… ${percent}%`);
    });
    request.upload.addEventListener("load", () => {
      progress.style.width = "25%";
      setMessage("Extracting frames from the video…");
    });
    request.addEventListener("load", () => {
      let payload = {};
      try { payload = JSON.parse(request.responseText); } catch { /* handled below */ }
      if (request.status >= 200 && request.status < 300) resolve(payload);
      else reject(new Error(payload.detail || `${request.status} ${request.statusText}`));
    });
    request.addEventListener("error", () => reject(new Error("Upload connection failed")));
    request.send(form);
  });
}

function setMessage(message, error = false) {
  jobState.textContent = message;
  jobState.classList.toggle("error", error);
}

function setFiles(files) {
  selectedFiles = [...files];
  const bytes = selectedFiles.reduce((total, file) => total + file.size, 0);
  fileSummary.textContent = selectedFiles.length
    ? `${selectedFiles.length} file${selectedFiles.length === 1 ? "" : "s"} · ${(bytes / 1048576).toFixed(1)} MiB`
    : "No files selected";
  reconstructButton.disabled = !unlocked || selectedFiles.length === 0;
}

unlockButton.addEventListener("click", async () => {
  if (!tokenInput.value.trim()) return setMessage("Enter the deployment token.", true);
  unlockButton.disabled = true;
  try {
    const info = await (await api("/api/info")).json();
    unlocked = true;
    sessionStorage.setItem("lingbot-token", tokenInput.value.trim());
    serviceState.textContent = info.ready ? "Model ready" : "Model loading";
    serviceState.classList.remove("muted");
    $("#max-frames").max = info.limits.frames;
    setMessage(`${info.model.weight} resident on ${info.model.device}.`);
  } catch (error) {
    unlocked = false;
    serviceState.textContent = "Locked";
    serviceState.classList.add("muted");
    setMessage(error.message, true);
  } finally {
    unlockButton.disabled = false;
    reconstructButton.disabled = !unlocked || selectedFiles.length === 0;
  }
});

const rememberedToken = sessionStorage.getItem("lingbot-token");
if (rememberedToken) tokenInput.value = rememberedToken;
fileInput.addEventListener("change", () => setFiles(fileInput.files));
confidence.addEventListener("input", () => { confidenceValue.textContent = `${confidence.value}%`; });

for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
}
dropzone.addEventListener("drop", (event) => setFiles(event.dataTransfer.files));

const canvas = $("#scene");
const viewport = $("#viewport");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x080a0e, 0.018);
const camera = new THREE.PerspectiveCamera(48, 1, 0.001, 10000);
camera.position.set(2, 1.4, 2);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
scene.add(new THREE.HemisphereLight(0xc8ffff, 0x252019, 2.2));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
keyLight.position.set(4, 7, 5);
scene.add(keyLight);

let currentRoot = null;
function disposeCurrent() {
  if (!currentRoot) return;
  currentRoot.traverse((node) => {
    node.geometry?.dispose();
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    materials.filter(Boolean).forEach((material) => material.dispose());
  });
  scene.remove(currentRoot);
  currentRoot = null;
}

function frameObject(root) {
  const box = new THREE.Box3().setFromObject(root);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.length() * 0.5, 0.01);
  root.position.sub(center);
  root.traverse((node) => {
    if (node.isPoints) {
      node.material.size = Math.max(radius * 0.0022, 0.0004);
      node.material.sizeAttenuation = true;
      node.material.vertexColors = true;
      node.material.needsUpdate = true;
    }
  });
  camera.near = Math.max(radius / 10000, 0.0001);
  camera.far = radius * 100;
  camera.position.set(radius * 1.6, radius * 1.0, radius * 1.6);
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  controls.minDistance = radius * 0.02;
  controls.maxDistance = radius * 20;
  controls.update();
  scene.fog.density = 0.08 / radius;
}

function resize() {
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);
function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
resize();
animate();

async function showResult(job) {
  setMessage("Downloading the GLB preview…");
  const response = await api(job.result_url);
  const blob = await response.blob();
  const buffer = await blob.arrayBuffer();
  const gltf = await new Promise((resolve, reject) => {
    new GLTFLoader().parse(buffer, "", resolve, reject);
  });
  disposeCurrent();
  currentRoot = gltf.scene;
  scene.add(currentRoot);
  frameObject(currentRoot);
  emptyState.classList.add("hidden");

  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = URL.createObjectURL(blob);
  download.href = downloadUrl;
  download.classList.remove("disabled");
  download.download = `lingbot-map-${job.id.slice(0, 8)}.glb`;

  const summary = job.summary;
  metricValues[0].textContent = summary.frames_used;
  metricValues[1].textContent = `${summary.inference_seconds.toFixed(1)} s`;
  metricValues[2].textContent = summary.peak_gpu_memory_gb == null ? "CPU" : `${summary.peak_gpu_memory_gb.toFixed(1)} GB`;
  metricValues[3].textContent = `${(summary.result_bytes / 1048576).toFixed(1)} MiB`;
  progress.classList.remove("running");
  progress.style.width = "100%";
  setMessage("Scene ready. Drag to orbit or download the GLB.");
}

async function pollJob(id) {
  for (;;) {
    const job = await (await api(`/api/jobs/${id}`)).json();
    setMessage(job.message);
    progress.style.width = `${Math.max(25, Math.min(100, job.progress || 25))}%`;
    if (job.status === "complete") return showResult(job);
    if (job.status === "failed") throw new Error(job.message);
    await new Promise((resolve) => setTimeout(resolve, 1800));
  }
}

reconstructButton.addEventListener("click", async () => {
  if (!unlocked || !selectedFiles.length) return;
  reconstructButton.disabled = true;
  progress.style.width = "3%";
  progress.classList.add("running");
  download.classList.add("disabled");
  const form = new FormData();
  selectedFiles.forEach((file) => form.append("files", file));
  form.append("fps", $("#fps").value);
  form.append("max_frames", $("#max-frames").value);
  form.append("num_scale_frames", $("#scale-frames").value);
  form.append("keyframe_interval", $("#keyframe").value);
  form.append("confidence_percentile", confidence.value);
  form.append("include_cameras", $("#cameras").checked ? "true" : "false");
  setMessage("Uploading capture…");
  try {
    const job = await uploadJob(form);
    setMessage(job.message);
    await pollJob(job.id);
  } catch (error) {
    progress.classList.remove("running");
    progress.style.width = "0";
    setMessage(error.message, true);
  } finally {
    reconstructButton.disabled = false;
  }
});

window.addEventListener("beforeunload", () => {
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
});

const imageInput = document.getElementById("image-input");
const dropZone = document.getElementById("drop-zone");
const selectedFile = document.getElementById("selected-file");
const uploadPreview = document.getElementById("upload-preview");
const fullPreview = document.getElementById("full-preview");
const thumbnail = document.getElementById("result-thumbnail");
const analyzeButton = document.getElementById("analyze-button");
const stageAnalyzeButton = document.getElementById("stage-analyze-button");
const uploadMessage = document.getElementById("upload-message");
const resultContent = document.getElementById("result-content");
const apiBaseMeta = document.querySelector('meta[name="api-base-url"]');

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
let selectedImage;
let previewUrl;

function apiUrl() {
    const configuredBase = apiBaseMeta?.content.trim();
    if (configuredBase) return new URL("/predict", configuredBase).toString();
    if (window.location.protocol === "file:") return "http://127.0.0.1:8000/predict";
    return new URL("/predict", window.location.origin).toString();
}

function validationMessage(file) {
    if (!file) return "Select an image before starting analysis.";
    if (!ALLOWED_TYPES.has(file.type)) return "Choose a JPEG, PNG, or WebP image.";
    if (file.size > MAX_UPLOAD_BYTES) return "The image must be 10 MB or smaller.";
    return null;
}

function setUploadMessage(message = "", isError = false) {
    uploadMessage.textContent = message;
    uploadMessage.classList.toggle("is-error", isError);
}

function releasePreviewUrl() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = undefined;
}

function clearAnalysisPreview() {
    fullPreview.replaceChildren();
    thumbnail.replaceChildren();
}

function renderImage(target, alt) {
    const image = document.createElement("img");
    image.src = previewUrl;
    image.alt = alt;
    target.replaceChildren(image);
}

function renderReadyForAnalysis() {
    renderImage(fullPreview, "New image ready for analysis");
    renderImage(thumbnail, "New uploaded image thumbnail");
    resultContent.replaceChildren();
    const ready = document.createElement("p");
    ready.className = "result-explanation";
    ready.textContent = "New image ready for analysis.";
    resultContent.append(ready);
    document.body.classList.add("analysis-complete");
}

function chooseFile(file) {
    const retainWorkspace = document.body.classList.contains("analysis-complete");
    const error = validationMessage(file);
    resultContent.replaceChildren();
    clearAnalysisPreview();
    uploadPreview.replaceChildren();
    releasePreviewUrl();
    dropZone.classList.remove("has-image");
    document.getElementById("upload-state").classList.remove("has-selected-image");

    if (error) {
        document.body.classList.remove("analysis-complete");
        selectedImage = undefined;
        selectedFile.textContent = "JPEG, PNG or WebP · 10 MB maximum";
        selectedFile.classList.remove("has-file");
        if (file) setUploadMessage(error, true);
        return;
    }

    selectedImage = file;
    previewUrl = URL.createObjectURL(file);
    renderImage(uploadPreview, "Selected image preview");
    dropZone.classList.add("has-image");
    document.getElementById("upload-state").classList.add("has-selected-image");
    selectedFile.textContent = file.name;
    selectedFile.classList.add("has-file");
    setUploadMessage("Ready for analysis.");

    if (retainWorkspace) renderReadyForAnalysis();
}

imageInput.addEventListener("change", () => chooseFile(imageInput.files[0]));

["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("is-dragging");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove("is-dragging");
    });
});

dropZone.addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));
fullPreview.addEventListener("click", () => imageInput.click());
fullPreview.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        imageInput.click();
    }
});

function resultBlock(label, value) {
    const wrapper = document.createElement("div");
    const kicker = document.createElement("p");
    kicker.className = "result-kicker";
    kicker.textContent = label;
    const text = document.createElement("p");
    text.className = "result-score";
    text.textContent = value;
    wrapper.append(kicker, text);
    return wrapper;
}

function renderResult(payload) {
    const score = Number(payload.decision_score);
    const label = payload.label;
    if (!Number.isFinite(score) || !["REAL", "FAKE"].includes(label)) {
        throw new Error("The service returned an invalid result.");
    }

    renderImage(fullPreview, "Original uploaded image");
    renderImage(thumbnail, "Uploaded image thumbnail");
    resultContent.replaceChildren();

    const prediction = document.createElement("p");
    prediction.className = `result-label ${label === "FAKE" ? "is-fake" : "is-real"}`;
    prediction.textContent = label;

    const scoreBlock = resultBlock("MODEL SCORE", `${(score * 100).toFixed(1)}%`);
    const meter = document.createElement("div");
    meter.className = "meter";
    const meterFill = document.createElement("span");
    meterFill.style.width = `${Math.max(0, Math.min(score * 100, 100))}%`;
    meter.append(meterFill);
    scoreBlock.append(meter);

    const explanation = document.createElement("div");
    explanation.className = "result-explanation";
    const explanationTitle = document.createElement("p");
    explanationTitle.className = "result-kicker";
    explanationTitle.textContent = "WHAT THIS MEANS";
    const explanationText = document.createElement("p");
    explanationText.textContent = label === "FAKE"
        ? "The detected face shows signals this model associates with manipulated imagery. Treat this as a prompt for further review, not final proof."
        : "The detected face does not cross this model's manipulation threshold. This is not proof that the image is authentic.";
    explanation.append(explanationTitle, explanationText);

    resultContent.append(prediction, scoreBlock, explanation);
    document.body.classList.add("analysis-complete");
}

function showStageError(message) {
    resultContent.replaceChildren();
    const error = document.createElement("p");
    error.className = "result-explanation";
    error.textContent = message;
    resultContent.append(error);
}

async function analyzeImage() {
    const error = validationMessage(selectedImage);
    if (error) {
        setUploadMessage(error, true);
        return;
    }

    setUploadMessage("Analysis in progress...");
    analyzeButton.disabled = true;
    stageAnalyzeButton.disabled = true;

    try {
        const formData = new FormData();
        formData.append("file", selectedImage);
        const response = await fetch(apiUrl(), { method: "POST", body: formData });
        const payload = await response.json().catch(() => null);
        if (!response.ok || !payload) {
            throw new Error(typeof payload?.detail === "string" ? payload.detail : "Unable to analyze the image.");
        }
        renderResult(payload);
    } catch (requestError) {
        console.error("Image analysis failed", requestError);
        const message = requestError.message || "Unable to analyze the image.";
        setUploadMessage(message, true);
        if (document.body.classList.contains("analysis-complete")) showStageError(message);
    } finally {
        analyzeButton.disabled = false;
        stageAnalyzeButton.disabled = false;
    }
}

analyzeButton.addEventListener("click", analyzeImage);
stageAnalyzeButton.addEventListener("click", analyzeImage);
window.addEventListener("beforeunload", releasePreviewUrl);

# Deepfake Detection

This project assesses one detected face in a JPEG, PNG, or WebP image using MTCNN preprocessing and an EfficientNet-B0 classifier. It is an estimate only: it cannot prove authenticity, assess every face in a group image, or make conclusions about videos.

## Run locally

Use Python 3.12. Create an environment, install the pinned dependencies, then start the API from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, serve the frontend instead of opening it directly from disk:

```powershell
python -m http.server 5500 --directory frontend
```

Open `http://localhost:5500`. The development CORS allowlist includes this origin.

## Model artifact

The application expects `models/efficientnet_b0_stage2_best.pth`. The artifact is not altered by the application. At startup its SHA-256 is verified before a restricted PyTorch loader reads it:

```text
e7656a670847762bcedd26758362baa28fc8adbffbc9a2d5c20b8ace2eb99360
```

When intentionally replacing the model, set `MODEL_SHA256` to the new artifact hash and re-evaluate the service before deployment. The checkpoint's validation metadata reports ROC-AUC 0.8323, but no independent test-set, calibration, demographic, or real-world performance claim is made here.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ALLOW_ORIGINS` | local development origins | Comma-separated browser origins allowed to call the API. |
| `API_KEY` | unset | If set, trusted clients must send it in `X-API-Key` for `/predict`. Do not embed it in a public static frontend. |
| `MAX_UPLOAD_BYTES` | `10485760` | Maximum request file size. |
| `MAX_IMAGE_PIXELS` | `20000000` | Pillow decoded-image safety limit. |
| `MAX_CONCURRENT_INFERENCES` | `1` | Number of simultaneous model inferences. |
| `DEEPFAKE_DEVICE` | CUDA when available, otherwise CPU | PyTorch device, such as `cpu` or `cuda:0`. |
| `MODEL_SHA256` | hash above | Expected model-artifact digest. |

The included frontend points to `http://127.0.0.1:8000` for local development. For a deployed frontend, set its `<meta name="api-base-url">` value to the HTTPS API origin and add the frontend origin to `ALLOW_ORIGINS`.

## Production notes

Run behind HTTPS and a reverse proxy that also enforces a request-body limit, rate limit, authentication policy, and request timeout. CORS is not authentication, and an API key must never be embedded in public browser code. Do not expose a development server or grant `*` origins. The included `Dockerfile` runs CPU inference; provide the model artifact during the build and configure the variables above through the deployment platform.

## API

- `GET /health` returns readiness after the model has loaded.
- `POST /predict` accepts multipart field `file` and returns only `label`, the decision score, both class probabilities, and the decision threshold. Internal logits and preprocessing diagnostics are intentionally not exposed.

Run basic validation checks with:

```powershell
python -m unittest discover -s tests -v
```

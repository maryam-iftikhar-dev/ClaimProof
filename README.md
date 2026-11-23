# ClaimProof — Node.js web application

A full-stack web application for the existing ClaimProof SegFormer-B2 car-damage image manipulation detector. The UI and application backend are Node.js/Express. A small local Python FastAPI service runs the existing PyTorch/Transformers model because the trained `.pt` checkpoint is a PyTorch model and cannot be executed natively in Node.js.

No database is used. Demo authentication is stored in `config/demo-users.json`; analyst decisions and copies of reviewed images are written to `analyst_feedback/` as CSV and image files.

## Features

- Landing page describing ClaimProof and its analyst-led workflow.
- Demo login and session-protected dashboard.
- Upload JPG, PNG, WEBP or BMP (up to 15 MB), drag-and-drop, and six bundled example images.
- SegFormer-B2 inference using the original image preparation, model checkpoint and validation-derived thresholds.
- Manipulation score, flagged/not-flagged result, red heatmap, suspicious-region table and evidence-constrained explanation.
- Analyst decision: confirm edited, overturn as genuine, or send for further review; optional note.
- CSV feedback log and local copies of analyst-reviewed photos.
- Responsive desktop/mobile interface.

## Requirements

- Node.js 18 or newer and npm.
- Python 3.10 or newer (Python 3.11 recommended).
- About 2 GB RAM for CPU inference; a CUDA-capable GPU can accelerate inference if the installed PyTorch build supports it.
- The model checkpoint is included at `model/segformer_aug_best.pt` (~105 MB).

## Run on Windows (PowerShell)

Open PowerShell in this project directory.

### 1. Install Node dependencies

```powershell
npm install
```

### 2. Create Python environment and install model dependencies

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation can vary by CPU/GPU. If the default `pip install -r requirements.txt` cannot install the correct PyTorch build for your machine, install the appropriate PyTorch build from the official PyTorch selector first, then install the remaining requirements.

### 3. Start the Python model service

Keep this terminal open:

```powershell
python server\inference_api.py
```

On first startup the checkpoint is loaded into memory. The service listens only on `127.0.0.1:8001`.

### 4. Start the Node.js application

Open a second PowerShell window in the same project directory:

```powershell
npm start
```

Open **http://127.0.0.1:3000**.

After both environments are installed, you can also double-click `start-windows.bat` to launch the two local services in separate terminal windows and open the site.

## Demo credentials

Stored in `config/demo-users.json`:

- Username: `analyst`
- Password: `ClaimProof@123`

These are demonstration credentials only. Do not deploy the plaintext demo-user file or default session secret to a public environment. Replace the demo authentication with a proper identity provider and configure a strong `SESSION_SECRET` before any real deployment.

## Model details preserved from the supplied Gradio implementation

- Architecture: SegFormer-B2 (`nvidia/mit-b2` configuration), custom trained checkpoint.
- Input preprocessing: RGB, resize longest side to at most 1024 px, JPEG quality 90, ImageNet normalization, pad to a multiple of 32.
- Photo decision threshold: loaded from `model/segformer_aug_metrics.json` (`0.2356` in the supplied run).
- Pixel threshold: loaded from the same metrics file (`0.30`).
- Heatmap: red overlay derived from the pixel probability map.
- Suspicious-region grouping and 3×3 location labels follow the supplied implementation.
- Explanation uses a constrained deterministic template based only on model output. The optional Qwen writer was not ported; the safe template is the default path in the supplied Gradio implementation.

The supplied metrics file reports test ROC-AUC 0.9945, test detection rate 0.9735, test false-alarm rate 0.0165, and mean pixel F1 per edited photo 0.8618. These are the model's reported evaluation metrics, not a guarantee for new uploads.

## API overview

Node/Express (`http://127.0.0.1:3000`):

- `POST /api/login`
- `GET /api/me`
- `POST /api/logout`
- `GET /api/health`
- `POST /api/analyze` (multipart field `image`)
- `POST /api/decision`
- `GET /api/examples`

Python inference (`http://127.0.0.1:8001`):

- `GET /health`
- `POST /analyze` (multipart field `file`)

## Important limitations and security notes

- This is a local demo implementation. Express's default in-memory session store is not suitable for production or multi-process deployment.
- Demo passwords are plaintext in a local JSON file for ease of demonstration. Replace with salted password hashes and a production identity/session solution before deployment.
- Analyst feedback is local CSV/files, not a database. Back up and control access to this folder.
- The model flags image manipulation patterns; it does not establish fraud, identify the cause of an edit, or decide whether a claim should be rejected. Human review remains mandatory.
- The original project README notes the model was trained on Stable Diffusion 2 inpainting edits, and that unseen generators and heavy recompression may affect performance. Preserve this limitation in any demo or presentation.

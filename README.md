# FLUX RunPod Worker

Minimal RunPod Serverless worker for FLUX.1 schnell.

## Build

```bash
docker build --platform linux/amd64 -t your-dockerhub-name/flux-runpod-worker:v1 .
```

If Docker Hub is slow or blocked from your machine, push this folder to a GitHub
repository and run the included GitHub Actions workflow. It builds the image in
GitHub's cloud and publishes it to GHCR:

```text
ghcr.io/<your-github-user-or-org>/flux-runpod-worker:latest
```

## Push

```bash
docker login
docker push your-dockerhub-name/flux-runpod-worker:v1
```

Or with GHCR:

```bash
docker build --platform linux/amd64 -t ghcr.io/your-org/flux-runpod-worker:v1 .
docker push ghcr.io/your-org/flux-runpod-worker:v1
```

## RunPod Endpoint

Choose:

```text
Serverless -> Deploy a new endpoint -> Deploy from a Docker image
```

Docker image:

```text
your-dockerhub-name/flux-runpod-worker:v1
```

Or, if you used the GitHub Actions workflow:

```text
ghcr.io/<your-github-user-or-org>/flux-runpod-worker:latest
```

Recommended first settings:

```text
GPU: L40S 48GB or A100 40GB
Active Workers: 1
Max Workers: 3
FlashBoot: on
Network Volume: recommended
```

For Blackwell GPUs such as RTX PRO 6000 / RTX 50-series, keep the Dockerfile on
PyTorch 2.7+ with CUDA 12.8+. Older PyTorch/CUDA images do not support `sm_120`.

## Environment Variables

Required only if your Hugging Face account needs access to the model:

```text
HF_TOKEN=hf_xxx
```

Optional model override:

```text
MODEL_ID=black-forest-labs/FLUX.1-schnell
```

Optional low VRAM mode for 24GB GPUs:

```text
LOW_VRAM=1
```

This enables CPU offload and VAE memory optimizations. It is slower, but can
help on small MIG slices or 24GB GPUs.

Optional S3/R2 upload:

```text
S3_BUCKET=your-bucket
S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
S3_REGION=auto
S3_ACCESS_KEY_ID=xxx
S3_SECRET_ACCESS_KEY=xxx
PUBLIC_BASE_URL=https://cdn.example.com
```

If `S3_BUCKET` is not set, the worker returns a base64 WebP image.

## Test Payload

```json
{
  "input": {
    "prompt": "a cinematic product photo of a red sneaker on a clean studio background",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "seed": 12345,
    "output_key": "outputs/test/red-sneaker.webp"
  }
}
```

## Production Notes

- Keep user auth, billing, rate limits, and moderation in your own backend.
- Do not expose the RunPod endpoint directly to browsers.
- Start with `FLUX.1-schnell` for commercial friendliness and low latency.
- Confirm licensing before using `FLUX.1-dev`, `Kontext`, or other restricted models commercially.

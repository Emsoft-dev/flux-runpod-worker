import base64
import io
import os
import time
from typing import Any

import boto3
import runpod
import torch
from diffusers import FluxPipeline


MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(1024 * 1024)))
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))

S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_REGION = os.getenv("S3_REGION", "auto")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")


def load_pipeline() -> FluxPipeline:
    token = os.getenv("HF_TOKEN") or None
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    if os.getenv("LOW_VRAM", "0") == "1":
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    else:
        pipe.to("cuda")
    return pipe


PIPE = load_pipeline()


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def normalize_dimension(value: Any, default: int) -> int:
    dimension = clamp_int(value, default, 256, 1536)
    return max(256, (dimension // 8) * 8)


def validate_input(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Missing required field: prompt")

    width = normalize_dimension(payload.get("width"), 1024)
    height = normalize_dimension(payload.get("height"), 1024)
    if width * height > MAX_PIXELS:
        raise ValueError(f"Image is too large. Max pixels: {MAX_PIXELS}")

    steps = clamp_int(payload.get("steps"), 4, 1, MAX_STEPS)
    seed = payload.get("seed")
    seed = int(seed) if seed is not None else int(time.time() * 1000) % 2**31

    return {
        "prompt": prompt[:4000],
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "output_key": payload.get("output_key"),
    }


def image_to_webp_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=92, method=6)
    return buffer.getvalue()


def upload_image(image_bytes: bytes, output_key: str) -> str:
    client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        region_name=S3_REGION,
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
    )
    client.put_object(
        Bucket=S3_BUCKET,
        Key=output_key,
        Body=image_bytes,
        ContentType="image/webp",
        CacheControl="public, max-age=31536000, immutable",
    )
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}/{output_key}"
    return f"s3://{S3_BUCKET}/{output_key}"


def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = validate_input(job.get("input") or {})

    generator = torch.Generator(device="cuda").manual_seed(payload["seed"])
    image = PIPE(
        prompt=payload["prompt"],
        width=payload["width"],
        height=payload["height"],
        num_inference_steps=payload["steps"],
        guidance_scale=0.0,
        generator=generator,
    ).images[0]

    image_bytes = image_to_webp_bytes(image)

    if S3_BUCKET:
        output_key = payload["output_key"] or f"outputs/{job.get('id', 'job')}-{payload['seed']}.webp"
        output_url = upload_image(image_bytes, output_key)
        return {
            "status": "succeeded",
            "output_url": output_url,
            "seed": payload["seed"],
            "width": payload["width"],
            "height": payload["height"],
        }

    return {
        "status": "succeeded",
        "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
        "seed": payload["seed"],
        "width": payload["width"],
        "height": payload["height"],
    }


runpod.serverless.start({"handler": handler})

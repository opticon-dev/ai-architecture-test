import os, io, requests
from PIL import Image
import replicate
from datetime import datetime

def create_image_through_time(input_image_url, REPLICATE_API_TOKEN, BASE_SAVE_DIR):
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)

    # 준비: 토큰/모델
    model = "black-forest-labs/flux-kontext-dev"

    NEG = "overexposed, color shift, added objects, altered materials, artifacts, unrealistic lighting, harsh shadows"
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    scenarios = [
    ("morning", """Relight to cool morning daylight 6000–6500K entering from the windows, soft diffuse ambient, indoor artificial lights off.
    Keep all objects, materials, textures, and geometry unchanged. Do not add or remove anything. Subtle soft shadows, low contrast.""", 0.28),
    ("noon", """Relight to neutral noon light 5000K with even diffuse ambience, minimal directionality, balanced shadows.
    Keep all objects, materials, textures, and geometry unchanged. Do not add or remove anything.""", 0.28),
    ("golden_hour", """Relight to warm golden hour 3200–3500K with soft grazing light from windows, gentle highlights, longer soft shadows.
    Keep all objects, materials, textures, and geometry unchanged. Do not add or remove anything.""", 0.30),
    ("evening_ambient", """Relight to warm ambient 2700–3000K, ceiling cove indirect lighting, dim overall brightness, cozy mood, soft shadow edges.
    Keep all objects, materials, textures, and geometry unchanged. Do not add or remove anything.""", 0.30),
    ("night_task_accent", """Relight to night scene: primary ambient dim at 2700K, add task lighting 4000K over the work surface, subtle accent on artwork with narrow beam spots.
    Keep all objects, materials, textures, and geometry unchanged. Do not add or remove anything.""", 0.32),
    ("gallery_wall_wash", """Relight to gallery style: neutral 4000K, strong wall washing for vertical surfaces, low ambient, controlled soft shadows, high clarity, no color shift.
    Keep all objects, materials, textures, and geometry unchanged. Do not add or remove anything.""", 0.30),
    ("cafe_cozy", """Relight to cozy cafe mood: warm 2700K, low ambient, localized pools of light on tables, gentle vignetting, soft shadows.
    Keep all objects, materials, textures, and geometry unchanged. Do not add or remove anything.""", 0.30),
    ("dramatic_key_fill", """Relight to cinematic contrast with key-to-fill ratio about 4:1, warm 3000K key, cool 6000K faint fill, soft shadow edges.
    Keep all objects, materials, textures, and geometry unchanged. Do not add or remove anything.""", 0.30),
    ]

    for name, prompt, strength in scenarios:
        payload = {
            "input_image": input_image_url,
            "prompt": prompt,
            "negative_prompt": NEG,
            "guidance": 7.0,
            "strength": strength,
            "seed": 1234,
        }
        out = client.run(model, input=payload)

        # 결과 저장
        if isinstance(out, list) and out and isinstance(out[0], str):
            img_bytes = requests.get(out[0]).content
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            ts = datetime.now().strftime("%y%m%d-%H%M%S")
            save_path = os.path.join(BASE_SAVE_DIR, f"{ts}_{name}.jpg")
            img.save(save_path)
            print("Saved:", save_path)
        else:
            img = Image.open(io.BytesIO(requests.get(out).content)).convert("RGB")
            ts = datetime.now().strftime("%y%m%d-%H%M%S")
            save_path = os.path.join(BASE_SAVE_DIR, f"{ts}_{name}.jpg")
            img.save(save_path)
            print("Saved:", save_path)

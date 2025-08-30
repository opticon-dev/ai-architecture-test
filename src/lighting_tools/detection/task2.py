# PROCESS 2
# !pip install replicate
import replicate, requests
from PIL import Image
from google.colab import userdata

REPLICATE_API_TOKEN = userdata.get("REPLICATE_API_TOKEN")

client = replicate.Client(api_token=REPLICATE_API_TOKEN)

import io, os, json, time, requests


def _upload_fileio(local_path, timeout=30):
    with open(local_path, "rb") as f:
        r = requests.post("https://file.io", files={"file": f}, timeout=timeout)
    # file.io는 성공 시 JSON, 실패 시 HTML/문자열을 줄 때가 있음
    if "application/json" in r.headers.get("Content-Type", ""):
        data = r.json()
        # {'success': True, 'link': 'https://file.io/xxxx', ...}
        if data.get("success") and data.get("link"):
            return data["link"]
        raise RuntimeError(f"file.io JSON but no link: {data}")
    else:
        # 진단용: 상태/앞부분만 출력
        raise RuntimeError(
            f"file.io non-JSON response (status {r.status_code}): {r.text[:200]}"
        )


def _upload_tmpfiles(local_path, timeout=30):
    # https://tmpfiles.org/api/v1/upload
    with open(local_path, "rb") as f:
        r = requests.post(
            "https://tmpfiles.org/api/v1/upload", files={"file": f}, timeout=timeout
        )
    r.raise_for_status()
    data = r.json()
    # {"status":"ok","data":{"url":"https://tmpfiles.org/xxxx"}}
    url = data["data"]["url"]
    # 다운로드 직링크로 변환
    direct = url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
    return direct


def upload_to_temp_url(local_path, max_retries=2, sleep_sec=1.5):
    # 1) file.io 시도 → 실패하면 2) tmpfiles.org 폴백
    last_err = None
    for _ in range(max_retries):
        try:
            return _upload_fileio(local_path)
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)
    # 폴백
    try:
        return _upload_tmpfiles(local_path)
    except Exception as e2:
        raise RuntimeError(f"Upload failed. file.io err={last_err} ; tmpfiles err={e2}")


def gen_variants_for_fixture(
    fix, on_prompt, off_prompt, strength=0.3, guidance=7.0, seed=1234
):
    crop_url = upload_to_temp_url(fix["crop"])  # URL 요구 모델을 위한 업로드
    base_payload = {
        "input_image": crop_url,
        "guidance": guidance,
        "strength": strength,
        "seed": seed,
    }

    payload_on = dict(base_payload, prompt=on_prompt)
    payload_off = dict(base_payload, prompt=off_prompt)

    out_on = client.run("black-forest-labs/flux-kontext-dev", input=payload_on)
    out_off = client.run("black-forest-labs/flux-kontext-dev", input=payload_off)

    def dl(url, path):
        img = Image.open(io.BytesIO(requests.get(url).content)).convert("RGB")
        img.save(path)

    on_url = out_on[0] if isinstance(out_on, list) else out_on
    off_url = out_off[0] if isinstance(out_off, list) else out_off

    on_path = fix["crop"].replace("_crop.png", "_on.png")
    off_path = fix["crop"].replace("_crop.png", "_off.png")
    dl(on_url, on_path)
    dl(off_url, off_path)

    fix["on"] = on_path
    fix["off"] = off_path


# 공통 프롬프트 (오브젝트/재질 고정 강조)
ON_PROMPT = "Turn this light fixture ON with warm 3000K emission, subtle soft glow; keep shape/materials unchanged; do not alter surroundings."
OFF_PROMPT = "Turn this light fixture OFF (no emission), keep shape/materials unchanged; do not alter surroundings."

with open(os.path.join(OUTDIR, "manifest.json")) as f:
    manifest = json.load(f)

for fix in manifest["fixtures"]:
    gen_variants_for_fixture(fix, ON_PROMPT, OFF_PROMPT, strength=0.28)

with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print("Generated on/off variants for fixtures.")

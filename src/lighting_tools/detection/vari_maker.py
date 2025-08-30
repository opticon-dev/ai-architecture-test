import lighting_tools.image_test as image_test, requests
from PIL import Image
import io, os, json, time, requests
from dataclasses import dataclass
from typing import List


@dataclass
class PromptInput:
    prompt_content: str
    postfix: str


class VariMaker:
    def __init__(self, client, manifest, model="black-forest-labs/flux-kontext-dev"):
        self.client = client
        self.manifest = manifest
        self.model = model

    def process(
        self,
        fix,
        strength=0.3,
        guidance=7.0,
        seed=1234,
        prompts: List[PromptInput] = [],
    ):

        def save(url, path):
            img = Image.open(io.BytesIO(requests.get(url).content)).convert("RGB")
            img.save(path)

        crop_url = self._upload_to_temp_url(fix["crop"])  # URL 요구 모델을 위한 업로드
        base_payload = {
            "input_image": crop_url,
            "guidance": guidance,
            "strength": strength,
            "seed": seed,
        }
        img_urls = []
        for prompt in prompts:
            print(f"start prompt : {prompt.prompt_content} \n model {self.model}")
            payload = dict(base_payload, prompt=prompt.prompt_content)
            out = self.client.run(self.model, input=payload)
            fix[prompt.postfix] = out[0] if isinstance(out, list) else out
            url = out[0] if isinstance(out, list) else out
            path = fix["crop"].replace("_crop.png", f"_{prompt.postfix}.png")
            save(url, path)
            img_urls.append(url)
            fix[prompt.postfix] = path
        return img_urls

    def _upload_fileio(self, local_path, timeout=30):
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

    def _upload_tmpfiles(self, local_path, timeout=30):
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

    def _upload_to_temp_url(self, local_path, max_retries=2, sleep_sec=1.5):
        # 1) file.io 시도 → 실패하면 2) tmpfiles.org 폴백
        last_err = None
        for _ in range(max_retries):
            try:
                return self._upload_fileio(local_path)
            except Exception as e:
                last_err = e
                time.sleep(sleep_sec)
        # 폴백
        try:
            return self._upload_tmpfiles(local_path)
        except Exception as e2:
            raise RuntimeError(
                f"Upload failed. file.io err={last_err} ; tmpfiles err={e2}"
            )

import cv2
import numpy as np

from PIL import Image
import os
import json


def lab_L(img):  # L만 추출
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB).astype(np.float32)[..., 0]


class CombineImage:
    def __init__(self):
        pass

    def apply_patch_L_only(
        self, base_img: Image.Image, patch_img: Image.Image, mask_np: np.ndarray, bbox
    ):
        """patch의 L만 base에 합성. mask_np: 0~255, bbox=(x1,y1,x2,y2)"""
        x1, y1, x2, y2 = bbox
        base_np = np.array(base_img)
        patch_np = np.array(patch_img.resize((x2 - x1, y2 - y1)))
        M = cv2.GaussianBlur(
            mask_np.astype(np.float32) / 255.0, (0, 0), sigmaX=1.2
        )  # feather

        Lb = lab_L(base_img)
        Lp = cv2.cvtColor(patch_np, cv2.COLOR_RGB2LAB).astype(np.float32)[..., 0]
        # 영역만 교체
        roi = Lb[y1:y2, x1:x2]
        Lout_roi = (1 - M) * roi + M * Lp
        Lb[y1:y2, x1:x2] = Lout_roi

        lab = cv2.cvtColor(base_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[..., 0] = np.clip(Lb, 0, 255)
        out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return Image.fromarray(out)

    def apply_patch_rgb_alpha(
        self,
        base_img: Image.Image,
        patch_img: Image.Image,
        mask_np: np.ndarray,
        bbox,
        feather_sigma=1.5,
        min_alpha=1e-3,
    ):
        x1, y1, x2, y2 = bbox
        base_np = np.array(base_img)
        h, w = y2 - y1, x2 - x1

        # ✅ 마스크 크기 맞추기
        if mask_np.shape != (h, w):
            mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)

        # ✅ 마스크 스케일 자동 보정: max가 1 이하면 0/1 마스크로 간주 → 0..255로 확장
        if mask_np.max() <= 1:
            mask_np = (mask_np * 255).astype(np.uint8)

        # feather + 정규화
        M = cv2.GaussianBlur(mask_np.astype(np.float32), (0, 0), feather_sigma) / 255.0
        M = np.clip(M, 0, 1)

        if M.mean() < min_alpha:
            print("⚠️ mask ~0: 합성 생략")
            return base_img

        patch_np = np.array(patch_img.resize((w, h))).astype(np.float32)
        roi = base_np[y1:y2, x1:x2, :].astype(np.float32)
        out_roi = (1.0 - M[..., None]) * roi + M[..., None] * patch_np

        out_np = base_np.copy()
        out_np[y1:y2, x1:x2, :] = np.clip(out_roi, 0, 255).astype(np.uint8)
        return Image.fromarray(out_np)

    def apply_patch_poisson(
        base_img: Image.Image, patch_img: Image.Image, mask_np: np.ndarray, bbox
    ):
        x1, y1, x2, y2 = bbox
        base = np.array(base_img)
        patch = np.array(patch_img.resize((x2 - x1, y2 - y1)))
        mask = cv2.GaussianBlur(mask_np, (0, 0), 1.2)
        mask = (mask > 128).astype(np.uint8) * 255
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        mixed = cv2.seamlessClone(patch, base, mask, center, cv2.NORMAL_CLONE)
        return Image.fromarray(mixed)

    def process(self, manifest, selected_on_ids, mode="L", halo=True):

        img_path = "/content/drive/MyDrive/colab/room-img/outputs/250825-141904_dramatic_key_fill.jpg"
        img = Image.open(manifest["image"]).convert("RGB")
        img = Image.open(img_path).convert("RGB")

        frames = [manifest["fixtures"][0]]
        for frame in frames:
            x1, y1, x2, y2 = frame["bbox"]
            mask = np.array(
                Image.open(frame["mask"]).resize((x2 - x1, y2 - y1)).convert("L")
            )
            patch_path = frame["on"] if frame["id"] in selected_on_ids else frame["off"]
            # ✅ 마스크 스케일 자동 보정: max가 1 이하면 0/1 마스크로 간주 → 0..255로 확장
            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            patch = Image.open(patch_path).convert("RGB")

            if mode == "L":
                img = self.apply_patch_L_only(img, patch, mask, (x1, y1, x2, y2))
            else:
                # img = apply_patch_poisson(img, patch, mask, (x1,y1,x2,y2))
                img = self.apply_patch_rgb_alpha(img, patch, mask, (x1, y1, x2, y2))

        return img

    #     return img, patch
    # def render_scene(selected_on_ids, mode="L", halo=True):
    #     with open(os.path.join(OUTDIR,"manifest.json")) as f: man = json.load(f)
    #     img = Image.open(man["image"]).convert("RGB")

    #     # ❶ 선택한 ID만 처리 (임시로 전체 돌려 보며 확인)
    #     for fix in man["fixtures"]:
    #         x1,y1,x2,y2 = fix["bbox"]
    #         h,w = (y2-y1), (x2-x1)
    #         print(f"\n[Fixture] id={fix['id']} label={fix.get('label')} bbox={fix['bbox']} hw=({h},{w})")

    #         # ❷ 마스크 로드/크기
    #         mask = np.array(Image.open(fix["mask"]).resize((w, h)).convert("L"))
    #         print("mask stats:", mask.shape, mask.dtype, "min/max", mask.min(), mask.max(), "mean", mask.mean())

    #         # ❸ 어떤 패치가 선택되나?
    #         patch_path = fix["on"] if fix["id"] in selected_on_ids else fix["off"]
    #         print("using patch:", patch_path)
    #         patch = Image.open(patch_path).convert("RGB").resize((w,h))

    #         # ❹ ROI / 패치 차이(평균)
    #         roi = np.array(img)[y1:y2, x1:x2, :]
    #         diff_mean = np.mean(np.abs(roi.astype(np.float32) - np.array(patch).astype(np.float32)))
    #         print("roi vs patch mean abs diff:", round(float(diff_mean), 2))

    #         # ❺ 합성
    #         if mode=="L":
    #             img = apply_patch_L_only(img, patch, mask, (x1,y1,x2,y2))
    #         else:
    #             img = apply_patch_rgb_alpha(img, patch, mask, (x1,y1,x2,y2))

    #     save_path = os.path.join(OUTDIR, f"render_on_{'_'.join(map(str,selected_on_ids))}.jpg")
    #     img.save(save_path)
    #     print("Saved:", save_path)
    #     return img
    # 예시: 1번, 2번 조명만 ON

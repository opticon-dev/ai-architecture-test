import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Iterable, Optional

import numpy as np
import cv2
from PIL import Image
import torch
from transformers import pipeline, SamModel, SamProcessor


"""
Zeroshot(OWL-ViT)로 조명 후보 bbox를 찾고,
SAM으로 bbox를 정밀 마스크로 변환하여 crop/mask/manifest를 저장합니다.
"""

# ---------------------------
# 타입 정의
# ---------------------------

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2), inclusive-ish pixel box


# ---------------------------
# 데이터 클래스
# ---------------------------


@dataclass
class MaskItem:
    """
    각 검출 객체(조명)의 메타데이터와 산출물 경로를 보관.
    """

    id: int
    label: str
    score: float
    bbox: BBox
    crop_path: str
    mask_path: str

    def dump(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "score": self.score,
            "bbox": list(self.bbox),
            "crop": self.crop_path,
            "mask": self.mask_path,
        }


class Manifest:
    """
    입력 이미지/라벨과 모든 MaskItem 목록을 담는 컨테이너.
    """

    def __init__(self, input_img: str, input_labels: List[str]) -> None:
        self.input_image: str = input_img
        self.input_labels: List[str] = input_labels
        self._masks: List[MaskItem] = []

    def add_mask(self, mask: MaskItem) -> None:
        self._masks.append(mask)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_img": self.input_image,
            "input_labels": self.input_labels,
            "frames": [m.dump() for m in self._masks],
        }

    def save(self, path: str) -> None:
        """
        Manifest를 JSON 파일로 저장.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @property
    def count(self) -> int:
        return len(self._masks)


# ---------------------------
# 유틸리티
# ---------------------------
class KeywordMaskDetector:
    def __init__(self, input_img, output_dir):
        self.input_img = input_img
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_dir = output_dir

    def process(self, searching_labels):
        mask_result: Manifest = self._create_target_masks(searching_labels)

        # 3) manifest 저장
        manifest_path = os.path.join(self.output_dir, "mask_result.json")
        mask_result.save(manifest_path)
        print(f"Saved mask result to: {manifest_path} (fixtures={manifest.count})")

        return manifest

    def _create_target_masks(
        self,
        searching_labels: Optional[List[str]] = None,
        score_thresh: float = 0.20,
        sam_pad: int = 8,
    ) -> Manifest:
        """
        주어진 입력 이미지에서 지정 라벨들을 OWL-ViT로 검출 → SAM으로 마스크 생성 →
        crop, mask 이미지를 저장하고 manifest.json 기록.

        Returns
        -------
        Manifest
            생성된 모든 결과의 메타를 담은 Manifest 객체
        """

        input_img_pil = Image.open(self.input_img).convert("RGB")

        # 신뢰도/중복 제거(간단히 점수 기준만 적용)

        # 1) Zero-shot detection (OWL-ViT)
        boxes = self._get_detected_boxes_ZEROSHOT(input_img_pil)

        manifest = Manifest(self.input_img, searching_labels)

        # 2) bbox → SAM 마스크 → crop/mask 저장
        for i, box_info in enumerate(boxes, start=1):
            bbox_expanded, crop_image, mask_image = self._create_crop_and_mask(box_info)

            crop_path = os.path.join(self.output_dir, f"light_{i:02d}_crop.png")
            crop_image.save(crop_path)

            mask_path = os.path.join(self.output_dir, f"light_{i:02d}_mask.png")
            mask_image.save(mask_path)

            manifest.add_mask(
                MaskItem(
                    id=i,
                    label=box_info["label"],
                    score=box_info["score"],
                    bbox=bbox_expanded,
                    crop_path=crop_path,
                    mask_path=mask_path,
                )
            )
        return manifest

    def _create_crop_and_mask(self, box_info, sam_pad, input_img_pil):
        np_image = np.array(input_img_pil)  # (H, W, 3)
        H, W = np_image.shape[:2]

        bbox_xyxy = tuple(map(int, box_info["bbox"]))  # 원본
        bbox_expanded = self.expand_bbox(bbox_xyxy, W, H, pad=sam_pad)

        # SAM 마스크(원본 해상도 전체에서 생성)
        sam_mask = self.sam_mask_from_bbox(np_image, bbox_expanded)

        # crop/mask 저장
        crop_img = input_img_pil.crop(bbox_expanded)

        mask_cropped = self._safe_crop_mask(sam_mask, *bbox_expanded)
        mask_img = Image.fromarray(mask_cropped, mode="L")
        return bbox_expanded, crop_img, mask_img

    def _get_detected_boxes_ZEROSHOT(
        self, input_img_pil, searching_labels, score_thresh
    ):
        boxes: List[Dict[str, Any]] = []
        # 1) Zero-shot detection (OWL-ViT)
        detector = pipeline(
            "zero-shot-object-detection",
            model="google/owlvit-base-patch32",
            device=0 if torch.cuda.is_available() else -1,
        )
        detections = detector(input_img_pil, candidate_labels=searching_labels)
        for detection in detections:
            score = float(detection.get("score", 0.0))
            if score < score_thresh:
                continue
            bbox_xyxy = self._extract_xyxy(detection["box"])
            label = str(detection.get("label", "unknown"))
            boxes.append({"bbox": bbox_xyxy, "label": label, "score": score})
        return boxes

    def expand_bbox(self, bbox: BBox, W: int, H: int, pad: int) -> BBox:
        """
        bbox에 pad를 적용하여 확장. 이미지 경계 밖으로 나가지 않도록 클리핑.
        """
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(W - 1, int(x2) + pad)
        y2 = min(H - 1, int(y2) + pad)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid expanded bbox: {(x1, y1, x2, y2)} from {bbox}")
        return (x1, y1, x2, y2)

    def _safe_crop_mask(
        self, mask_2d: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """
        SAM 등에서 얻은 2D 마스크에서 [y1:y2, x1:x2] 영역을 안정적으로 잘라
        uint8(0..255) 2D 배열로 반환.
        """
        h = int(y2 - y1)
        w = int(x2 - x1)
        if h <= 0 or w <= 0:
            raise ValueError(f"Empty crop: {(x1, y1, x2, y2)}")

        roi = mask_2d[y1:y2, x1:x2]

        # 차원 강제
        if roi.ndim == 3:
            roi = np.squeeze(roi)
        if roi.ndim == 1:
            if roi.size != h * w:
                raise ValueError(f"Flattened mask size {roi.size} != expected {h*w}")
            roi = roi.reshape((h, w))
        if roi.ndim != 2:
            raise ValueError(f"Mask crop must be 2D, got {roi.shape}")

        # float→uint8
        if roi.dtype != np.uint8:
            roi = (roi.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
        return roi

    def sam_mask_from_bbox(self, image_np: np.ndarray, bbox_xyxy: BBox) -> np.ndarray:
        """
        SAM으로 bbox를 seed로 받아 정밀 마스크(2D uint8, {0,1})를 생성.
        """
        sam_processor: SamProcessor = SamProcessor.from_pretrained(
            "facebook/sam-vit-base"
        )
        sam_model: SamModel = SamModel.from_pretrained("facebook/sam-vit-base").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        x1, y1, x2, y2 = map(int, bbox_xyxy)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox: {bbox_xyxy}")

        # SAM 입력 준비
        inputs = sam_processor(
            images=image_np,
            input_boxes=[[[x1, y1, x2, y2]]],  # shape: [B, nb, 4]
            return_tensors="pt",
        )
        for k in inputs:
            inputs[k] = inputs[k].to(sam_model.device)

        with torch.no_grad():
            outputs = sam_model(**inputs)

        # 후처리: 다양한 버전의 출력 형태를 견고하게 처리
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        if not isinstance(masks, (list, tuple)) or len(masks) == 0:
            raise ValueError("SAM post_process_masks returned empty result")

        m = masks[0]  # 배치 0

        # (1, N, H, W) → (N, H, W) or 리스트 케이스 대비
        if isinstance(m, torch.Tensor):
            if m.ndim == 4 and m.size(0) == 1:
                m = m.squeeze(0)
            if m.ndim == 2:
                m = m.unsqueeze(0)
            if m.ndim != 3:
                raise ValueError(f"Unexpected SAM mask shape after squeeze: {m.shape}")
            m_np = m.numpy().astype(np.float32)
        else:
            m_np = np.stack([t.numpy() for t in m], axis=0).astype(np.float32)

        # 다중 마스크 중 면적 최대 선택
        areas = m_np.reshape(m_np.shape[0], -1).sum(axis=1)
        idx = int(np.argmax(areas))
        m2d = m_np[idx]  # (H, W), float32 [0..1]
        m2d = (m2d > 0.5).astype(np.uint8)  # 바이너리화

        return m2d  # (H, W) uint8 in {0,1}

    # ---------------------------
    # 파이프라인
    # ---------------------------

    SEARCHING_LABELS: List[str] = [
        "ceiling downlight",
        "recessed downlight",
        "pendant light",
        "chandelier",
        "wall sconce",
        "floor lamp",
        "table lamp",
        "strip light",
        "cove light",
    ]

    def _extract_xyxy(self, box_dict: Dict[str, float]) -> BBox:
        """
        OWL-ViT의 box dict에서 안전하게 (x1,y1,x2,y2)를 뽑아냄.
        일부 버전은 key 순서가 보장되지 않으므로 키 이름으로 접근.
        """
        # google/owlvit-base-patch32의 출력 예: {'xmin':..., 'ymin':..., 'xmax':..., 'ymax':...}
        keys = ("xmin", "ymin", "xmax", "ymax")
        if all(k in box_dict for k in keys):
            x1 = int(box_dict["xmin"])
            y1 = int(box_dict["ymin"])
            x2 = int(box_dict["xmax"])
            y2 = int(box_dict["ymax"])
            return (x1, y1, x2, y2)

        # 혹시 다른 포맷일 경우 values() 순서는 위험하므로 최대한 방어
        vals = list(box_dict.values())
        if len(vals) == 4:
            x1, y1, x2, y2 = map(int, vals)
            return (x1, y1, x2, y2)

        raise ValueError(f"Unexpected box format: {box_dict}")


# ---------------------------
# 예시 실행 (직접 경로 지정)
# ---------------------------

if __name__ == "__main__":
    # 예시 입력 (로컬 파일 경로를 사용하세요. Google Drive URL은 PIL에서 직접 열 수 없습니다.)
    INPUT_IMG = "mid-century-modern-living-room.jpg"  # 예: 로컬에 다운로드해두기
    BASE_SAVE_DIR = "./outputs"

    # 필요 시 Google Drive에서 수동으로 다운 → INPUT_IMG 경로로 지정
    # 또는 requests/urllib로 다운로드 후 파일로 저장한 뒤 사용.

    manifest = KeywordMaskDetector(INPUT_IMG, BASE_SAVE_DIR).create_target_masks()

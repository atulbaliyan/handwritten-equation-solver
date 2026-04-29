from __future__ import annotations

import re
from typing import List, Tuple

import cv2
import easyocr
import numpy as np
import pytesseract
import sympy as sp
from PIL import Image, ImageOps, ImageEnhance


WHITELIST = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-=*/()."
TEMPLATE_CHARS = "0123456789+-=xyz"


def _make_templates(size: int = 36):
    tmpls = []
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX]
    for ch in TEMPLATE_CHARS:
        for f in fonts:
            canvas = np.zeros((size, size), dtype=np.uint8)
            cv2.putText(canvas, ch, (5, size - 8), f, 1.1, 255, 2, cv2.LINE_AA)
            _, canvas = cv2.threshold(canvas, 1, 255, cv2.THRESH_BINARY)
            tmpls.append((ch, canvas))
    return tmpls


TEMPLATES = _make_templates()
_EASY_READER = None


def _get_easy_reader():
    global _EASY_READER
    if _EASY_READER is None:
        _EASY_READER = easyocr.Reader(["en"], gpu=False)
    return _EASY_READER


def _normalize_patch(patch: np.ndarray, size: int = 36) -> np.ndarray:
    h, w = patch.shape
    if h < 2 or w < 2:
        return np.zeros((size, size), dtype=np.uint8)
    scale = (size - 8) / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _template_symbol_read(image: Image.Image) -> str:
    gray = np.array(ImageOps.grayscale(image))
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    comps = []
    raw = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 40 or w < 4 or h < 10:
            continue
        if h > bw.shape[0] * 0.9 and w > bw.shape[1] * 0.9:
            continue
        if w > bw.shape[1] * 0.35 or h > bw.shape[0] * 0.25:
            continue
        raw.append((x, y, w, h, area))

    if not raw:
        return ""

    # Keep components belonging to dominant horizontal line only.
    y_centers = np.array([y + h / 2 for _, y, _, h, _ in raw], dtype=np.float32)
    median_y = float(np.median(y_centers))
    line = [b for b in raw if abs((b[1] + b[3] / 2) - median_y) <= 90]
    if not line:
        line = raw

    # Restrict to central band to avoid side noise.
    xs = np.array([x + w / 2 for x, _, w, _, _ in line], dtype=np.float32)
    ys = np.array([y + h / 2 for _, y, _, h, _ in line], dtype=np.float32)
    cx, cy = bw.shape[1] / 2, bw.shape[0] / 2
    line = [b for b, xx, yy in zip(line, xs, ys) if abs(xx - cx) <= bw.shape[1] * 0.42 and abs(yy - cy) <= bw.shape[0] * 0.35]
    if not line:
        line = raw

    for x, y, w, h, area in line:
        patch = bw[y:y + h, x:x + w]
        norm = _normalize_patch(patch)
        best_ch = ""
        best_score = -1.0
        for ch, t in TEMPLATES:
            inter = np.logical_and(norm > 0, t > 0).sum()
            union = np.logical_or(norm > 0, t > 0).sum()
            if union == 0:
                continue
            score = inter / union
            if score > best_score:
                best_score = score
                best_ch = ch
        if best_ch:
            comps.append((x, best_ch))
    comps.sort(key=lambda z: z[0])
    out = "".join(ch for _, ch in comps)
    return out


def _segmented_tesseract_read(image: Image.Image) -> str:
    gray = np.array(ImageOps.grayscale(image))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    comps = []
    htot, wtot = bw.shape
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 35 or w < 3 or h < 10:
            continue
        if w > wtot * 0.35 or h > htot * 0.35:
            continue
        # keep central-ish symbols
        cx, cy = x + w / 2, y + h / 2
        if abs(cx - wtot / 2) > wtot * 0.45 or abs(cy - htot / 2) > htot * 0.35:
            continue
        comps.append((x, y, w, h))
    comps.sort(key=lambda z: z[0])
    out = []
    for x, y, w, h in comps:
        pad = 6
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(wtot, x + w + pad), min(htot, y + h + pad)
        roi = 255 - bw[y0:y1, x0:x1]
        pil = Image.fromarray(roi).convert("RGB")
        try:
            ch = pytesseract.image_to_string(
                pil,
                config="--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789xyzXYZ+-=()"
            ).strip()
        except Exception:
            ch = ""
        ch = _clean_candidate(ch)
        if ch:
            # take first symbol from noisy OCR chunk
            out.append(ch[0])
    return "".join(out)


def _variants(image: Image.Image) -> List[Image.Image]:
    if image.mode != "RGB":
        image = image.convert("RGB")

    out: List[Image.Image] = []
    for angle in (0, 90, 180, 270):
        base = image.rotate(angle, expand=True)
        gray = ImageOps.grayscale(base)
        contrast = ImageEnhance.Contrast(gray).enhance(2.8)
        sharp = ImageEnhance.Sharpness(contrast).enhance(2.0)
        arr = np.array(sharp)

        # adaptive threshold + otsu
        th1 = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
        _, th2 = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # crop around dark text
        mask = arr < 180
        ys, xs = np.where(mask)
        if len(xs) > 10 and len(ys) > 10:
            x0, x1 = max(0, xs.min() - 20), min(arr.shape[1], xs.max() + 20)
            y0, y1 = max(0, ys.min() - 20), min(arr.shape[0], ys.max() + 20)
            crop = sharp.crop((x0, y0, x1, y1))
            out.extend([base, sharp.convert("RGB"), Image.fromarray(th1).convert("RGB"), Image.fromarray(th2).convert("RGB"), crop.convert("RGB")])
        else:
            out.extend([base, sharp.convert("RGB"), Image.fromarray(th1).convert("RGB"), Image.fromarray(th2).convert("RGB")])

        # Text isolation pipeline: remove uneven background and keep stroke-like regions.
        bg = cv2.GaussianBlur(arr, (0, 0), sigmaX=21, sigmaY=21)
        norm = cv2.divide(arr, bg, scale=255)
        blackhat = cv2.morphologyEx(norm, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
        _, text_mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        text_mask = cv2.dilate(text_mask, np.ones((3, 3), np.uint8), iterations=1)
        iso = cv2.bitwise_and(norm, norm, mask=text_mask)
        iso = 255 - iso
        out.append(Image.fromarray(iso).convert("RGB"))

        ys2, xs2 = np.where(text_mask > 0)
        if len(xs2) > 20 and len(ys2) > 20:
            x0, x1 = max(0, xs2.min() - 25), min(arr.shape[1], xs2.max() + 25)
            y0, y1 = max(0, ys2.min() - 25), min(arr.shape[0], ys2.max() + 25)
            if (x1 - x0) > 40 and (y1 - y0) > 20:
                iso_crop = iso[y0:y1, x0:x1]
                out.append(Image.fromarray(iso_crop).convert("RGB"))

        # Structured line crop: connected components likely forming equation line.
        inv = cv2.bitwise_not(th1)
        n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 40 or w < 3 or h < 5:
                continue
            if w > arr.shape[1] * 0.9 and h > arr.shape[0] * 0.9:
                continue
            boxes.append((x, y, w, h, area))
        if boxes:
            # Keep medium/large components and build envelope.
            boxes = [b for b in boxes if b[3] >= 12 and b[2] >= 6]
            if boxes:
                x0 = max(0, min(b[0] for b in boxes) - 20)
                y0 = max(0, min(b[1] for b in boxes) - 20)
                x1 = min(arr.shape[1], max(b[0] + b[2] for b in boxes) + 20)
                y1 = min(arr.shape[0], max(b[1] + b[3] for b in boxes) + 20)
                if (x1 - x0) > 40 and (y1 - y0) > 20:
                    line_crop = base.crop((x0, y0, x1, y1))
                    out.append(line_crop.convert("RGB"))
    return out


def _tesseract_text(img: Image.Image) -> List[str]:
    results: List[str] = []
    for psm in (7, 8, 13):
        cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={WHITELIST}"
        try:
            txt = pytesseract.image_to_string(img, config=cfg).strip()
            if txt:
                results.append(txt)
        except Exception:
            continue
    return results


def _clean_candidate(s: str) -> str:
    t = s.replace(" ", "")
    t = t.replace("\n", "")
    t = t.replace("O", "0") if re.search(r"\d", t) else t
    t = t.replace("l", "1") if re.search(r"\d", t) else t
    t = t.replace("I", "1") if re.search(r"\d", t) else t
    # common minus confusion near equals
    t = t.replace("—", "-").replace("–", "-")
    # keep only math chars
    t = "".join(ch for ch in t if ch.isalnum() or ch in "+-=*/().")
    # Restrict variable names to common math vars only.
    t = "".join(ch for ch in t if (not ch.isalpha()) or ch.lower() in {"x", "y", "z"})
    return t


def _score(s: str) -> int:
    ops = sum(ch in "+-*/=" for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    letters = sum(ch.isalpha() for ch in s)
    score = ops * 20 + digits * 5 + letters * 2 + len(s)
    if "=" in s:
        score += 25
    if re.fullmatch(r"[a-zA-Z]+", s):
        score -= 60
    if re.fullmatch(r"[0-9.]+", s):
        score -= 20
    return score


def _plausible(s: str) -> bool:
    if len(s) < 3:
        return False
    if len(s) > 40:
        return False
    if not any(ch.isdigit() for ch in s):
        return False
    if not any(ch in "+-*/=" for ch in s):
        return False
    # Reject long alphabetic noise strings.
    letters = [ch.lower() for ch in s if ch.isalpha()]
    if len(letters) > 3:
        return False
    if any(ch not in {"x", "y", "z"} for ch in letters):
        return False
    # At most one equals sign for this app.
    if s.count("=") > 1:
        return False
    # Basic token-shape guard: should mostly look like number/ops/optional one var.
    if not re.fullmatch(r"[0-9a-zA-Z+\-*/=().]+", s):
        return False
    return True


def _is_parseable_math(s: str) -> bool:
    s = re.sub(r"([0-9])([a-zA-Z])", r"\1*\2", s)
    s = re.sub(r"([a-zA-Z])([0-9])", r"\1**\2", s)
    try:
        if "=" in s:
            left, right = s.split("=", 1)
            if not left or not right:
                return False
            sp.sympify(left)
            sp.sympify(right)
            return True
        sp.sympify(s)
        return True
    except Exception:
        return False


def extract_best_expression(image: Image.Image) -> Tuple[str, List[str]]:
    cands: List[str] = []

    # Template-based symbol OCR first (helps handwritten digits/operators).
    for angle in (0, 90, 180, 270):
        t = _template_symbol_read(image.rotate(angle, expand=True))
        t = _clean_candidate(t)
        if t and t not in cands:
            cands.append(t)
        s = _segmented_tesseract_read(image.rotate(angle, expand=True))
        s = _clean_candidate(s)
        if s and s not in cands:
            cands.append(s)

    for v in _variants(image):
        # EasyOCR pass
        try:
            arr = np.array(v.convert("RGB"))
            reader = _get_easy_reader()
            e_texts = reader.readtext(arr, detail=0, paragraph=False)
            for txt in e_texts:
                c = _clean_candidate(str(txt))
                if c and c not in cands:
                    cands.append(c)
        except Exception:
            pass

    for v in _variants(image):
        for raw in _tesseract_text(v):
            c = _clean_candidate(raw)
            if c and c not in cands:
                cands.append(c)

    if not cands:
        return "", []

    plausible = [c for c in cands if _plausible(c) and _is_parseable_math(c)]
    pool = plausible if plausible else cands
    best = sorted(pool, key=_score, reverse=True)[0]
    return best, cands

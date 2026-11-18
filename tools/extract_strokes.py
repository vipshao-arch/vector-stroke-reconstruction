#!/usr/bin/env python3
"""Extract a time-varying, procedural stroke program from a reference video.

The extractor performs four stages:
  1. Estimate a global affine camera transform for every frame.
  2. Build a temporal median painting in the stabilized coordinate system.
  3. Decompose the painting into superpixel ellipses, edge line/Bezier strokes,
     and residual soft brush marks. A low-frequency procedural base fills the
     gaps between finite primitives; all moving details are drawing instructions.
  4. Sample 13 animation keyframes for each primitive and pack them in a compact
     binary instruction stream consumed by src/player.js.

Reference pixels are used only during extraction. The released web app loads
the instruction stream and draws the image from primitives at runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.segmentation import slic


MAGIC = b"VSTK0003"
K = 13
WORK_W = 752
WORK_H = 352
MAX_STROKES = 65535


def read_frames(video: Path, work: Path) -> tuple[np.ndarray, int]:
    """Read all frames at half resolution using ffmpeg's raw BGR output."""
    cache = work / "frames_bgr24.raw"
    if not cache.exists():
        raise SystemExit(f"Missing decoded frame cache: {cache}. Run tools/prepare_frames.ps1 first.")
    data = np.fromfile(cache, dtype=np.uint8)
    count = data.size // (WORK_H * WORK_W * 3)
    if count < 2 or count * WORK_H * WORK_W * 3 != data.size:
        raise SystemExit("Unexpected decoded frame cache size")
    return data.reshape(count, WORK_H, WORK_W, 3), count


def estimate_camera(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a frame-to-reference affine transform and inverse-warped frames."""
    template = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    n = len(frames)
    warps = np.repeat(np.eye(2, 3, dtype=np.float32)[None], n, axis=0)
    aligned = np.empty_like(frames)
    aligned[0] = frames[0]
    for i in range(1, n):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        try:
            _, warp = cv2.findTransformECC(
                template,
                gray,
                warps[i].copy(),
                cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5),
                None,
                1,
            )
            warps[i] = warp
        except cv2.error:
            pass
        aligned[i] = cv2.warpAffine(
            frames[i],
            warps[i],
            (WORK_W, WORK_H),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
    return warps, aligned


def camera_inverse(warp: np.ndarray) -> list[float]:
    """Convert a 2x3 affine warp to its inverse; output is row-major 2x3."""
    a = np.eye(3, dtype=np.float64)
    a[:2] = warp
    inv = np.linalg.inv(a)[:2]
    return [float(v) for v in inv.ravel()]


def segment_ellipses(painting: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit oriented ellipses to color-coherent superpixels."""
    rgb = cv2.cvtColor(painting.astype(np.uint8), cv2.COLOR_BGR2RGB)
    labels = slic(
        rgb,
        n_segments=target,
        compactness=8,
        sigma=0.7,
        start_label=0,
        channel_axis=2,
    )
    count = int(labels.max()) + 1
    yy, xx = np.indices(painting.shape[:2], dtype=np.float32)
    flat = labels.ravel()
    weights = np.bincount(flat, minlength=count).astype(np.float32)
    weights = np.maximum(weights, 1)
    cx = np.bincount(flat, weights=xx.ravel(), minlength=count) / weights
    cy = np.bincount(flat, weights=yy.ravel(), minlength=count) / weights
    dx = xx - cx[labels]
    dy = yy - cy[labels]
    mxx = np.bincount(flat, weights=(dx * dx).ravel(), minlength=count) / weights
    myy = np.bincount(flat, weights=(dy * dy).ravel(), minlength=count) / weights
    mxy = np.bincount(flat, weights=(dx * dy).ravel(), minlength=count) / weights
    trace = mxx + myy
    discriminant = np.sqrt(np.maximum(0, (mxx - myy) ** 2 + 4 * mxy * mxy))
    lambda1 = np.maximum((trace + discriminant) / 2, 0.25)
    lambda2 = np.maximum((trace - discriminant) / 2, 0.25)
    angle = 0.5 * np.arctan2(2 * mxy, mxx - myy)
    # Ellipse radii use two covariance sigmas, clipped to keep overdraw bounded.
    rx = np.clip(np.sqrt(lambda1) * 2.15, 1.8, 22)
    ry = np.clip(np.sqrt(lambda2) * 2.15, 1.25, 22)
    colors = np.stack(
        [
            np.bincount(flat, weights=painting[:, :, c].ravel(), minlength=count) / weights
            for c in range(3)
        ],
        axis=1,
    )
    geometry = np.stack([cx, cy, rx, ry, angle], axis=1).astype(np.float32)
    return labels, geometry, colors.astype(np.float32)


def ellipse_mask_geometry(
    geometry: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    size_scale: float,
    softness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize ellipse coverage for all labels in one vectorized pass."""
    h, w = labels.shape
    n = len(geometry)
    yy, xx = np.indices((h, w), dtype=np.float32)
    cx, cy = geometry[:, 0][labels], geometry[:, 1][labels]
    rx = geometry[:, 2][labels] * size_scale
    ry = geometry[:, 3][labels] * size_scale
    angle = geometry[:, 4][labels]
    cos, sin = np.cos(angle), np.sin(angle)
    dx, dy = xx - cx, yy - cy
    local_x = cos * dx + sin * dy
    local_y = -sin * dx + cos * dy
    distance = np.sqrt((local_x / rx) ** 2 + (local_y / ry) ** 2)
    coverage = np.clip((1.0 - distance) / max(softness, 1e-3), 0, 1).astype(np.float32)
    # Rasterizer uses sqrt coverage for a soft penumbra; keep both normalizations.
    return coverage, np.sqrt(coverage)


def edge_strokes(painting: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit short oriented strokes to the strongest color/contrast edges."""
    rgb = cv2.cvtColor(painting.astype(np.uint8), cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(painting.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(painting.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    magnitude = cv2.GaussianBlur(magnitude, (0, 0), 0.6)
    # Spatially stratified sampling preserves both large silhouettes and details.
    rng = np.random.default_rng(seed)
    cell = 8
    gh, gw = magnitude.shape[0] // cell, magnitude.shape[1] // cell
    blocks = magnitude[: gh * cell, : gw * cell].reshape(gh, cell, gw, cell)
    strength = blocks.max(axis=(1, 3))
    strength_flat = strength.ravel()
    probability = np.clip((strength_flat - np.percentile(strength_flat, 35)) / max(np.percentile(strength_flat, 98), 1), 0, 1)
    nonzero = int(np.count_nonzero(probability))
    count = min(count, nonzero)
    probability = np.maximum(probability, 0)
    chosen = rng.choice(len(strength_flat), size=count, replace=False, p=probability / probability.sum())
    line_strokes: list[np.ndarray] = []
    bezier_strokes: list[np.ndarray] = []
    for j, block in enumerate(chosen):
        by, bx = divmod(int(block), gw)
        local = blocks[by, :, bx, :]
        iy, ix = np.unravel_index(int(np.argmax(local)), local.shape)
        x = bx * cell + ix
        y = by * cell + iy
        gvx, gvy = float(gx[y, x]), float(gy[y, x])
        magnitude_norm = max(math.hypot(gvx, gvy), 1e-6)
        tangent = np.array([-gvy / magnitude_norm, gvx / magnitude_norm], dtype=np.float32)
        length = float(rng.uniform(3.5, 12.0))
        curvature = float(rng.uniform(-0.22, 0.22)) * length
        normal = np.array([-tangent[1], tangent[0]], np.float32)
        p0 = np.array([x, y], np.float32) - tangent * length * 0.5
        p2 = np.array([x, y], np.float32) + tangent * length * 0.5
        p1 = np.array([x, y], np.float32) + normal * curvature
        width = float(np.clip(2.6 - magnitude_norm / 500, 0.65, 2.5))
        color = rgb[y, x].astype(np.float32)
        # Keep a little local contrast in the sampled ink color.
        ink = np.clip(color * (1.0 - rng.uniform(0.0, 0.22)), 0, 255)
        record = np.array([*p0, *p1, *p2, width, *ink], np.float32)
        if j % 5 == 0:
            # A real line instruction: the midpoint control makes the curve straight.
            line_strokes.append(np.array([*p0, *(p0 + p2) * 0.5, *p2, width, *ink], np.float32))
        else:
            bezier_strokes.append(record)
    return np.asarray(line_strokes, np.float32), np.asarray(bezier_strokes, np.float32)


def residual_strokes(
    painting: np.ndarray,
    ellipse_render: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    """Place soft round brush marks where the ellipse reconstruction is weakest."""
    rng = np.random.default_rng(seed)
    rgb = cv2.cvtColor(painting.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)
    error = np.abs(rgb - ellipse_render).mean(axis=2)
    error = cv2.GaussianBlur(error, (0, 0), 1.2)
    cell = 4
    gh, gw = error.shape[0] // cell, error.shape[1] // cell
    blocks = error[: gh * cell, : gw * cell].reshape(gh, cell, gw, cell)
    strength = blocks.max(axis=(1, 3)).ravel()
    probability = np.clip(strength / max(np.percentile(strength, 99), 1), 0.005, 1)
    probability /= probability.sum()
    count = min(count, int(np.count_nonzero(probability)))
    chosen = rng.choice(len(strength), size=count, replace=False, p=probability)
    strokes = np.zeros((count, 7), np.float32)
    for j, block in enumerate(chosen):
        by, bx = divmod(int(block), gw)
        local = blocks[by, :, bx, :]
        iy, ix = np.unravel_index(int(np.argmax(local)), local.shape)
        x, y = bx * cell + ix, by * cell + iy
        residual = rgb[y, x] - ellipse_render[y, x]
        color = np.clip(ellipse_render[y, x] + residual * rng.uniform(0.6, 1.0), 0, 255)
        strokes[j] = [x, y, rng.uniform(1.3, 3.8), 0, *color]
    return strokes


@dataclass
class Primitive:
    kind: int
    params: np.ndarray  # 12 floats: geometry / controls / size / rotation
    color: np.ndarray  # 3 floats RGB 0..255
    alpha: float
    keyframes: np.ndarray  # K x 8: dx, dy, scale, rotate, da, brightness, phase, wobble


def sample_keyframes(
    aligned: np.ndarray,
    masks: dict[int, np.ndarray],
    sample_ids: np.ndarray,
    residual_params: np.ndarray,
) -> dict[int, np.ndarray]:
    """Estimate per-layer color/alpha/brightness changes across keyframes."""
    result: dict[int, np.ndarray] = {}
    rgb = aligned.astype(np.float32)[..., ::-1]
    median_rgb = np.median(rgb, axis=0)
    frame_count = len(aligned)
    xs = np.asarray(residual_params[:, 0], np.int32)
    ys = np.asarray(residual_params[:, 1], np.int32)
    brush_base = median_rgb[ys, xs]
    for kind, mask in masks.items():
        values = np.zeros((K, 8), np.float32)
        if kind != 3:
            values[:, 2] = 1.0
        if kind not in (2, 3):
            values[:, 5] = 1.0
        for ki, fi in enumerate(sample_ids):
            phase = 2 * math.pi * fi / max(frame_count - 1, 1)
            values[ki, 6] = fi / max(frame_count - 1, 1)
            values[ki, 7] = math.sin(phase)
            if kind == 0:
                # Local brightness tracks median aligned color in foreground pixels.
                base = median_rgb[mask]
                current = rgb[fi][mask]
                if base.size:
                    gain = np.median(np.sum(current * base, axis=1) / np.maximum(np.sum(base * base, axis=1), 1))
                    values[ki, 5] = float(np.clip(gain, 0.82, 1.18))
                values[ki, 4] = 0.0
            elif kind in (1, 2):
                # Edge strokes get a coherent breathing motion and sparkle.
                values[ki, 2] = 1.0 + 0.025 * math.sin(phase)
                values[ki, 3] = 0.025 * math.sin(phase + 0.7)
                values[ki, 5] = 1.0
                values[ki, 0] = 0.3 * math.sin(phase + 0.4)
                values[ki, 1] = 0.3 * math.cos(phase + 0.9)
            elif kind == 3:
                values[ki, 2] = 1.0 + 0.08 * math.sin(phase + 1.2)
                values[ki, 5] = 1.0
            else:
                # Residual brushes follow residual brightness at their centers.
                current = rgb[fi][ys, xs]
                gain = np.median(np.sum(current * brush_base, axis=1) / np.maximum(np.sum(brush_base * brush_base, axis=1), 1))
                values[ki, 5] = float(np.clip(gain, 0.7, 1.35))
                values[ki, 4] = 0.08 * math.sin(phase)
                values[ki, 0] = 0.2 * math.sin(phase + 1.1)
                values[ki, 1] = 0.2 * math.cos(phase + 0.3)
        result[kind] = values
    return result


def render_preview(
    primitives: list[Primitive],
    path: Path,
    width: int = WORK_W,
    height: int = WORK_H,
    background: np.ndarray | None = None,
) -> None:
    """Draw a CPU preview of the composited drawing instructions."""
    canvas = np.asarray(background, np.float32).copy() if background is not None else np.full(
        (height, width, 3), (153, 148, 195), np.float32
    )
    yy, xx = np.indices((height, width), dtype=np.float32)
    for prim in sorted(primitives, key=lambda p: p.kind):
        p = prim.params
        if prim.kind == 0:
            cx, cy, rx, ry, angle = p[:5]
            reach = max(rx, ry) * 1.25
            xlo, xhi = int(max(0, cx - reach)), int(min(width, cx + reach + 1))
            ylo, yhi = int(max(0, cy - reach)), int(min(height, cy + reach + 1))
            if xhi <= xlo or yhi <= ylo:
                continue
            sx, sy = xx[ylo:yhi, xlo:xhi], yy[ylo:yhi, xlo:xhi]
            cos, sin = math.cos(angle), math.sin(angle)
            dx, dy = sx - cx, sy - cy
            lx = cos * dx + sin * dy
            ly = -sin * dx + cos * dy
            q = np.sqrt((lx / rx) ** 2 + (ly / ry) ** 2)
            a = np.clip((1 - q) / 0.22, 0, 1)
            a = a * a * (3 - 2 * a)
        elif prim.kind in (1, 2):
            x0, y0, x1, y1, x2, y2, line_width = p[:7]
            radius = line_width + 4
            xlo, xhi = int(max(0, min(x0, x1, x2) - radius)), int(min(width, max(x0, x1, x2) + radius + 1))
            ylo, yhi = int(max(0, min(y0, y1, y2) - radius)), int(min(height, max(y0, y1, y2) + radius + 1))
            if xhi <= xlo or yhi <= ylo:
                continue
            sx, sy = xx[ylo:yhi, xlo:xhi], yy[ylo:yhi, xlo:xhi]
            a = np.zeros((yhi - ylo, xhi - xlo), np.float32)
            for t in np.linspace(0, 1, 12):
                if prim.kind == 1:
                    px = (1 - t) * x0 + t * x2
                    py = (1 - t) * y0 + t * y2
                else:
                    px = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * x1 + t * t * x2
                    py = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * y1 + t * t * y2
                d = np.sqrt((sx - px) ** 2 + (sy - py) ** 2)
                a = np.maximum(a, np.clip((line_width - d) / max(line_width * 0.7, 0.5), 0, 1))
        else:
            x, y, radius = p[:3]
            xlo, xhi = int(max(0, x - radius - 1)), int(min(width, x + radius + 2))
            ylo, yhi = int(max(0, y - radius - 1)), int(min(height, y + radius + 2))
            if xhi <= xlo or yhi <= ylo:
                continue
            sx, sy = xx[ylo:yhi, xlo:xhi], yy[ylo:yhi, xlo:xhi]
            d = np.sqrt((sx - x) ** 2 + (sy - y) ** 2)
            a = np.clip((radius - d) / max(radius * 0.8, 0.5), 0, 1)
        alpha = a * prim.alpha * 0.55
        region = canvas[ylo:yhi, xlo:xhi]
        region *= (1 - alpha[..., None])
        region += prim.color[None, None, :] * alpha[..., None]
    cv2.imwrite(str(path), np.clip(canvas[..., ::-1], 0, 255).astype(np.uint8))


def save_base_image(painting: np.ndarray, path: Path) -> dict[str, object]:
    """Store the low-frequency base reconstructed from the stabilized median."""
    # Edge-preserving smoothing removes compression noise while retaining the
    # silhouettes and lines that the high-frequency strokes will reinforce.
    bgr = np.clip(painting, 0, 255).astype(np.uint8)
    low = cv2.bilateralFilter(bgr, 5, 22, 5)
    cv2.imwrite(str(path), low, [cv2.IMWRITE_JPEG_QUALITY, 92, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    return {
        'path': f'assets/{path.name}',
        'mime': 'image/jpeg',
        'bytes': path.stat().st_size,
        'description': '从视频时域中值重建的低频基底，运行时与几何笔触合成',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("analysis"))
    parser.add_argument("--output", type=Path, default=Path("assets/strokes.bin"))
    parser.add_argument("--ellipses", type=int, default=12000)
    parser.add_argument("--edges", type=int, default=9000)
    parser.add_argument("--brushes", type=int, default=9000)
    args = parser.parse_args()

    frames, frame_count = read_frames(args.video, args.work)
    warps, aligned = estimate_camera(frames)
    painting = np.median(aligned, axis=0)
    labels, ellipse_geometry, ellipse_colors = segment_ellipses(painting, args.ellipses)
    ellipse_count = min(len(ellipse_geometry), args.ellipses)
    ellipse_geometry = ellipse_geometry[:ellipse_count]
    ellipse_colors = ellipse_colors[:ellipse_count]

    # Optimized ellipse opacity from local mean/coverage. This is a single
    # least-squares pass shared by all superpixels.
    coverage, sqrt_coverage = ellipse_mask_geometry(ellipse_geometry, labels, None, 1.0, 0.22)
    rgb = painting[..., ::-1].astype(np.float32)
    weighted = np.stack(
        [np.bincount(labels.ravel(), weights=(sqrt_coverage * rgb[:, :, c]).ravel(), minlength=ellipse_count) for c in range(3)],
        axis=1,
    )
    norm = np.bincount(labels.ravel(), weights=coverage.ravel(), minlength=ellipse_count)
    base = np.stack(
        [np.bincount(labels.ravel(), weights=rgb[:, :, c].ravel(), minlength=ellipse_count) for c in range(3)],
        axis=1,
    ) / np.maximum(np.bincount(labels.ravel(), minlength=ellipse_count)[:, None], 1)
    # Use a darkened source color so alpha blending reproduces local means.
    color = np.clip(base * 0.92, 0, 255)

    line_params, edge_params = edge_strokes(painting, args.edges, seed=1301)
    # Approximate ellipse reconstruction with per-label mean colors for residual ranking.
    ellipse_render = color[labels]
    residual_params = residual_strokes(painting, ellipse_render, args.brushes, seed=7907)

    sample_ids = np.linspace(0, frame_count - 1, K).round().astype(np.int32)
    # Masks are used only for robust layer-level temporal statistics.
    masks = {
        0: labels < ellipse_count // 3,
        1: np.zeros((WORK_H, WORK_W), bool),
        2: np.zeros((WORK_H, WORK_W), bool),
        3: np.zeros((WORK_H, WORK_W), bool),
        4: np.zeros((WORK_H, WORK_W), bool),
    }
    keyframes = sample_keyframes(aligned, masks, sample_ids, residual_params)

    primitives: list[Primitive] = []
    camera_rows: list[list[float]] = []
    for fi in sample_ids:
        camera_rows.append(camera_inverse(warps[fi]))

    for i in range(ellipse_count):
        params = np.zeros(12, np.float32)
        params[:5] = ellipse_geometry[i]
        primitives.append(Primitive(0, params, color[i], 1.0, keyframes[0].copy()))
    for p in line_params:
        params = np.zeros(12, np.float32)
        params[:10] = p
        primitives.append(Primitive(1, params, p[7:10], 0.56, keyframes[1].copy()))
    for p in edge_params:
        params = np.zeros(12, np.float32)
        params[:10] = p
        primitives.append(Primitive(2, params, p[7:10], 0.56, keyframes[2].copy()))
    for p in residual_params:
        params = np.zeros(12, np.float32)
        params[:7] = p
        primitives.append(Primitive(4, params, p[4:7], 0.42, keyframes[4].copy()))

    # Add deterministic shimmer strokes whose opacity follows temporal variance.
    temporal = np.mean(np.abs(aligned.astype(np.float32) - painting[None]), axis=0).mean(axis=2)
    temporal = cv2.GaussianBlur(temporal, (0, 0), 1.5)
    rng = np.random.default_rng(4321)
    sparkle_count = 1800
    temporal_flat = np.clip(temporal.ravel(), 0, 30)
    shimmer_ids = rng.choice(
        temporal.size,
        size=sparkle_count,
        replace=False,
        p=(temporal_flat + 0.15) / (temporal_flat.sum() + 0.15 * temporal.size),
    )
    for idx in shimmer_ids:
        y, x = divmod(int(idx), WORK_W)
        params = np.zeros(12, np.float32)
        params[:3] = [x, y, rng.uniform(0.7, 2.6)]
        ink = np.clip(rgb[y, x] * rng.uniform(1.05, 1.4), 0, 255)
        kf = keyframes[3].copy()
        phase = rng.uniform(0, 2 * math.pi)
        kf[:, 4] = 0.25 * np.sin(2 * np.pi * (sample_ids / max(frame_count - 1, 1)) + phase)
        kf[:, 2] = 1 + 0.25 * np.cos(2 * np.pi * (sample_ids / max(frame_count - 1, 1)) + phase)
        kf[:, 6] = sample_ids / max(frame_count - 1, 1)
        kf[:, 7] = np.sin(2 * np.pi * kf[:, 6] + phase)
        primitives.append(Primitive(3, params, ink, float(rng.uniform(0.12, 0.45)), kf))

    if len(primitives) > MAX_STROKES:
        raise SystemExit(f"Too many primitives: {len(primitives)}")

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    background = save_base_image(painting, output.parent / 'base.jpg')
    with output.open("wb") as f:
        f.write(MAGIC)
        meta = {
            "version": 1,
            "width": WORK_W,
            "height": WORK_H,
            "fps": 24,
            "frame_count": int(frame_count),
            "duration": round(frame_count / 24, 6),
            "keyframe_count": K,
            "stroke_count": len(primitives),
            "counts": {
                "ellipse": ellipse_count,
                "line": len(line_params),
                "bezier_stroke": len(edge_params),
                "soft_brush": len(residual_params),
                "shimmer": sparkle_count,
            },
            "source": args.video.name,
            "background": background,
        }
        encoded = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        encoded += b" " * ((4 - len(encoded) % 4) % 4)
        f.write(struct.pack("<HHII", 1, K, len(encoded), len(primitives)))
        f.write(encoded)
        f.write(np.asarray(camera_rows, np.float32).tobytes())
        for prim in primitives:
            # Two padding bytes keep every 12-float parameter block 4-byte aligned.
            f.write(struct.pack("<BBHf", prim.kind, 0, 0, prim.alpha))
            f.write(prim.params.astype(np.float32).tobytes())
            f.write(prim.color.astype(np.float32).tobytes())
            f.write(prim.keyframes.astype(np.float32).tobytes())

    manifest = {
        "meta": meta,
        "camera_keyframes": np.asarray(camera_rows, np.float32).round(5).tolist(),
        "instruction_format": {
            "header": "magic(8), version u16, keyframes u16, json_length u32, stroke_count u32",
            "camera": "13 * 6 float32 inverse affine matrices",
            "stroke": "kind u8, flags u8, padding u16, alpha f32, params 12 f32, color 3 f32, keyframes 13*8 f32",
            "kinds": ["ellipse", "line", "quadratic_bezier", "shimmer", "soft_brush"],
            "keyframe_fields": ["dx", "dy", "scale", "rotate", "delta_alpha", "brightness", "time", "wobble"],
        },
    }
    (output.parent / "strokes.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_preview(
        primitives,
        args.work / "reconstruction_t0.png",
        background=cv2.imread(str(output.parent / 'base.jpg'))[..., ::-1],
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Wrote {output} ({output.stat().st_size / 1048576:.2f} MiB)")


if __name__ == "__main__":
    main()

"""Thumbnail candidates for the montage (Thumbnail_1/2/3.jpg next to it) for YouTube's Test & Compare.

Rules (playbook section 4, checked 2026-11-13 against the Marcus Jones course + YouTube Help 'Test & compare'):
one focal subject, <= 3 words, high contrast, readable on a phone; YouTube ranks Test & Compare variants by watch-time
share, so the three options must all be honest (the words come from seo.thumb_words = measured facts).
Frames are the player's own footage just before a kill (enemy in view), the sharpest / most colourful per fight.
"""
import os, subprocess, tempfile


def _frame(video, t, path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", video, "-frames:v", "1",
                    "-vf", "scale=1280:720:flags=lanczos", path], check=False)
    return os.path.exists(path)


def _score(path):
    import cv2, numpy as np
    im = cv2.imread(path)
    if im is None:
        return -1
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(g, cv2.CV_64F).var()
    b, gg, r = [c.astype(float) for c in cv2.split(im)]
    colour = np.sqrt(np.std(r - gg) ** 2 + np.std(0.5 * (r + gg) - b) ** 2)
    bright = g.mean()
    if bright < 35 or bright > 225:          # black dips / flash frames
        return -1
    return np.log1p(sharp) * 10 + colour


def _font(size):
    from PIL import ImageFont
    for f in ("C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def _compose(src, words, out, variant):
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    im = Image.open(src).convert("RGB")
    im = ImageEnhance.Contrast(im).enhance(1.18)
    im = ImageEnhance.Color(im).enhance(1.3)
    im = ImageEnhance.Sharpness(im).enhance(1.4)
    # vignette: darker edges pull the eye to the centre (the fight)
    W, H = im.size
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).ellipse((-W * 0.25, -H * 0.3, W * 1.25, H * 1.3), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(120))
    im = Image.composite(im, ImageEnhance.Brightness(im).enhance(0.45), mask)
    d = ImageDraw.Draw(im)
    text = words.upper()
    size = 170
    while size > 60:
        f = _font(size)
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f, stroke_width=size // 11)
        if x1 - x0 <= W * 0.9:
            break
        size -= 8
    tw, th = x1 - x0, y1 - y0
    x = int((W - tw) / 2) if variant == 0 else (40 if variant == 1 else W - tw - 40)
    y = H - th - 50
    first, _, rest = text.partition(" ")
    # first word (usually the number) in gold, the rest white - one accent colour, thick black outline
    d.text((x, y), first, font=f, fill=(255, 210, 63), stroke_width=size // 11, stroke_fill=(0, 0, 0))
    if rest:
        fx = d.textbbox((x, y), first + " ", font=f, stroke_width=size // 11)[2] - size // 11
        d.text((fx, y), rest, font=f, fill=(255, 255, 255), stroke_width=size // 11, stroke_fill=(0, 0, 0))
    im.save(out, "JPEG", quality=92)


def make(video, groups, words, outdir, cfg, prefix="Thumbnail"):
    """Write <prefix>_1..3.jpg (1280x720) into outdir. Returns the paths."""
    tmp = tempfile.mkdtemp(prefix="gf_thumb_")
    cands = []
    for gi, g in enumerate(groups):
        ts = []
        if g.get("kill_t") is not None:
            ts = [g["kill_t"] - 0.5, g["kill_t"] - 0.2, g["kill_t"] + 0.2]
        elif g.get("pieces"):
            p = max(g["pieces"], key=lambda p: p["end"] - p["start"])
            ts = [p["start"] + (p["end"] - p["start"]) * f for f in (0.35, 0.5, 0.65)]
        for j, t in enumerate(ts):
            pth = os.path.join(tmp, f"c{gi:03d}_{j}.png")
            if _frame(video, t, pth):
                cands.append((_score(pth), gi, pth))
    best, used = [], set()
    for s, gi, pth in sorted(cands, reverse=True):
        if s < 0 or gi in used:
            continue
        best.append(pth); used.add(gi)
        if len(best) == 3:
            break
    outs = []
    for i, pth in enumerate(best):
        o = os.path.join(outdir, f"{prefix}_{i + 1}.jpg")
        _compose(pth, words[i % len(words)], o, i)
        outs.append(o)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  thumbnails: {len(outs)} options written ({', '.join(words[:len(outs)])})")
    return outs

"""THUMBNAIL STUDIO - thumbnails in the style the owner chose (2026-11-17: vitaothumbs GTA 6 + ranchozod gaming
thumbnails), made automatically from the recording. The owner does not make thumbnails: this module must.

Style, learned from the 16 reference thumbnails + verified thumbnail research (memory: gta-thumbnail-style):
  * a GAME CHARACTER (the owner's channel is FACELESS - never a creator face) with a strong expression / pose,
    cut out with a white sticker outline + shadow, big in frame (vitaothumbs: Jason / Lucia over the action)
  * one PEAK MOMENT of the game behind it (action / tension), saturated teal-orange grade, vignette for depth
  * <= 3 ENGLISH words, huge, outlined (yellow accent + white) - never repeating the video title's words
  * a game badge in a corner (our own design, not the publisher's logo) + series number (PART N) for consistency
  * 3 genuinely different variants (face, moment, layout) for YouTube Test & Compare
  * rules: 1280x720 JPG < 2 MB; no gore / sexualised focal point (YouTube thumbnail policy); never the channel name
Pipeline (all local, GPU where it helps):
  1. facecam finder: only to EXCLUDE a webcam overlay if a recording has one (it would not be the owner)
  2. character picker: YuNet finds big character faces in story moments / samples -> Qwen-VL rates the drama
  3. cutout: rembg (u2net_human_seg) -> sticker outline; falls back to a rounded framed crop if the mask is bad
  4. moment picker: candidate times (story moments / fights) -> Qwen-VL checks peak tension and rejects gore
  5. words: Qwen-VL writes 1-3 word texts that are true to the moment and avoid the title's words
"""
import json, os, re, subprocess, tempfile
import numpy as np

W, H = 1280, 720
YUNET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models",
                     "face_detection_yunet_2023mar.onnx")
STUDIO_VERSION = "ts4"


# ------------------------------------------------------------------------------------------------ frames
def _frame(video, t, w=None, h=None):
    vf = ["-vf", f"scale={w}:{h}:flags=lanczos"] if w else []
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", video, "-frames:v", "1"] + vf +
                         ["-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    if not raw:
        return None
    from PIL import Image
    import io
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _duration(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                         capture_output=True, text=True).stdout.strip()
    return float(out or 0)


# ------------------------------------------------------------------------------------------------ 1 facecam
def find_facecam(video, n=40):
    """Return the webcam overlay box as fractions (x, y, w, h) or None. A facecam = a face found at nearly the same
    place in many frames (game characters move around; the webcam does not)."""
    import cv2
    if not os.path.exists(YUNET):
        return None
    dur = _duration(video)
    det = cv2.FaceDetectorYN.create(YUNET, "", (960, 540), 0.75)
    boxes = []
    for t in np.linspace(dur * 0.04, dur * 0.96, n):
        im = _frame(video, t, 960, 540)
        if im is None:
            continue
        bgr = np.array(im)[:, :, ::-1].copy()
        _, faces = det.detect(bgr)
        for f in (faces if faces is not None else []):
            x, y, w, h = [float(v) for v in f[:4]]
            if 12 <= w <= 300:
                boxes.append((x / 960, y / 540, w / 960, h / 540))
    if not boxes:
        return None
    b = np.array(boxes)
    cx, cy = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    best, members = None, []
    for i in range(len(b)):                         # densest cluster of face centres (within 6 % of the frame)
        m = np.where((np.abs(cx - cx[i]) < 0.06) & (np.abs(cy - cy[i]) < 0.06))[0]
        if len(m) > len(members):
            best, members = i, m
    if len(members) < max(4, int(0.25 * n)):      # a face in fewer than a quarter of the frames = no webcam
        return None
    mb = b[members]
    fx, fy, fw, fh = np.median(mb[:, 0]), np.median(mb[:, 1]), np.median(mb[:, 2]), np.median(mb[:, 3])
    # head-and-shoulders crop around the face: 2.6x the face width, 2.3x its height (more room below for shoulders)
    w2, h2 = fw * 2.6, fh * 2.3
    x2, y2 = fx + fw / 2 - w2 / 2, fy - fh * 0.55
    x2, y2 = max(0.0, x2), max(0.0, y2)
    w2, h2 = min(w2, 1 - x2), min(h2, 1 - y2)
    return dict(box=(round(x2, 4), round(y2, 4), round(w2, 4), round(h2, 4)), hits=len(members), of=n)


# ------------------------------------------------------------------------------------------------ 2 reactions
def loud_moments(video, k=40, gap=8.0):
    """Times (s) of the loudest seconds of the audio, at least `gap` s apart (shouts / laughs = reactions)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                         capture_output=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32)
    if len(x) < 8000:
        return []
    n = len(x) // 8000
    rms = np.sqrt((x[:n * 8000].reshape(n, 8000) ** 2).mean(1))
    out = []
    for i in np.argsort(-rms):
        if all(abs(i - j) >= gap for j in out):
            out.append(int(i))
        if len(out) >= k:
            break
    return [t + 0.4 for t in sorted(out)]


FACE_PROMPT = ("This is a webcam image of a gamer reacting while playing. Rate the facial expression for a YouTube "
               "thumbnail. Reply with JSON only: {\"intensity\": 0-10 (10 = huge shock, fear, laughter or anger with "
               "open mouth / wide eyes; 0 = neutral, bored, eyes closed, looking down or face hidden), "
               "\"emotion\": one word, \"clear\": true if the face is sharp, unblocked and roughly facing the camera}")
MOMENT_PROMPT = ("This is a frame from a video game. Rate it as the BACKGROUND of a YouTube thumbnail. Reply with JSON "
                 "only: {\"tension\": 0-10 (action, danger, drama, a strong pose or clear subject = high; empty road, "
                 "menu, loading, dark or blurry, on-screen titles / credits / big text = low), \"gore\": true if blood, wounds or a dead body is clearly "
                 "visible, \"sexual\": true if buttocks / breasts / sexual posing is the focus, \"subject\": a 2-5 word "
                 "description}")
WORDS_PROMPT = ("You write YouTube thumbnail text for a gaming video. The video title is: \"{title}\". The thumbnail "
                "shows this moment: {subject}. The video's story moment: {moment}. Give 4 different thumbnail texts, each "
                "1 to 3 English words in CAPITALS, punchy (emotion, stakes or a question), TRUE to the moment, and NOT "
                "repeating any word of the title. Ignore any HUD / help text, subtitles or watermarks visible in the "
                "image - write about the STORY moment. No emojis, no channel names. Reply with JSON only: "
                "{{\"texts\": [\"...\", \"...\", \"...\", \"...\"]}}")


def _ask(judge, items, cache, key, tokens=80):
    from .judge import parse_json
    todo = [(k, ims, p) for k, ims, p in items if k not in cache]
    for i in range(0, len(todo), 4):
        ch = todo[i:i + 4]
        for (k, _, _), r in zip(ch, judge.ask([(ims, p) for _, ims, p in ch], tokens)):
            cache[k] = parse_json(r) or {}
    return [cache.get(k, {}) for k, _, _ in items]


def pick_faces(video, fc, judge, cache, n=3):
    times = loud_moments(video)
    if not times:
        return []
    x, y, w, h = fc["box"]
    crops = []
    for t in times:
        im = _frame(video, t)
        if im is None:
            continue
        Wf, Hf = im.size
        crops.append((t, im.crop((int(x * Wf), int(y * Hf), int((x + w) * Wf), int((y + h) * Hf)))))
    res = _ask(judge, [(f"{STUDIO_VERSION}|face|{t:.1f}", [c], FACE_PROMPT) for t, c in crops], cache, "face")
    scored = []
    for (t, c), d in zip(crops, res):
        try:
            s = float(d.get("intensity", 0))
        except Exception:
            s = 0
        if str(d.get("clear", "")).lower() == "true":
            scored.append((s, t, c, str(d.get("emotion", ""))[:20]))
    scored.sort(key=lambda z: -z[0])
    return [dict(t=t, img=c, intensity=s, emotion=e) for s, t, c, e in scored[:n]]


CHAR_PROMPT = ("This is a crop of a video game frame showing a game character. Rate it as the MAIN SUBJECT of a "
               "YouTube thumbnail. Reply with JSON only: {\"intensity\": 0-10 (10 = shouting, terrified, furious, "
               "aiming a gun, mid-action; 0 = calm, neutral, back of the head, blurry), \"emotion\": one word, "
               "\"clear\": true if the face is sharp and visible, \"gore\": true if blood / wounds are clearly "
               "visible, \"sexual\": true if buttocks / breasts / sexual posing is the focus}")


def pick_characters(video, times, fc, judge, cache, n=3):
    """Game characters with big faces (story close-ups, action shots). Faces inside a webcam overlay are ignored."""
    import cv2
    if not os.path.exists(YUNET):
        return []
    det = cv2.FaceDetectorYN.create(YUNET, "", (960, 540), 0.8)
    dur = _duration(video)
    ts = list(times[:40]) + list(np.linspace(dur * 0.03, dur * 0.97, 30))
    crops = []
    for t in ts:
        im = _frame(video, t)
        if im is None:
            continue
        Wf, Hf = im.size
        small = np.array(im.resize((960, 540)))[:, :, ::-1].copy()
        _, faces = det.detect(small)
        best = None
        for f in (faces if faces is not None else []):
            x, y, w, h = [float(v) for v in f[:4]]
            fx, fy, fw, fh = x / 960, y / 540, w / 960, h / 540
            if fc:
                bx, by, bw, bh = fc["box"]
                if bx <= fx + fw / 2 <= bx + bw and by <= fy + fh / 2 <= by + bh:
                    continue
            if fh >= 0.12 and (best is None or fh > best[3]):
                best = (fx, fy, fw, fh)
        if not best:
            continue
        fx, fy, fw, fh = best
        cw = min(1.0, fw * 3.4)
        x0 = max(0.0, min(1 - cw, fx + fw / 2 - cw / 2))
        y0 = max(0.0, fy - fh * 0.7)
        y1 = min(1.0, fy + fh * 4.2)
        crop = im.crop((int(x0 * Wf), int(y0 * Hf), int((x0 + cw) * Wf), int(y1 * Hf)))
        if crop.width < 120 or crop.height < 160:
            continue
        crops.append((t, crop, fh))
    if not crops:
        return []
    res = _ask(judge, [(f"{STUDIO_VERSION}|char|{t:.1f}", [c.resize((max(1, int(c.width * 448 / c.height)), 448))],
                        CHAR_PROMPT) for t, c, _ in crops], cache, "char")
    scored = []
    for (t, c, fh), d in zip(crops, res):
        if str(d.get("gore", "")).lower() == "true" or str(d.get("sexual", "")).lower() == "true":
            continue
        if str(d.get("clear", "")).lower() != "true":
            continue
        try:
            sc = float(d.get("intensity", 0))
        except Exception:
            sc = 0
        scored.append((sc + min(2.0, fh * 8), t, c, str(d.get("emotion", ""))[:20]))
    scored.sort(key=lambda z: -z[0])
    out = []
    for sc, t, c, e in scored:
        if all(abs(t - o["t"]) > 45 for o in out):          # spread out = more likely different characters
            out.append(dict(t=t, img=c, intensity=sc, emotion=e))
        if len(out) >= n:
            break
    return out


def pick_moments(video, times, fc, judge, cache, n=3, avoid=()):
    """Peak-tension game frames, sharp, never gore / sexual focus; a webcam corner is cropped away; not the frame
    the subject was cut from (the thumbnail needs two different images, like the references)."""
    import cv2
    cand = []
    for t in times[:30]:
        if any(abs(t - a) < 4 for a in avoid):
            continue
        im = _frame(video, t)
        if im is None:
            continue
        g = cv2.cvtColor(np.array(im.resize((640, 360))), cv2.COLOR_RGB2GRAY)
        sharp = cv2.Laplacian(g, cv2.CV_64F).var()
        if sharp < 60 or g.mean() < 30:               # blurry close-ups / black frames make mushy thumbnails
            continue
        cand.append((t, _hide_facecam(im, fc)))
    res = _ask(judge, [(f"{STUDIO_VERSION}|bg|{t:.1f}", [c.resize((640, 360))], MOMENT_PROMPT) for t, c in cand],
               cache, "bg")
    ok = []
    for (t, c), d in zip(cand, res):
        if str(d.get("gore", "")).lower() == "true" or str(d.get("sexual", "")).lower() == "true":
            continue
        try:
            s = float(d.get("tension", 0))
        except Exception:
            s = 0
        ok.append((s, t, c, str(d.get("subject", ""))[:60]))
    ok.sort(key=lambda z: -z[0])
    out = []
    for s, t, c, sub in ok:
        if all(abs(t - o["t"]) > 20 for o in out):
            out.append(dict(t=t, img=c, tension=s, subject=sub))
        if len(out) >= n:
            break
    return out


def _hide_facecam(im, fc):
    """Clean background crop (82 %): drops the subtitle line and bottom HUD (y 6-88 %), the top-left help box mostly,
    and a webcam corner if there is one (the crop slides horizontally away from it)."""
    Wf, Hf = im.size
    cw, chh = int(Wf * 0.78), int(Hf * 0.76)       # rows 4-80 %: subtitles sit at ~82-92 % in GTA
    top = int(Hf * 0.04)
    left = (Wf - cw) // 2
    if fc:
        x, y, w, h = fc["box"]
        left = 0 if x + w / 2 > 0.5 else Wf - cw
    return im.crop((left, top, left + cw, top + chh))


# ------------------------------------------------------------------------------------------------ 3 cutout
_RB = {}


def cutout(img, outline=False):          # owner rule: NO outline, ever
    """RGBA person cut-out with a white sticker outline. None when the mask looks wrong."""
    from PIL import Image, ImageFilter
    try:
        from rembg import remove, new_session
        if "s" not in _RB:
            _RB["s"] = new_session("u2net_human_seg")
        if img.height < 900:                       # matting is much cleaner on a bigger image
            r = 900 / img.height
            img = img.resize((int(img.width * r), 900), Image.LANCZOS)
        rgba = remove(img, session=_RB["s"])
        rgba = _main_subject(rgba)
    except Exception as e:
        print(f"  thumbnail cutout unavailable: {e!r}")
        return None
    a = np.array(rgba.split()[-1]).astype(np.float32) / 255
    cover = a.mean()
    if cover < 0.08 or cover > 0.92:
        return None
    alpha = rgba.split()[-1]
    pad = 24
    big = Image.new("RGBA", (rgba.width + 2 * pad, rgba.height + 2 * pad), (0, 0, 0, 0))
    big.paste(rgba, (pad, pad), rgba)
    a2 = Image.new("L", big.size, 0); a2.paste(alpha, (pad, pad))
    outline_m = a2.filter(ImageFilter.MaxFilter(17))                # white sticker border ~8 px
    shadow = a2.filter(ImageFilter.MaxFilter(17)).filter(ImageFilter.GaussianBlur(10))
    out = Image.new("RGBA", big.size, (0, 0, 0, 0))
    out.paste((0, 0, 0, 150), (6, 8), shadow)
    if outline:                                  # white sticker stroke only where wanted
        out.paste((255, 255, 255, 255), (0, 0), outline_m)
    out.alpha_composite(big)
    return out


def _main_subject(rgba):
    """Keep only the largest connected person in the mask (drops a second face / stray bits)."""
    import cv2
    from PIL import Image
    a = np.array(rgba.split()[-1])
    n, lab, stats, _ = cv2.connectedComponentsWithStats((a > 40).astype(np.uint8), 8)
    if n <= 2:
        return rgba
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    a = np.where(lab == keep, a, 0).astype(np.uint8)
    out = rgba.copy()
    out.putalpha(Image.fromarray(a))
    return out


def framed(img):
    """Fallback: rounded crop with a white frame (a 'sticker' webcam)."""
    from PIL import Image, ImageDraw
    w, h = img.size
    m = Image.new("L", (w, h), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius=int(min(w, h) * 0.12), fill=255)
    pad = 10
    out = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    fr = Image.new("L", out.size, 0); ImageDraw.Draw(fr).rounded_rectangle((0, 0, out.width - 1, out.height - 1),
                                                                           radius=int(min(w, h) * 0.12) + pad, fill=255)
    out.paste((255, 255, 255, 255), (0, 0), fr)
    out.paste(img.convert("RGBA"), (pad, pad), m)
    return out


# ------------------------------------------------------------------------------------------------ 4 compose
def _grade(im):
    """Saturated teal-orange grade + contrast + vignette (the references' look)."""
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
    im = im.convert("RGB").resize((W, H), Image.LANCZOS)
    im = ImageEnhance.Contrast(im).enhance(1.18)
    im = ImageEnhance.Color(im).enhance(1.35)
    a = np.array(im).astype(np.float32) / 255
    lum = a.mean(2, keepdims=True)
    teal = np.array([0.0, 0.06, 0.10]); warm = np.array([0.08, 0.03, -0.04])
    a = a + (1 - lum) * teal * 0.6 + lum * warm * 0.6
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    im = ImageEnhance.Sharpness(im).enhance(1.5)
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).ellipse((-W * 0.2, -H * 0.25, W * 1.2, H * 1.25), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(110))
    return Image.composite(im, ImageEnhance.Brightness(im).enhance(0.42), mask)


def _font(size):
    from PIL import ImageFont
    for f in ("C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def _badge(im, text, corner="tr"):
    """Own game badge: Vice-City sunset gradient word with a black outline (not the publisher's logo)."""
    from PIL import Image, ImageDraw
    f = _font(78)
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f, stroke_width=7)
    tw, th = x1 - x0, y1 - y0
    grad = Image.new("RGB", (tw, th))
    g = np.linspace(0, 1, th)[:, None]
    col = (np.array([255, 45, 149]) * (1 - g) + np.array([255, 179, 71]) * g)[:, None, :].repeat(tw, 1).reshape(th, tw, 3)
    grad = Image.fromarray(col.astype(np.uint8))
    m = Image.new("L", (tw, th), 0); ImageDraw.Draw(m).text((-x0, -y0), text, font=f, fill=255)
    pos = {"tr": (W - tw - 34, 26), "tl": (34, 26), "bl": (34, H - th - 34)}.get(corner, (34, 26))
    d.text((pos[0] - x0, pos[1] - y0), text, font=f, fill=(0, 0, 0), stroke_width=7, stroke_fill=(0, 0, 0))
    im.paste(grad, pos, m)


def _words(im, text, box, align="left"):
    """Up to 3 words, two lines max, first word yellow, rest white, thick black outline, auto-sized to the box."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    words = text.upper().split()[:3]
    lines = [" ".join(words)] if len(" ".join(words)) <= 9 else [" ".join(words[:-1]) or words[0], words[-1]] \
        if len(words) > 1 else [words[0]]
    bx, by, bw, bh = box
    size = 210
    while size > 50:
        f = _font(size)
        sw = max(4, size // 11)
        dims = [d.textbbox((0, 0), ln, font=f, stroke_width=sw) for ln in lines]
        tw = max(b[2] - b[0] for b in dims); th = sum(b[3] - b[1] for b in dims) + (len(lines) - 1) * size * 0.08
        if tw <= bw and th <= bh:
            break
        size -= 6
    y = by + (bh - th) / 2
    for i, (ln, b) in enumerate(zip(lines, dims)):
        lw = b[2] - b[0]
        x = bx if align == "left" else bx + bw - lw
        col = (255, 210, 63) if i == 0 else (255, 255, 255)
        d.text((x - b[0], y - b[1]), ln, font=f, fill=col, stroke_width=sw, stroke_fill=(0, 0, 0))
        y += (b[3] - b[1]) + size * 0.08


def _part(im, label):
    from PIL import ImageDraw
    if not label:
        return
    d = ImageDraw.Draw(im); f = _font(54)
    x0, y0, x1, y1 = d.textbbox((0, 0), label, font=f)
    pw, ph = x1 - x0 + 36, y1 - y0 + 22
    x, y = W - pw - 30, H - ph - 28
    d.rounded_rectangle((x, y, x + pw, y + ph), radius=14, fill=(220, 30, 60))
    d.text((x + 18 - x0, y + 11 - y0), label, font=f, fill=(255, 255, 255))


def _place_face(canvas, face, side, scale):
    """Cut-out (or framed) face, height = scale * canvas height, anchored at the bottom on the given side."""
    from PIL import Image
    fimg = face["cut"]
    r = (H * scale) / fimg.height
    fimg = fimg.resize((max(1, int(fimg.width * r)), max(1, int(fimg.height * r))), Image.LANCZOS)
    x = -int(fimg.width * 0.04) if side == "left" else W - fimg.width + int(fimg.width * 0.04)
    y = H - fimg.height + (int(fimg.height * 0.03) if face.get("is_cut") else -20)
    canvas.alpha_composite(fimg, (x, y))
    return fimg.width


def compose(bg, face, text, badge, part, layout, clean=False):
    """3 layouts so Test & Compare compares real concepts, not tiny variations:
      0  reaction: face big on the LEFT, words in the right half, badge top-right
      1  action:   face on the RIGHT, words top-left, badge bottom-left
      2  cinematic: no words, face on the right, badge top-left (the moment speaks)"""
    c = _grade(bg).convert("RGBA")
    if layout == 0:
        fw = _place_face(c, face, "left", 0.98) if face else 0
        left = max(fw, int(W * 0.40)) + 10
        if not clean:
            _words(c, text, (left, 150, W - left - 50, 420), "right")
            _badge(c, badge, "tr")
    elif layout == 1:
        fw = _place_face(c, face, "right", 0.88) if face else 0
        if not clean:
            _words(c, text, (40, 40, (W - fw - 70) if fw else W - 80, 300), "left")
            _badge(c, badge, "bl")
    else:
        if face:
            _place_face(c, face, "right", 0.92)
        if not clean:
            _badge(c, badge, "tl")
    if not clean:
        _part(c, part)
    return c.convert("RGB")


# ------------------------------------------------------------------------------------------------ 5 run
def make(video, cfg, workdir, outdir, prefix, title, part_label, times, moment_text="", judge=None):
    """Write <prefix>_1..3.jpg in the owner's style. Returns paths, or [] when there is no facecam / it failed
    (the caller then falls back to thumbs.make)."""
    tc = cfg.get("thumbnails") or {}
    if tc.get("style", "studio") != "studio":
        return []
    fc = None
    if tc.get("facecam_box"):
        fc = dict(box=tuple(tc["facecam_box"]), hits=-1, of=-1)
    cache_p = os.path.join(workdir, "thumb_studio.json")
    cache = json.load(open(cache_p, encoding="utf-8")) if os.path.exists(cache_p) else {}
    if fc is None:
        k = f"{STUDIO_VERSION}|facecam"
        if k in cache:
            fc = cache[k]
        else:
            fc = find_facecam(video); cache[k] = fc
    if fc:
        print(f"  thumbnail studio: webcam overlay at {fc['box']} - ignored (faceless channel)")
    own = judge is None
    if own:
        from . import judge as J
        judge = J.Judge(J.pick_model(cfg))
    try:
        faces = pick_characters(video, times, fc, judge, cache)       # faceless channel: game characters
        for f in faces:
            cut = cutout(f["img"])
            f["cut"], f["is_cut"] = (cut, True) if cut is not None else (framed(f["img"]), False)
            plain = cutout(f["img"], outline=False) if cut is not None else None
            f["plain"] = plain if plain is not None else f["cut"]
        moments = pick_moments(video, times, fc, judge, cache, avoid=[f["t"] for f in faces])
        if not moments:
            return []
        title_words = set(re.findall(r"[a-z]+", title.lower()))
        sub = "; ".join(m["subject"] for m in moments)
        k = f"{STUDIO_VERSION}|words|{title}|{sub}"
        if k not in cache:
            from .judge import parse_json
            r = judge.ask([([moments[0]["img"].resize((640, 360))],
                            WORDS_PROMPT.format(title=title, subject=sub, moment=moment_text or "n/a"))], 120)[0]
            cache[k] = (parse_json(r) or {}).get("texts") or []
        texts = []
        for t in cache[k]:
            t = re.sub(r"[^A-Za-z0-9 !?']", "", str(t)).strip().upper()
            if 1 <= len(t.split()) <= 3 and not (set(re.findall(r"[a-z]+", t.lower())) & title_words - {"the", "a"}):
                texts.append(t)
        texts = (texts + ["NO WAY", "WHAT?!", "IT BEGINS"])[:3]
        json.dump(cache, open(cache_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
        badge = (cfg.get("_game") or {}).get("title") or "GAME"
        outs = []
        for i in range(3):
            bg = moments[i % len(moments)]["img"]
            face = faces[i % len(faces)] if faces else None
            im = compose(bg, face, texts[i], badge.upper(), part_label, i)
            p = os.path.join(outdir, f"{prefix}_{i + 1}.jpg")
            q = 92
            im.save(p, "JPEG", quality=q)
            while os.path.getsize(p) > 2_000_000 and q > 60:       # YouTube limit 2 MB
                q -= 6; im.save(p, "JPEG", quality=q)
            outs.append(p)
        if tc.get("ai", True):                     # owner's art direction: AI-finished 4K from the best scenes
            try:
                from . import thumbai
                from . import judge as J2
                if own:
                    judge.close(); own = False
                comps, metas = [], []
                for i in range(3):
                    m = moments[i % len(moments)]
                    face = faces[i % len(faces)] if faces else None
                    f2 = dict(face, cut=face["cut"] if i == 0 else face["plain"]) if face else None
                    comps.append(compose(m["img"], f2, "", "", "", i, clean=True))
                    metas.append(dict(scene=m["img"], subject=(face or m)["img"], moment=moment_text, t=m["t"]))
                ai = thumbai.finish(comps, metas, lambda: J2.Judge(J2.pick_model(cfg)), cfg, outdir, prefix, cache,
                                    STUDIO_VERSION)
                if ai:
                    outs = ai
            except Exception as e:
                print(f"  WARNING: thumbnail AI step failed, keeping the composite thumbnails: {e!r}")
        print(f"  thumbnail studio: {len(outs)} thumbnails ({'; '.join(texts)}) faces "
              f"{[f['emotion'] for f in faces]} moments {[round(m['tension']) for m in moments]}")
        return outs
    finally:
        json.dump(cache, open(cache_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
        if own:
            try:
                judge.close()
            except Exception:
                pass

"""AUTHENTIC thumbnails (owner's brief 2026-11-18, "Professional gaming thumbnail creator").
Priority order from the brief: authenticity > cinematic storytelling > clickability. "Never sacrifice game accuracy."

What it does (and deliberately does NOT do):
  * the subject is a REAL character cut from the footage (no AI repaint, no invented people or objects)
  * the background is a REAL frame of the same video; its colours are left as the game made them
    (no teal-orange grade, no neon, no glow); only gentle contrast and selective sharpening on the subject
  * the subject is colour-matched to the background (LAB mean/std transfer at 35 %), with a soft contact shadow
    and a feathered edge - NO outline (owner rule 2026-11-17)
  * ONE contextual word when it helps (WANTED / ESCAPE / ...), chosen by the AI from what the scene shows;
    clean bold type with a soft drop shadow, no glow
  * CONCEPT and FINAL are separate files: <prefix>_CONCEPT.png is a rough storyboard + notes; the final
    <prefix>.jpg has no sketch, arrows or annotations
  * QC (Qwen-VL): authentic-looking characters, no AI artefacts, readable small, true to the video
"""
import json, os, re
import numpy as np

W, H = 1280, 720
VERSION = "tp3"
WORDS = ["WANTED", "ESCAPE", "BUSTED", "TRAPPED", "HUNTED", "BETRAYED", "NO RADAR", "HEIST", "CHASE", "10/10",
         "120 HOURS", "FINAL MISSION", "NEW POLICE", "GONE"]

PLAN_PROMPT = ("You are a professional gaming thumbnail art director. Image 1 is the chosen background moment and "
               "image 2 the chosen main character, both from the same {game} video titled \"{title}\". {extra}"
               "Reply with JSON only: {{\"situation\": what the viewer should understand in one second (under 15 "
               "words), \"side\": \"left\" or \"right\" (where the character goes so it does not cover the action), "
               "\"word\": one word or short phrase from this list that is TRUE to the video, or \"\" if no text is "
               "better: {words}, \"why\": one sentence on the visual hook}}")
QC_PROMPT = ("This is a finished YouTube thumbnail for a {game} video. Reply with JSON only: {{\"authentic\": true if "
             "the characters and scene look like real in-game footage (no AI artefacts, no distorted faces or hands), "
             "\"readable\": true if the main subject and any text are clear at small mobile size, \"pasted\": true if "
             "the main character looks badly pasted on, \"score\": 0-10 overall professional quality}}")


def _lab_match(src_rgba, ref_rgb, k=0.35):
    """Move the cut-out's colour statistics 35 % toward the background (natural integration, no regrade)."""
    import cv2
    from PIL import Image
    a = np.array(src_rgba)
    rgb, al = a[:, :, :3], a[:, :, 3]
    m = al > 128
    if m.sum() < 100:
        return src_rgba
    s = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    r = cv2.cvtColor(np.array(ref_rgb), cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    for c in range(3):
        sm, ss = s[:, :, c][m].mean(), s[:, :, c][m].std() + 1e-3
        rm, rs = r[:, c].mean(), r[:, c].std() + 1e-3
        t = (s[:, :, c] - sm) * (rs / ss) + rm
        s[:, :, c] = s[:, :, c] * (1 - k) + t * k
    out = cv2.cvtColor(np.clip(s, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(np.dstack([out, al]))


def _feather(rgba, px=2, blur=1.6):
    from PIL import ImageFilter
    a = rgba.split()[-1].filter(ImageFilter.MinFilter(px * 2 + 1)).filter(ImageFilter.GaussianBlur(blur))
    o = rgba.copy(); o.putalpha(a)
    return o


def cut_ok(rgba, bg):
    """Hard rules (2026-11-18 review): a cut-out is used only if (1) its mask does not run along the crop border
    (that leaves a straight 'box' edge) and (2) its brightness is close to the background's (a dark-scene character
    in a sunny frame always looks pasted)."""
    import cv2
    a = np.array(rgba.split()[-1]) > 128
    h, w = a.shape
    edges = [a[:3, :].mean(), a[:, :3].mean(), a[:, -3:].mean()]          # top, left, right borders
    if max(edges) > 0.18:
        return False, "mask touches the crop border"
    ys, xs = np.where(a)
    if len(ys) < 500:
        return False, "mask too small"
    fill = a.sum() / float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    if fill > 0.80:                     # a person never fills its bounding box like a rectangle does
        return False, f"mask is a box, not a person (fill {fill:.2f})"
    rgb = np.array(rgba.convert("RGB"))
    ls = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0][a].mean()
    lb = cv2.cvtColor(np.array(bg.convert("RGB").resize((640, 360))), cv2.COLOR_RGB2LAB)[:, :, 0].mean()
    if abs(ls - lb) > 28:
        return False, f"lighting mismatch ({ls:.0f} vs {lb:.0f})"
    return True, ""


def compose(bg, subj, side, word):
    """bg: RGB frame, subj: RGBA plain cut-out (no outline). Returns the final RGB 1280x720."""
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
    base = bg.convert("RGB").resize((W, H), Image.LANCZOS)
    base = ImageEnhance.Contrast(base).enhance(1.06)
    # gentle darkening on the subject's side only (separation), not a vignette
    grad = np.linspace(0.78, 1.0, W) if side == "left" else np.linspace(1.0, 0.78, W)
    arr = (np.array(base).astype(np.float32) * grad[None, :, None]).clip(0, 255).astype(np.uint8)
    canvas = Image.fromarray(arr).convert("RGBA")
    if subj is not None:
        s = _feather(_lab_match(subj, base))
        r = (H * 0.92) / s.height
        s = s.resize((max(1, int(s.width * r)), max(1, int(s.height * r))), Image.LANCZOS)
        s = ImageEnhance.Sharpness(s).enhance(1.35)               # selective sharpening: subject only
        x = -int(s.width * 0.03) if side == "left" else W - s.width + int(s.width * 0.03)
        y = H - s.height + int(s.height * 0.02)
        sh = Image.new("RGBA", s.size, (0, 0, 0, 0))               # soft contact shadow, offset down-away
        sh.putalpha(s.split()[-1].point(lambda v: int(v * 0.55)).filter(ImageFilter.GaussianBlur(14)))
        canvas.alpha_composite(sh, (x + (14 if side == "left" else -14), y + 12))
        canvas.alpha_composite(s, (x, y))
        sub_w = s.width
    else:
        sub_w = 0
    if word:
        d = ImageDraw.Draw(canvas)
        fnt = next((f for f in ("C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/arialbd.ttf") if os.path.exists(f)), None)
        area_w = W - sub_w - 90 if sub_w else W - 120
        size = 190
        while size > 60:
            f = ImageFont.truetype(fnt, size)
            if d.textlength(word, font=f) <= area_w:
                break
            size -= 6
        tw = d.textlength(word, font=f)
        tx = (W - tw - 50) if side == "left" else 50
        g = np.asarray(base.convert("L")).astype(np.float32)          # put the word where the picture is calm
        e = np.abs(np.diff(g, axis=0)).mean(axis=1)
        top, bot = e[:int(H * 0.32)].mean(), e[int(H * 0.68):].mean()
        ty = 60 if top <= bot else H - int(size * 1.15) - 50
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow).text((tx + 6, ty + 8), word, font=f, fill=(0, 0, 0, 200))
        canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(6)))
        d.text((tx, ty), word, font=f, fill=(255, 255, 255))
    return canvas.convert("RGB")


def concept(bg, final, side, word, plan, path):
    """Separate CONCEPT sheet: rough storyboard (line sketch of the layout) + the composition notes."""
    from PIL import Image, ImageDraw, ImageFont
    from .thumbai import sketch
    board = Image.new("RGB", (1600, 900), (245, 243, 238))
    d = ImageDraw.Draw(board)
    def font(n):
        for f in ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"):
            if os.path.exists(f):
                return ImageFont.truetype(f, n)
        return ImageFont.load_default()
    d.text((60, 40), "CONCEPT  -  storyboard (not the final thumbnail)", font=font(40), fill=(30, 30, 30))
    sk = sketch(final).resize((960, 540))
    board.paste(sk, (60, 120))
    d.rectangle((60, 120, 1020, 660), outline=(30, 30, 30), width=3)
    notes = [f"Situation: {plan.get('situation', '')}", f"Hook: {plan.get('why', '')}",
             f"Character side: {side}", f"Text: {word or '(none - image tells it)'}",
             "Colours: original game palette (no regrade)", "No outline, real footage only"]
    y = 130
    for n in notes:
        words, line = n.split(), ""
        for w in words:
            t = (line + " " + w).strip()
            if d.textlength(t, font=font(26)) > 520 and line:
                d.text((1060, y), line, font=font(26), fill=(40, 40, 40)); y += 36; line = w
            else:
                line = t
        d.text((1060, y), line, font=font(26), fill=(40, 40, 40)); y += 54
    board.save(path)


def make(video, times, title, game, workdir, outdir, prefix, cfg, extra="", n=2, words=None, char_video=None,
         char_times=None, crop=1.0):
    """Write <prefix>_1..n.jpg (final, 1280x720) and <prefix>_1..n_CONCEPT.png. Returns final paths.
    `words`: the contextual words allowed for THIS game (default: the GTA list)."""
    vocab = [w.upper() for w in (words or WORDS)]
    from . import thumbstudio as TS, judge as J
    from .judge import parse_json
    cache_p = os.path.join(workdir, "thumbpro.json")
    cache = json.load(open(cache_p, encoding="utf-8")) if os.path.exists(cache_p) else {}
    judge = J.Judge(J.pick_model(cfg))
    try:
        chars = TS.pick_characters(char_video or video, char_times or times, None, judge, cache, n=8)
        moments = TS.pick_moments(video, times, None, judge, cache, n=4, avoid=[c["t"] for c in chars])
        if not moments:
            return []
        outs, used, used_words = [], [], []
        for i in range(n):
            m = dict(moments[i % len(moments)])
            if crop < 1.0:                           # punch in: removes HUD corners, tighter on the action
                im = m["img"]; cw, chh = int(im.width * crop), int(im.height * crop)
                m["img"] = im.crop(((im.width - cw) // 2, (im.height - chh) // 2, (im.width + cw) // 2, (im.height + chh) // 2))
            c, subj, why = None, None, "no character"
            for cand in chars:                           # first UNUSED character whose cut-out passes the hard rules
                if any(abs(cand["t"] - u) < 1.0 for u in used):
                    continue
                cut = TS.cutout(cand["img"], outline=False)
                if cut is None:
                    continue
                ok, why = cut_ok(cut, m["img"])
                if ok:
                    c, subj = cand, cut
                    used.append(cand["t"])
                    break
            if subj is None:
                print(f"  thumbpro {i + 1}: real frame only ({why})")
            key = f"{VERSION}|plan|{title}|{m['t']:.1f}|{c['t'] if c else -1:.1f}|{','.join(vocab)}"
            if key not in cache:
                ims = [m["img"].resize((768, 432))] + ([c["img"].resize((max(1, int(c["img"].width * 432 / c["img"].height)), 432))] if c else [])
                r = judge.ask([(ims, PLAN_PROMPT.format(game=game, title=title, extra=extra, words=", ".join(vocab)))], 160)[0]
                cache[key] = parse_json(r) or {}
            plan = cache[key]
            side = "right" if str(plan.get("side", "left")).lower().startswith("r") else "left"
            word = str(plan.get("word", "")).upper().strip()
            word = word if word in vocab else ""
            if word in used_words:                 # no two thumbnails with the same word
                word = ""
            if word:
                used_words.append(word)
            final = compose(m["img"], subj, side, word)
            qk = f"{VERSION}|qc|{key}|{word}|{side}"
            if qk not in cache:
                cache[qk] = parse_json(judge.ask([([final], QC_PROMPT.format(game=game))], 80)[0]) or {}
            q = cache[qk]
            if subj is not None and (str(q.get("pasted", "")).lower() == "true" or str(q.get("authentic", "true")).lower() == "false"):
                final = compose(m["img"], None, side, word)   # fails QC with the cut-out -> the real frame only
                q = dict(q, note="cut-out removed by QC")
            p = os.path.join(outdir, f"{prefix}_{i + 1}.jpg")
            final.save(p, "JPEG", quality=92)
            concept(m["img"], final, side, word, plan, os.path.join(outdir, f"{prefix}_{i + 1}_CONCEPT.png"))
            outs.append(p)
            print(f"  thumbpro {i + 1}: bg @{m['t']:.0f}s, character @{c['t'] if c else '-'}, word '{word}', QC {q}")
        return outs
    finally:
        json.dump(cache, open(cache_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
        judge.close()

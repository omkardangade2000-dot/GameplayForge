"""AI ART DIRECTION for thumbnails (owner 2026-11-17): turn the composite made from the BEST SCENE of the video into a
cinematic, professionally blended 3840x2160 thumbnail, following the owner's art-direction prompt
(templates/thumbnail_art_direction.txt). The prompt is a GUIDE: the scene, characters and situation always come
from the video itself, so the thumbnail is true to the video (YouTube: thumbnails must not mislead) and the
characters are the real in-game ones (the guide's #1 rule: character accuracy, no generic AI people).

Steps (all local on the RTX 5070 Ti, nothing uploaded):
  1. scenario: Qwen-VL looks at the chosen scene + character and writes the "THUMBNAIL SCENARIO" (like the owner's
     example: who, what happens, camera, expression, focus) + a short render prompt + labels for the sketch
  2. render:   RealVisXL V5 (SDXL, openrail++) img2img on the CLEAN composite (no text / logo - the guide says
               avoid unnecessary text, watermarks and logos) at moderate strength: relights and blends the cut-out
               into the scene, keeps faces, poses and layout from the real frames
  3. upscale:  Real-ESRGAN x4plus (BSD-3) -> 3840x2160 master PNG + 1280x720 upload JPG (< 2 MB)
  4. QC:       Qwen-VL checks faces / hands / 'pasted look'; a distorted result is rejected (lower strength, then the
               un-AI'd composite) - never ship a broken face
  5. process image: our OWN layout (dark board: scene from the video -> sketch -> final), deliberately different
               from the reference designer's white sketch + photo + red-arrow layout
"""
import json, os, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "thumbnail_art_direction.txt")
ESRGAN = os.path.join(ROOT, "models", "RealESRGAN_x4plus.pth")
SDXL = "SG161222/RealVisXL_V5.0"
GEN_W, GEN_H = 1360, 768                       # SDXL-friendly 16:9 (multiples of 8); x4 -> 5440x3072 -> 3840x2160

SCENARIO_PROMPT = (
    "You are an art director for {game} YouTube thumbnails. Image 1 is the chosen scene from the video, image 2 is the "
    "character who will be the main subject. Story moment (may be empty): \"{moment}\".\n"
    "Write the thumbnail plan for THIS situation only (do not invent events that are not in the images). Reply with "
    "JSON only: {{\"scenario\": 3-5 short sentences like: '<who> is <doing what> <where>. <supporting characters / "
    "objects>. Camera: <angle / shot>. Expression: <emotion>. Make <the key thing> the immediate visual focus.', "
    "\"render\": a comma-separated image prompt under 50 words (subject, action, setting, camera, lighting, mood), "
    "\"labels\": up to 5 short labels of the people / objects that matter (names if you know them, else roles)}}"
)
QC_PROMPT = ("This is an AI-enhanced video game thumbnail. Reply with JSON only: {\"faces_ok\": true if every visible "
             "face looks natural and undistorted, \"hands_ok\": true if hands look natural or are not visible, "
             "\"pasted\": true if the main character looks badly pasted on, \"quality\": 0-10 overall}")
NEGATIVE = ("text, watermark, logo, signature, distorted face, deformed hands, extra fingers, extra limbs, bad anatomy, "
            "blurry, lowres, flat lighting, excessive glow, cartoon, painting, generic poster, duplicate person")


def full_prompt(scenario):
    t = open(TEMPLATE, encoding="utf-8").read()
    return t.replace("{scenario}", scenario.strip())


def scenarios(judge, items, game, cache, version):
    """items: [(key, scene_img, subject_img, moment)] -> [dict(scenario, render, labels)]"""
    from .judge import parse_json
    out = []
    for k, scene, subj, moment in items:
        ck = f"{version}|scen|{k}"
        if ck not in cache:
            r = judge.ask([([scene.resize((768, 432)), subj.resize((max(1, int(subj.width * 432 / subj.height)), 432))],
                            SCENARIO_PROMPT.format(game=game, moment=moment or ""))], 260)[0]
            d = parse_json(r) or {}
            cache[ck] = dict(scenario=str(d.get("scenario", ""))[:900], render=str(d.get("render", ""))[:400],
                             labels=[str(x)[:24] for x in (d.get("labels") or [])][:5])
        out.append(cache[ck])
    return out


# ------------------------------------------------------------------------------------------------ render
def _sdxl():
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline
    p = StableDiffusionXLImg2ImgPipeline.from_pretrained(SDXL, torch_dtype=torch.float16, variant="fp16",
                                                         use_safetensors=True)
    p.to("cuda")
    p.set_progress_bar_config(disable=True)
    return p


def render(pipe, img, prompt, strength, seed=7):
    import torch
    from PIL import Image
    g = torch.Generator("cuda").manual_seed(seed)
    base = ("ultra-realistic cinematic video game thumbnail, GTA 6 style, Vice City, " + prompt +
            ", sharp detailed faces, vibrant tropical colours, cinematic contrast, warm sunlight and cool shadows, "
            "professional photoshop compositing, high detail")
    out = pipe(prompt=base, negative_prompt=NEGATIVE, image=img.convert("RGB").resize((GEN_W, GEN_H), Image.LANCZOS),
               strength=strength, guidance_scale=5.0, num_inference_steps=36, generator=g).images[0]
    return out


def _esrgan():
    import torch
    from spandrel import ModelLoader
    m = ModelLoader().load_from_file(ESRGAN).model.eval().cuda().half()
    return m


def upscale(model, img, tile=384, pad=16):
    """x4 Real-ESRGAN with tiles (fits in VRAM at any size), then exact 3840x2160."""
    import torch
    from PIL import Image
    a = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda().half()
    _, _, h, w = a.shape
    out = torch.zeros((1, 3, h * 4, w * 4), dtype=torch.float16, device="cuda")
    with torch.no_grad():
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y0, x0 = max(0, y - pad), max(0, x - pad)
                y1, x1 = min(h, y + tile + pad), min(w, x + tile + pad)
                r = model(a[:, :, y0:y1, x0:x1])
                ty0, tx0 = (y - y0) * 4, (x - x0) * 4
                th, tw = min(tile, h - y) * 4, min(tile, w - x) * 4
                out[:, :, y * 4:y * 4 + th, x * 4:x * 4 + tw] = r[:, :, ty0:ty0 + th, tx0:tx0 + tw]
    o = (out.clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
    big = Image.fromarray(o)
    # 5440x3072 (1.771) -> 3840x2160 (1.778): scale to 2160 high, centre-crop the few extra pixels
    r = 2160 / big.height
    big = big.resize((int(round(big.width * r)), 2160), Image.LANCZOS)
    x = max(0, (big.width - 3840) // 2)
    return big.crop((x, 0, x + 3840, 2160)) if big.width >= 3840 else big.resize((3840, 2160), Image.LANCZOS)


def qc(judge, img, cache, key):
    from .judge import parse_json
    if key not in cache:
        r = judge.ask([([img.resize((1024, 576))], QC_PROMPT)], 80)[0]
        cache[key] = parse_json(r) or {}
    d = cache[key]
    ok = str(d.get("faces_ok", "")).lower() == "true" and str(d.get("hands_ok", "true")).lower() != "false"
    try:
        q = float(d.get("quality", 0))
    except Exception:
        q = 0
    return ok and str(d.get("pasted", "")).lower() != "true", q


# ------------------------------------------------------------------------------------------------ process image
def sketch(img):
    """Pencil line drawing (XDoG) of the composite - the 'idea' stage of the process image."""
    import cv2
    from PIL import Image
    g = cv2.cvtColor(np.array(img.convert("RGB").resize((960, 540))), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    g1, g2 = cv2.GaussianBlur(g, (0, 0), 1.0), cv2.GaussianBlur(g, (0, 0), 1.6)
    d = g1 - 0.985 * g2
    e = np.where(d >= 0.012, 1.0, 1 + np.tanh(160 * (d - 0.012)))
    e = np.clip(e, 0, 1)
    return Image.fromarray((e * 255).astype(np.uint8)).convert("RGB")


def process_image(scene, comp, final, labels, ts_label, path):
    """Our own process board (dark, left -> right): 1 scene from the video, 2 sketch + labels, 3 final."""
    from PIL import Image, ImageDraw, ImageFont
    Wb, Hb = 1920, 1080
    board = Image.new("RGB", (Wb, Hb), (14, 20, 30))
    d = ImageDraw.Draw(board)

    def font(n):
        for f in ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"):
            if os.path.exists(f):
                return ImageFont.truetype(f, n)
        return ImageFont.load_default()
    pw, ph, gap, top = 580, 326, 50, 250
    xs = [55, 55 + pw + gap, 55 + 2 * (pw + gap)]
    heads = [("1", "SCENE FROM THE VIDEO", ts_label), ("2", "SKETCH", "composition + labels"), ("3", "FINAL", "3840x2160")]
    panels = [scene, sketch(comp), final]
    for x, (n, h, sub), im in zip(xs, heads, panels):
        d.ellipse((x, top - 120, x + 56, top - 64), fill=(255, 196, 0))
        d.text((x + 28, top - 92), n, font=font(34), fill=(14, 20, 30), anchor="mm")
        d.text((x + 72, top - 112), h, font=font(38), fill=(255, 255, 255))
        d.text((x + 72, top - 64), sub, font=font(24), fill=(150, 165, 185))
        board.paste(im.convert("RGB").resize((pw, ph)), (x, top))
        d.rounded_rectangle((x - 3, top - 3, x + pw + 3, top + ph + 3), radius=8, outline=(255, 196, 0), width=3)
    for i in range(2):                                         # chevrons between panels
        cx = xs[i] + pw + gap // 2; cy = top + ph // 2
        d.polygon([(cx - 12, cy - 22), (cx + 14, cy), (cx - 12, cy + 22)], fill=(255, 196, 0))
    ly = top + ph + 40
    lx = xs[1]
    for lab in labels[:5]:                                     # label pills under the sketch
        t = lab.upper()
        tw = d.textlength(t, font=font(24))
        d.rounded_rectangle((lx, ly, lx + tw + 30, ly + 44), radius=22, fill=(30, 42, 60), outline=(255, 196, 0), width=2)
        d.text((lx + 15, ly + 8), t, font=font(24), fill=(255, 255, 255))
        lx += tw + 46
        if lx > xs[1] + pw:
            lx, ly = xs[1], ly + 56
    board.save(path)


# ------------------------------------------------------------------------------------------------ run
def finish(composites, metas, judge_factory, cfg, outdir, prefix, cache, version):
    """composites: clean 1280x720 RGB images; metas: per variant dict(scene, subject, moment, t).
    Writes <prefix>_N.jpg (1280x720 upload), <prefix>_N_4K.png, <prefix>_N_PROCESS.png, <prefix>_N_PROMPT.txt.
    Returns upload paths (or [] if the AI step is unavailable -> caller keeps its non-AI thumbnails)."""
    import torch, gc
    from .util import fmt
    tc = cfg.get("thumbnails") or {}
    if not tc.get("ai", True) or not os.path.exists(ESRGAN):
        return []
    game = (cfg.get("_game") or {}).get("title") or "GTA 6"
    judge = judge_factory()
    try:
        plans = scenarios(judge, [(f"{m['t']:.1f}|{i}", m["scene"], m["subject"], m["moment"])
                                  for i, m in enumerate(metas)], game, cache, version)
    finally:
        judge.close()
    del judge; gc.collect(); torch.cuda.empty_cache()
    strengths = tc.get("ai_strengths") or [0.38, 0.28]
    pipe = _sdxl()
    renders = []
    for i, (comp, pl) in enumerate(zip(composites, plans)):
        renders.append([(s, render(pipe, comp, pl["render"] or pl["scenario"], s, seed=11 + i)) for s in strengths])
    del pipe; gc.collect(); torch.cuda.empty_cache()
    judge = judge_factory()
    try:
        chosen = []
        for i, (comp, rs) in enumerate(zip(composites, renders)):
            best = None
            for s, im in rs:
                ok, q = qc(judge, im, cache, f"{version}|qc|{metas[i]['t']:.1f}|{i}|{s}")
                if ok and (best is None or q > best[1]):
                    best = (im, q, s)
            chosen.append(best)
    finally:
        judge.close()
    del judge; gc.collect(); torch.cuda.empty_cache()
    up = _esrgan()
    outs = []
    for i, (comp, best, pl, m) in enumerate(zip(composites, chosen, plans, metas)):
        src = best[0] if best else comp.resize((GEN_W, GEN_H))       # QC failed twice -> honest composite, no AI
        big = upscale(up, src)
        n = i + 1
        big.save(os.path.join(outdir, f"{prefix}_{n}_4K.png"))
        small = big.resize((1280, 720))
        p = os.path.join(outdir, f"{prefix}_{n}.jpg")
        q = 92
        small.save(p, "JPEG", quality=q)
        while os.path.getsize(p) > 2_000_000 and q > 60:
            q -= 6; small.save(p, "JPEG", quality=q)
        process_image(m["scene"], comp, small, pl["labels"], f"from the video at {fmt(m['t'])}",
                      os.path.join(outdir, f"{prefix}_{n}_PROCESS.png"))
        with open(os.path.join(outdir, f"{prefix}_{n}_PROMPT.txt"), "w", encoding="utf-8") as f:
            f.write(f"# Scene from the video at {fmt(m['t'])}. "
                    f"{'AI strength ' + str(best[2]) + ', QC ' + str(best[1]) if best else 'AI rejected by QC - composite only'}\n"
                    "# Full art-direction prompt (also usable in any image generator):\n\n" + full_prompt(pl["scenario"]))
        outs.append(p)
    del up; gc.collect(); torch.cuda.empty_cache()
    print(f"  thumbnail AI: {len(outs)} x 3840x2160 ({sum(1 for c in chosen if c)} AI-finished, "
          f"{sum(1 for c in chosen if not c)} composite-only after QC)")
    return outs

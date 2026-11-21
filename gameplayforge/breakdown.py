"""BREAKDOWN VIDEOS (GTA 6 launch, owner 2026-11-17): an original commentary / analysis video about a trailer or
showcase, where OUR narration is the video and the source footage only appears as short proof shots.

Why this format (policy, checked 2026-11-17):
  * YouTube allows "using clips for a critical review" and edited footage "where you add a storyline and commentary";
    it does not allow clips "edited together with little or no narrative" - credit is NOT permission.
  * Rockstar's policy covers fans' own non-commercial use; trailers are their marketing -> keep clips SHORT, never
    play in-game TV on its own, never show story reveals; credit + link the source.
So every point is: number card -> <= clip_max s of the moment (proof) -> freeze-frame with our callout while the
narration explains WHY it matters. Moving source footage stays a minority of the runtime (reported in the log).

Steps:  script.md (points + [SHOW: ...] notes)  ->  Kokoro TTS per segment (local, Apache-2.0 voice)
        ->  shot finder: SigLIP2 text->frame search over the source + Qwen-VL confirmation for every [SHOW]
        ->  4K render with numbered cards, callouts, subtitles  ->  thumbnail studio + upload text
"""
import json, os, re, string, subprocess

VOICE = "am_michael"


def parse_script(path):
    """script.md -> segments [{id, title, text, show}] in order (hook, points, closing sections)."""
    segs, cur, section = [], None, ""
    for raw in open(path, encoding="utf-8"):
        ln = raw.strip()
        if ln.startswith("## "):
            section = re.sub(r"\(.*?\)", "", ln[3:]).strip()
            section = re.sub(r"^\d+\.\s*", "", section)
            cur = dict(id=f"S{len(segs):02d}", title=string.capwords(section), text="", show="", point=None)
            segs.append(cur)
            continue
        m = re.match(r"\*\*Point (\d+)\s*[—-]\s*(.+?)\*\*", ln)
        if m:
            cur = dict(id=f"P{int(m.group(1)):02d}", title=m.group(2).strip().rstrip("."), text="", show="",
                       point=int(m.group(1)), section=section)
            segs.append(cur)
            continue
        if cur is None or not ln or ln.startswith(("#", "---", "*Credit", "**Voice", "**Target", "**Rules", "- ")):
            continue
        if ln.startswith("[SHOW"):
            cur["show"] = (cur["show"] + " " + ln.strip("[]").split(":", 1)[-1]).strip()
            continue
        cur["text"] = (cur["text"] + " " + ln).strip()
    return [s for s in segs if s["text"]]


def speak(segs, outdir, voice=VOICE, speed=1.0):
    """One wav per segment (24 kHz). Cached by text. Returns segs with wav + dur."""
    import numpy as np, soundfile as sf
    from kokoro import KPipeline
    os.makedirs(outdir, exist_ok=True)
    pipe = KPipeline(lang_code="b" if voice.startswith("b") else "a")
    for s in segs:
        p = os.path.join(outdir, f"{s['id']}.wav")
        key = os.path.join(outdir, f"{s['id']}.txt")
        if not (os.path.exists(p) and os.path.exists(key) and open(key, encoding="utf-8").read() == voice + "|" + s["text"]):
            audio = np.concatenate([a for _, _, a in pipe(s["text"], voice=voice, speed=speed)])
            audio = np.concatenate([np.zeros(int(0.15 * 24000), np.float32), audio, np.zeros(int(0.35 * 24000), np.float32)])
            sf.write(p, audio, 24000)
            open(key, "w", encoding="utf-8").write(voice + "|" + s["text"])
        s["wav"] = p
        s["dur"] = round(sf.info(p).duration, 2)
    return segs


# ------------------------------------------------------------------------------------------------ shot finder
VERIFY_PROMPT = ("These 3 frames are 1 second apart from a GTA 6 gameplay video. Do they clearly show: \"{want}\"? "
                 "Reply with JSON only: {{\"match\": 0-10 (10 = exactly this, clearly visible), \"seen\": what is "
                 "actually visible in under 15 words}}")


def frame_embeddings(video, workdir, step=1.0):
    """SigLIP2 image embeddings every `step` s (cached)."""
    import numpy as np, torch
    cache = os.path.join(workdir, f"siglip_{step}.npz")
    if os.path.exists(cache):
        z = np.load(cache); return z["t"], z["e"]
    from transformers import AutoModel, AutoProcessor
    from PIL import Image
    name = "google/siglip2-so400m-patch16-256"
    model = AutoModel.from_pretrained(name, dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained(name)
    W, H = 512, 288
    cmd = ["ffmpeg", "-v", "error", "-i", video, "-vf", f"fps={1 / step},scale={W}:{H}", "-pix_fmt", "rgb24",
           "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    embs, buf = [], []

    def flush():
        with torch.no_grad():
            x = proc(images=buf, return_tensors="pt")["pixel_values"].cuda().half()
            from .vision import _as_tensor
            e = _as_tensor(model.get_image_features(pixel_values=x))
            e = e / e.norm(dim=-1, keepdim=True)
        embs.append(e.float().cpu().numpy()); buf.clear()
    while True:
        b = p.stdout.read(W * H * 3)
        if len(b) < W * H * 3:
            break
        buf.append(Image.fromarray(np.frombuffer(b, np.uint8).reshape(H, W, 3)))
        if len(buf) == 64:
            flush()
    if buf:
        flush()
    p.wait()
    e = np.concatenate(embs); t = np.arange(len(e)) * step
    np.savez(cache, t=t, e=e)
    del model; torch.cuda.empty_cache()
    return t, e


def text_embeddings(texts):
    import numpy as np, torch
    from transformers import AutoModel, AutoProcessor
    name = "google/siglip2-so400m-patch16-256"
    model = AutoModel.from_pretrained(name, dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained(name)
    with torch.no_grad():
        x = proc(text=[f"a video game screenshot of {t}" for t in texts], padding="max_length", max_length=64,
                 truncation=True, return_tensors="pt").to("cuda")
        from .vision import _as_tensor
        e = _as_tensor(model.get_text_features(**x))
        e = e / e.norm(dim=-1, keepdim=True)
    out = e.float().cpu().numpy()
    del model; torch.cuda.empty_cache()
    return out


def find_shots(video, segs, workdir, judge, k=8, gap=6.0):
    """For every segment with a [SHOW] note: SigLIP2 top-k candidate seconds -> Qwen-VL verifies -> best second.
    Adds shot_t, shot_score, shot_seen. Cached in workdir/shots.json."""
    import numpy as np
    from .judge import grab, parse_json
    cache_p = os.path.join(workdir, "shots.json")
    cache = json.load(open(cache_p, encoding="utf-8")) if os.path.exists(cache_p) else {}
    t, e = frame_embeddings(video, workdir)
    want = [s for s in segs if s.get("show")]
    q = text_embeddings([re.split(r"→|;|\.", s["show"])[0].strip() + ", " + s["title"] for s in want])
    for s, qe in zip(want, q):
        key = s["id"] + "|" + s["show"]
        if key not in cache:
            sim = e @ qe
            cands = []
            for i in np.argsort(-sim):
                if all(abs(t[i] - c) >= gap for c in cands):
                    cands.append(float(t[i]))
                if len(cands) >= k:
                    break
            items = [(grab(video, [c - 1, c, c + 1], 640, 360, True), VERIFY_PROMPT.format(want=s["show"][:160]))
                     for c in cands]
            res = []
            for i in range(0, len(items), 4):
                res += judge.ask(items[i:i + 4], 60)
            best = (0, None, "")
            for c, r in zip(cands, res):
                d = parse_json(r) or {}
                try:
                    m = float(d.get("match", 0))
                except Exception:
                    m = 0
                if m > best[0]:
                    best = (m, c, str(d.get("seen", ""))[:120])
            cache[key] = dict(t=best[1], score=best[0], seen=best[2], cands=cands)
            json.dump(cache, open(cache_p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        c = cache[key]
        s["shot_t"], s["shot_score"], s["shot_seen"] = c["t"], c["score"], c["seen"]
    return segs

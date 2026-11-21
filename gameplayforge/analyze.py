"""Stage 1: one streaming pass over the recording -> per-sample features + embeddings (cached on disk)."""
import os, sys, subprocess, numpy as np, cv2
from .util import probe, hw_decode_ok, fmt
from . import vision

FW, FH = 384, 216          # frame size fed to the vision model (resized on GPU to the model's input size)
SW, SH = 96, 54            # tiny gray frames for cheap signals


def _decode_cmd(video, fps, hw):
    c = ["ffmpeg", "-v", "error"]
    if hw:
        c += ["-hwaccel", "cuda"]
    return c + ["-i", video, "-an", "-vf", f"fps={fps},scale={FW}:{FH}:flags=area", "-pix_fmt", "rgb24",
                "-f", "rawvideo", "-"]


def _audio_features(video, fps, n):
    cmd = ["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    rms = np.full(n, -90.0)
    if len(raw) > 16000:
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        win = 16000 // fps
        k = min(n, len(x) // win)
        w = x[:k * win].reshape(k, win)
        rms[:k] = 20 * np.log10(np.sqrt((w ** 2).mean(1)) + 1e-6)
    onset = np.maximum(0, np.diff(rms, prepend=rms[0]))
    return rms, onset


def run(video, cfg, workdir, force=False):
    cache = os.path.join(workdir, "analysis.npz")
    if os.path.exists(cache) and not force:
        print("  using cached analysis:", cache)
        F = dict(np.load(cache, allow_pickle=True))
        if str(F.get("prompt_version", "")) != vision.PROMPT_VERSION:
            if "emb" in F:
                print(f"  re-scoring cached frames with prompt set {vision.PROMPT_VERSION} (no video decode needed)")
                F = vision.rescore(F, cfg)
            else:
                F["P_scene"], F["P_mom"] = vision.heuristic_probs(F)
                F["prompt_version"] = vision.PROMPT_VERSION
            np.savez_compressed(cache, **F)
        return F
    os.makedirs(workdir, exist_ok=True)
    info = probe(video)
    fps = cfg["analysis"]["fps"]
    hw = cfg["hardware"]["decode"] in ("auto", "cuda") and hw_decode_ok()
    backend = vision.load_backend(cfg)
    print(f"  decode={'NVDEC' if hw else 'CPU'}  vision={'siglip2 ('+cfg['analysis']['model']+')' if backend else 'HEURISTIC (no visual AI)'}")
    p = subprocess.Popen(_decode_cmd(video, fps, hw), stdout=subprocess.PIPE, bufsize=10 ** 8)
    size, B = FW * FH * 3, (backend.batch if backend else 64)
    gray, letter, contrast, bright, embs, batch = [], [], [], [], [], []
    n = 0
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        f = np.frombuffer(buf, np.uint8).reshape(FH, FW, 3)
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        gray.append(cv2.resize(g, (SW, SH), interpolation=cv2.INTER_AREA))
        nb = max(2, int(FH * 0.07))
        top, bot = g[:nb], g[-nb:]
        letter.append(float(top.mean() < 12 and bot.mean() < 12 and top.std() < 6 and bot.std() < 6))
        contrast.append(float(g.std())); bright.append(float(g.mean()))
        if backend:
            batch.append(f)
            if len(batch) == B:
                embs.append(backend.encode_images(np.stack(batch))); batch = []
        n += 1
        if n % (fps * 300) == 0:
            print(f"    scanned {fmt(n / fps)} / {fmt(info['duration'])}", file=sys.stderr, flush=True)
    p.wait()
    if backend and batch:
        embs.append(backend.encode_images(np.stack(batch)))
    if n < fps * 10:
        raise RuntimeError("Video unreadable or too short")
    G = np.stack(gray).astype(np.float32)
    motion = np.concatenate([[0], np.abs(np.diff(G, axis=0)).mean((1, 2))])
    # automatic HUD detection: pixels that stay (almost) constant for the whole recording
    M, S = G.mean(0), G.std(0)
    mask = (S < np.percentile(S, 12)) & (M > 25)
    if mask.sum() < 20:
        mask = S < np.percentile(S, 5)
    dev = np.abs(G[:, mask] - M[mask]).mean(1)
    med = np.median(dev); mad = np.median(np.abs(dev - med)) + 1e-6
    hud_absent = np.clip((dev - med) / (6 * 1.4826 * mad), 0, 1)
    rms, onset = _audio_features(video, fps, n)
    feat = dict(fps=fps, n=n, motion=motion, contrast=np.array(contrast), brightness=np.array(bright),
                letterbox=np.array(letter), hud_absent=hud_absent, rms=rms[:n], onset=onset[:n],
                backend=("siglip2" if backend else "heuristic"), duration=info["duration"])
    if backend:
        E = np.concatenate(embs)[:n]
        P_scene, P_mom = vision.score_embeddings(E, backend)
        feat["emb"] = E.astype(np.float16)
        backend.close()
    else:
        P_scene, P_mom = vision.heuristic_probs(feat)
    feat["P_scene"], feat["P_mom"] = P_scene, P_mom
    feat["prompt_version"] = vision.PROMPT_VERSION
    np.savez_compressed(cache, **feat)
    return feat

"""4K renderer for breakdown videos (see breakdown.py for why the format is built this way).

Per point:  number card (our graphic)  ->  proof clip <= 4 s from the source at the verified moment (source audio low)
            ->  slow-zoom freeze-frame for the rest of the point (our lower-third + word-timed subtitles)
Always visible on source footage: "Footage: Rockstar Games - ..." credit. Moving source footage is reported as a
share of the runtime (target: well under a third - the commentary is the video).
"""
import os, re, subprocess

W4, H4, FPS = 3840, 2160, 30
CREDIT = "Footage: Rockstar Games - Grand Theft Auto VI: An Extended Look"


def _font(n, bold=True):
    from PIL import ImageFont
    cands = (["C:/Windows/Fonts/impact.ttf"] if bold else []) + ["C:/Windows/Fonts/segoeuib.ttf",
                                                                  "C:/Windows/Fonts/arialbd.ttf"]
    for f in cands:
        if os.path.exists(f):
            return ImageFont.truetype(f, n)
    return ImageFont.load_default()


def still(video, t, path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t):.2f}", "-i", video, "-frames:v", "1",
                    "-vf", f"scale={W4}:{H4}:flags=lanczos", path], check=True)
    return path


def card(seg, still_png, path):
    """Original number card: blurred dark still + big #N + title."""
    from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
    im = Image.open(still_png).convert("RGB").filter(ImageFilter.GaussianBlur(28))
    im = ImageEnhance.Brightness(im).enhance(0.35)
    d = ImageDraw.Draw(im)
    if seg.get("point"):
        d.text((300, 560), f"#{seg['point']}", font=_font(520), fill=(255, 196, 0), stroke_width=14,
               stroke_fill=(0, 0, 0))
    title = seg["title"].upper()
    f = _font(230)
    while d.textlength(title, font=f) > W4 - 600 and f.size > 110:
        f = _font(f.size - 10)
    d.text((300, 1240), title, font=f, fill=(255, 255, 255), stroke_width=10, stroke_fill=(0, 0, 0))
    if seg.get("section"):
        d.text((310, 1560), seg["section"].upper(), font=_font(90, bold=False), fill=(150, 170, 200))
    im.save(path)
    return path


def lower_third(seg, path):
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (W4, H4), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    label = (f"#{seg['point']}  " if seg.get("point") else "") + seg["title"].upper()
    if seg["title"].lower() not in ("hook", "outro"):      # no 'HOOK' / 'OUTRO' label on screen
        f = _font(96)
        tw = d.textlength(label, font=f)
        d.rounded_rectangle((120, 1500, 120 + tw + 120, 1660), radius=26, fill=(14, 20, 30, 230))
        d.rectangle((120, 1500, 150, 1660), fill=(255, 196, 0, 255))
        d.text((200, 1522), label, font=f, fill=(255, 255, 255, 255))
    cf = _font(54, bold=False)
    cw = d.textlength(CREDIT, font=cf)
    d.rounded_rectangle((W4 - cw - 160, 90, W4 - 90, 190), radius=18, fill=(0, 0, 0, 150))
    d.text((W4 - cw - 125, 108), CREDIT, font=cf, fill=(220, 225, 235, 255))
    im.save(path)
    return path


def word_caps(wav, model):
    """Word-timed caption chunks of our own narration (faster-whisper on the TTS wav), seconds from segment start."""
    segs, _ = model.transcribe(wav, language="en", word_timestamps=True)
    words = [(w.start, w.end, w.word.strip()) for s in segs for w in (s.words or [])]
    out, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= 4 or w[2].endswith((".", "?", "!", ",")):
            out.append((cur[0][0], cur[-1][1], " ".join(x[2] for x in cur))); cur = []
    if cur:
        out.append((cur[0][0], cur[-1][1], " ".join(x[2] for x in cur)))
    return out


def _esc(t):
    t = re.sub(r"[^A-Za-z0-9 ,.!?'#\-]", "", t)
    return t.replace("'", "\u2019")


def segment(video, seg, caps, tmp, enc, fallback_still, card_s=1.4, clip_max=4.0):
    dur = round(float(seg["dur"]) * FPS) / FPS          # whole frames: audio and video end together
    ok = seg.get("shot_t") is not None and (seg.get("shot_score") or 0) >= 6
    t0 = float(seg["clip_t"]) if seg.get("clip_t") is not None else max(0.0, float(seg.get("shot_t") or 0) - 1.0)
    cs = card_s if seg.get("point") else 0.0
    clip = min(float(seg.get("clip_max", clip_max)), max(0.0, dur - cs - 1.0)) if ok else 0.0
    rest = max(0.5, dur - cs - clip)
    st_t = float(seg["still_t"]) if seg.get("still_t") else t0 + clip       # hold a better frame if one was chosen
    st = still(video, st_t, os.path.join(tmp, f"{seg['id']}_still.png")) if ok else fallback_still
    inputs, fc, parts, n = [], [], [], 0
    if cs:
        cp = card(seg, st, os.path.join(tmp, f"{seg['id']}_card.png"))
        inputs += ["-loop", "1", "-t", f"{cs:.2f}", "-i", cp]
        fc.append(f"[{n}:v]scale={W4}:{H4},fps={FPS},format=yuv420p,setsar=1[v{n}]")
        parts.append(f"[v{n}]"); n += 1
    src = None
    if clip:
        inputs += ["-ss", f"{t0:.2f}", "-t", f"{clip:.2f}", "-i", video]
        fc.append(f"[{n}:v]scale={W4}:{H4}:flags=lanczos,fps={FPS},format=yuv420p,setsar=1[v{n}]")
        ms = int(cs * 1000)
        fc.append(f"[{n}:a]volume=0.22,aresample=48000,aformat=channel_layouts=stereo,adelay={ms}|{ms},apad[src]")
        src = n; parts.append(f"[v{n}]"); n += 1
    frames = max(1, int(rest * FPS))
    inputs += ["-loop", "1", "-t", f"{rest:.2f}", "-i", st]
    fc.append(f"[{n}:v]scale={W4}:{H4},zoompan=z='1+0.07*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
              f":d=1:s={W4}x{H4}:fps={FPS},format=yuv420p,setsar=1[v{n}]")
    parts.append(f"[v{n}]"); n += 1
    lt = lower_third(seg, os.path.join(tmp, f"{seg['id']}_lt.png"))
    inputs += ["-i", lt]; lt_i = n; n += 1
    inputs += ["-i", seg["wav"]]; nar = n; n += 1
    fc.append("".join(parts) + f"concat=n={len(parts)}:v=1:a=0[base]")
    v = f"[base][{lt_i}:v]overlay=0:0:enable='gte(t,{cs:.2f})'"
    fnt = "C\\:/Windows/Fonts/segoeuib.ttf"
    for a, b, txt in caps:
        tx = _esc(txt)
        if tx:
            v += (f",drawtext=fontfile='{fnt}':text='{tx}':fontsize=110:fontcolor=white:borderw=8:bordercolor=black"
                  f":x=(w-tw)/2:y=h-300:enable='between(t,{a:.2f},{b:.2f})'")
    # exact length on both tracks (stops A/V drift adding up over 20 segments)
    fc.append(v + f",tpad=stop_mode=clone:stop_duration=3,trim=duration={dur:.3f},setpts=PTS-STARTPTS[vout]")
    fc.append(f"[{nar}:a]aresample=48000,aformat=channel_layouts=stereo,apad[na]")
    if src is not None:
        fc.append(f"[na][src]amix=inputs=2:duration=first:normalize=0,atrim=0:{dur:.3f}[aout]")
    else:
        fc.append(f"[na]atrim=0:{dur:.3f}[aout]")
    out = os.path.join(tmp, f"{seg['id']}.mkv")     # PCM audio: no AAC padding per segment
    r = subprocess.run(["ffmpeg", "-v", "error", "-y"] + inputs + ["-filter_complex", ";".join(fc), "-map", "[vout]",
                        "-map", "[aout]", "-t", f"{dur:.3f}", "-r", str(FPS), "-fps_mode", "cfr"] + enc +
                       ["-c:a", "pcm_s16le", out], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{seg['id']}: {r.stderr[-800:]}")
    return out, clip, (st if ok else None)


def _sync(path, enc):
    """If video and audio lengths differ by > 0.1 s (frame rounding over many segments), stretch the picture
    evenly to the narration length (invisible 0.1-0.3 % retime) so captions stay on the words to the end."""
    def dur(kind):
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", kind, "-show_entries", "stream=duration",
                            "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
        return float(r or 0)
    v, a = dur("v:0"), dur("a:0")
    if v <= 0 or abs(v - a) <= 0.1:
        return
    tmp = path[:-4] + "_sync.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vf", f"setpts=PTS*{a / v:.6f}", "-r", str(FPS)] + enc +
                   ["-c:a", "copy", "-movflags", "+faststart", tmp], check=True)
    os.replace(tmp, path)
    print(f"  sync: picture {v:.2f}s -> narration {a:.2f}s")


def render_all(video, segs, outdir, cfg, name):
    import tempfile, shutil, torch
    from .util import pick_encoder
    tmp = tempfile.mkdtemp(prefix="gf_bd_")
    lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(lib)
    from faster_whisper import WhisperModel
    wm = WhisperModel("small", device="cuda", compute_type="float16")
    _, enc = pick_encoder(cfg)
    files, moving = [], 0.0
    fb = still(video, 60, os.path.join(tmp, "fallback.png"))
    for s in segs:
        caps = word_caps(s["wav"], wm)
        f, c, st = segment(video, s, caps, tmp, enc, fb)
        files.append(f); moving += c
        if st:
            fb = st
        print(f"  {s['id']} {float(s['dur']):5.1f}s  clip {c:.1f}s  {s['title'][:44]}", flush=True)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        for f in files:
            fh.write("file '" + f.replace("\\", "/") + "'\n")
    joined = os.path.join(tmp, "joined.mkv")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined],
                   check=True)
    out = os.path.join(outdir, name)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", joined, "-c:v", "copy", "-af",
                    "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out],
                   check=True)
    _sync(out, enc)
    total = sum(round(float(s["dur"]) * FPS) / FPS for s in segs)
    print(f"  breakdown: {total / 60:.1f} min; moving source footage {moving:.0f}s = {100 * moving / total:.0f}% "
          f"of the runtime (the rest is our narration over cards and freeze-frames)")
    shutil.rmtree(tmp, ignore_errors=True)
    return out, moving, total


# ------------------------------------------------------------------------------------------------ vertical Short
SW, SH = 1080, 1920


def short_render(video, segs, outdir, cfg, name, headline, shots, credit):
    """Vertical Short for a breakdown: blurred full-height background + the 16:9 picture in the middle (never
    cropped - the source creator's watermark stays visible), a big headline on top, word-timed captions under the
    picture, the source credit always on screen. `shots` = list of (source_t, is_clip) cycled through the narration:
    clips play <= 3 s, the rest slow-zooms on a still."""
    import tempfile, shutil, torch
    from PIL import Image, ImageDraw
    from .util import pick_encoder
    tmp = tempfile.mkdtemp(prefix="gf_bs_")
    lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(lib)
    from faster_whisper import WhisperModel
    wm = WhisperModel("small", device="cuda", compute_type="float16")
    _, enc = pick_encoder(cfg)
    # static overlay: headline (top) + credit (bottom)
    ov = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    words = headline.upper().split()
    lines, cur = [], ""
    f = _font(118)
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) > SW - 120 and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    lines.append(cur)
    y = 150
    for i, ln in enumerate(lines[:3]):
        tw = d.textlength(ln, font=f)
        d.text(((SW - tw) / 2, y), ln, font=f, fill=(255, 210, 63) if i == 0 else (255, 255, 255),
               stroke_width=9, stroke_fill=(0, 0, 0))
        y += 135
    cf = _font(34, bold=False)
    cw = d.textlength(credit, font=cf)
    d.rounded_rectangle(((SW - cw) / 2 - 24, SH - 230, (SW + cw) / 2 + 24, SH - 170), radius=16, fill=(0, 0, 0, 170))
    d.text(((SW - cw) / 2, SH - 222), credit, font=cf, fill=(230, 235, 245))
    ovp = os.path.join(tmp, "ov.png"); ov.save(ovp)
    files, k = [], 0
    for s in segs:
        dur = round(float(s["dur"]) * FPS) / FPS
        t, is_clip = shots[k % len(shots)]; k += 1
        caps = word_caps(s["wav"], wm)
        inputs, fc = [], []
        if is_clip:
            clip = min(3.0, dur)
            inputs += ["-ss", f"{t:.2f}", "-t", f"{clip:.2f}", "-i", video]
            st = still(video, t + clip, os.path.join(tmp, f"{s['id']}_st.png"))
            inputs += ["-loop", "1", "-t", f"{max(0.3, dur - clip):.2f}", "-i", st]
            fc.append(f"[0:v]scale=1920:1080,fps={FPS},format=yuv420p,setsar=1[a0];"
                      f"[1:v]scale=1920:1080,fps={FPS},format=yuv420p,setsar=1[a1];[a0][a1]concat=n=2:v=1:a=0[pic]")
            nxt = 2
        else:
            st = still(video, t, os.path.join(tmp, f"{s['id']}_st.png"))
            fr = max(1, int(dur * FPS))
            inputs += ["-loop", "1", "-t", f"{dur:.2f}", "-i", st]
            fc.append(f"[0:v]scale=1920:1080,zoompan=z='1+0.06*on/{fr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                      f":d=1:s=1920x1080:fps={FPS},format=yuv420p,setsar=1[pic]")
            nxt = 1
        inputs += ["-i", ovp, "-i", s["wav"]]
        fc.append(f"[pic]split[p1][p2];[p1]scale={SW}:{SH}:force_original_aspect_ratio=increase,crop={SW}:{SH},"
                  f"boxblur=40:4,eq=brightness=-0.25[bg];[p2]scale={SW}:-2[fg];[bg][fg]overlay=0:(H-h)/2-60[c];"
                  f"[c][{nxt}:v]overlay=0:0[v0]")
        v = "[v0]null"
        fnt = "C\:/Windows/Fonts/segoeuib.ttf"
        for a, b, txt in caps:
            tx = _esc(txt).upper()
            if tx:
                fs = max(40, min(76, int((SW - 100) / (0.64 * max(1, len(tx))))))     # always fits the width
                v += (f",drawtext=fontfile='{fnt}':text='{tx}':fontsize={fs}:fontcolor=white:borderw=7:bordercolor=black"
                      f":x=(w-tw)/2:y=1300:enable='between(t,{a:.2f},{b:.2f})'")
        fc.append(v + f",tpad=stop_mode=clone:stop_duration=3,trim=duration={dur:.3f},setpts=PTS-STARTPTS[vout]")
        fc.append(f"[{nxt + 1}:a]aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:{dur:.3f}[aout]")
        out = os.path.join(tmp, f"{s['id']}.mkv")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y"] + inputs + ["-filter_complex", ";".join(fc), "-map", "[vout]",
                            "-map", "[aout]", "-t", f"{dur:.3f}", "-r", str(FPS), "-fps_mode", "cfr"] + enc +
                           ["-c:a", "pcm_s16le", out], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"short {s['id']}: {r.stderr[-800:]}")
        files.append(out)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        for fl in files:
            fh.write("file '" + fl.replace("\\", "/") + "'\n")
    joined = os.path.join(tmp, "joined.mkv")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined], check=True)
    out = os.path.join(outdir, name)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", joined, "-c:v", "copy", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return out

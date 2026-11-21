"""Stage 5: rendering. All encoding is done by FFmpeg (NVENC when the GPU/driver supports it)."""
import os, shutil, subprocess, tempfile
from .util import run, pick_encoder, probe, fmt


def grade_filter(cfg):
    g = cfg["style"]["grade"]
    return f"eq=contrast={g['contrast']}:saturation={g['saturation']},unsharp=5:5:{g['sharpen']}:5:5:0.0"


def short_filter(cfg, src_w, src_h):
    s = cfg["style"]["short"]; W, H, lay = s["width"], s["height"], s["layout"]
    if lay == "crop34_black":      # reference layout: 3:4 centre band on a black 9:16 canvas
        bh = int(W * 4 / 3) // 2 * 2
        return (f"crop='min(iw,ih*3/4)':ih,scale={W}:{bh}:flags=lanczos,"
                f"pad={W}:{H}:0:({H}-{bh})/2:black,setsar=1")
    if lay == "full_crop":
        return f"crop='min(iw,ih*9/16)':ih,scale={W}:{H}:flags=lanczos,setsar=1"
    return None   # blur layout handled with filter_complex


def cut_clip(video, st, dur, out, cfg, vertical=False, has_audio=True, info=None):
    _, enc = pick_encoder(cfg)
    gr = grade_filter(cfg)
    fade = 0.15
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{st:.3f}", "-t", f"{dur:.3f}", "-i", video]
    if vertical and cfg["style"]["short"]["layout"] == "blur":
        W, H = cfg["style"]["short"]["width"], cfg["style"]["short"]["height"]
        fc = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30:5[bg];"
              f"[0:v]scale={W}:-2[fg];[bg][fg]overlay=0:(H-h)/2,{gr},setsar=1[v]")
        cmd += ["-filter_complex", fc, "-map", "[v]"]
    else:
        vf = (short_filter(cfg, info["width"], info["height"]) + "," if vertical else "") + gr
        if not vertical:
            vf += ",setsar=1"
        cmd += ["-map", "0:v:0", "-vf", vf]
    if has_audio:
        cmd += ["-map", "0:a:0", "-af", f"afade=t=in:st=0:d={fade},afade=t=out:st={max(0, dur - fade):.3f}:d={fade}",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    else:
        cmd += ["-an"]
    cmd += ["-r", str(int(round(info["fps"] if not vertical else min(info["fps"], 60))))] + enc + [out]
    run(cmd)


def concat_and_master(files, out, loud, tmp, has_audio=True):
    lst = os.path.join(tmp, "concat.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in files:
            f.write("file '" + os.path.abspath(p).replace("\\", "/").replace("'", r"'\''") + "'\n")
    raw = os.path.join(tmp, "joined.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", raw])
    if has_audio:   # loudness-normalise the game audio only; video stream is copied untouched
        run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-c:v", "copy", "-af",
             f"loudnorm=I={loud}:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
             "-movflags", "+faststart", out])
    else:
        run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-c", "copy", "-movflags", "+faststart", out])


def _long_size(cfg, info):
    lc = cfg["style"]["long"]
    if lc.get("upscale_4k") and info["height"] < 2160:
        return 3840, 2160
    return info["width"] // 2 * 2, info["height"] // 2 * 2


def render_group(video, group, out, cfg, info, vertical=False):
    """Render one continuous group: pieces joined by dissolve or cut, slow-motion/punch-in pieces, dips to black
    at the edges. One FFmpeg call, frame-accurate, NVENC."""
    _, enc = pick_encoder(cfg)
    lc = cfg["style"]["long"]; xd = float(lc["xfade"]); fps = 60 if info["fps"] >= 50 else int(round(info["fps"]))
    has_a = info["has_audio"]
    if vertical:
        lay = short_filter(cfg, info["width"], info["height"]) or \
            f"scale={cfg['style']['short']['width']}:{cfg['style']['short']['height']}"
        sharpen = cfg["style"]["grade"]["sharpen"]
        cas = float(cfg["style"]["grade"].get("cas", 0))
    else:
        W, H = _long_size(cfg, info)
        lay = f"scale={W}:{H}:flags=lanczos"
        sharpen = lc.get("sharpen", cfg["style"]["grade"]["sharpen"])
        cas = float(lc.get("cas", 0))
    g = cfg["style"]["grade"]
    pre = f"cas=strength={cas}," if cas > 0 else ""      # sharpen at native resolution, before any scaling
    grade = f"eq=contrast={g['contrast']}:saturation={g['saturation']},unsharp=5:5:{sharpen}:5:5:0.0"
    cmd = ["ffmpeg", "-y", "-v", "error"]
    fc, lens = [], []
    for k, p in enumerate(group["pieces"]):
        d = p["end"] - p["start"]
        cmd += ["-ss", f"{p['start']:.3f}", "-t", f"{d:.3f}", "-i", video]
        sp, z = float(p.get("speed", 1.0)), float(p.get("zoom", 1.0))
        vf = f"[{k}:v]setpts=(PTS-STARTPTS)/{sp}"
        if pre:
            vf += "," + pre[:-1]
        cx, cy = float(p.get("cx", 0.5)), float(p.get("cy", 0.5))
        iw, ih = info["width"], info["height"]
        if z > 1.001 and p.get("ramp"):
            # smooth punch-in: zoom eases 1 -> z over 0.5 s, holds, eases back to 1 over the last 0.5 s, aimed at
            # (cx, cy) - starts and ends at full frame so the joins on both sides are invisible
            nf = max(2, int(round(d / sp * fps))); rf = max(1, int(0.5 * fps))
            u = f"min(1,min(on/{rf},({nf}-1-on)/{rf}))"
            zexp = f"1+({z - 1:.3f})*({u})*({u})*(3-2*({u}))"
            vf += (f",fps={fps},zoompan=z='{zexp}':d=1:s={iw}x{ih}:fps={fps}"
                   f":x='max(0,min(iw-iw/zoom,{cx:.3f}*iw-iw/zoom/2))'"
                   f":y='max(0,min(ih-ih/zoom,{cy:.3f}*ih-ih/zoom/2))'")
        elif z > 1.001:
            vf += (f",crop=iw/{z}:ih/{z}:'max(0,min(iw-ow,{cx:.3f}*iw-ow/2))':'max(0,min(ih-oh,{cy:.3f}*ih-oh/2))'"
                   f",scale={iw}:{ih}:flags=lanczos")
        lay_k = lay
        if vertical and "cx" in p and z <= 1.001 and cfg["style"]["short"]["layout"] == "crop34_black":
            # the 3:4 band follows the action instead of always taking the centre (off-centre kills stay in frame)
            lay_k = lay.replace("crop='min(iw,ih*3/4)':ih,",
                                f"crop='min(iw,ih*3/4)':ih:'max(0,min(iw-ow,{cx:.3f}*iw-ow/2))':0,", 1)
        # every piece is forced to EXACTLY d/sp seconds on both tracks: on the 2026-11-08 stream short pieces came
        # out with less audio than video, and the drift added up (1.8 s in a 14 s short)
        Lk = d / sp
        vf += (f",{lay_k},{grade},fps={fps},format=yuv420p,setsar=1,"
               f"tpad=stop_mode=clone:stop_duration=1,trim=duration={Lk:.4f},setpts=PTS-STARTPTS,settb=AVTB[v{k}]")
        fc.append(vf)
        if has_a:
            af = f"[{k}:a]asetpts=PTS-STARTPTS"
            if abs(sp - 1) > 1e-3:
                af += f",atempo={sp}"
            af += (f",aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                   f"apad,atrim=0:{Lk:.4f},asetpts=PTS-STARTPTS[a{k}]")
            fc.append(af)
        lens.append(d / sp)
    accv, acca, L = "v0", "a0", lens[0]
    starts = [0.0]                                          # output time where each piece begins
    for k in range(1, len(lens)):
        join = group["pieces"][k].get("join", "xfade")
        tr = group["pieces"][k].get("trans", "fade")       # "fade" dissolve, or a fast whip (e.g. smoothleft)
        dd = min(float(group["pieces"][k].get("trans_d", xd)), lens[k] / 2, L / 2)
        starts.append(L - dd if join == "xfade" and dd > 0.05 else L)
        if join == "xfade" and dd > 0.05:
            fc.append(f"[{accv}][v{k}]xfade=transition={tr}:duration={dd:.3f}:offset={L - dd:.3f}[xv{k}]")
            if has_a:
                fc.append(f"[{acca}][a{k}]acrossfade=d={dd:.3f}[xa{k}]")
            L += lens[k] - dd
        else:
            if has_a:
                fc.append(f"[{accv}][{acca}][v{k}][a{k}]concat=n=2:v=1:a=1[xv{k}][xa{k}]")
            else:
                fc.append(f"[{accv}][v{k}]concat=n=2:v=1:a=0[xv{k}]")
            L += lens[k]
        accv, acca = f"xv{k}", f"xa{k}"
    fi, fo = float(group.get("fade_in", 0)), float(group.get("fade_out", 0))
    vend = f"[{accv}]null"
    if vertical:
        vend += _short_text(group, starts, cfg, L)
    if fi > 0: vend += f",fade=t=in:st=0:d={fi}"
    if fo > 0: vend += f",fade=t=out:st={max(0, L - fo):.3f}:d={fo}"
    fc.append(vend + "[vout]")
    if has_a:
        aend = f"[{acca}]anull"
        if fi > 0: aend += f",afade=t=in:st=0:d={fi}"
        if fo > 0: aend += f",afade=t=out:st={max(0, L - fo):.3f}:d={fo}"
        fc.append(aend + "[aout]")
    cmd += ["-filter_complex", ";".join(fc), "-map", "[vout]"]
    cmd += (["-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"] if has_a else ["-an"])
    cmd += ["-r", str(fps)] + enc + [out]
    run(cmd)
    return L


def _font():
    for f in ("C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf"):
        if os.path.exists(f):
            return f.replace(":", "\\:")
    return ""


def _safe(s):
    """drawtext-safe text: letters, digits, spaces and ! ? - only (no quotes / colons / emoji)."""
    import re
    return re.sub(r"[^A-Za-z0-9 !?\-]", "", str(s)).strip().upper()


def src_to_out(group, starts, t):
    """Output time of source time t inside a rendered group (None when t was cut out)."""
    for k, p in enumerate(group["pieces"]):
        if p["start"] <= t < p["end"]:
            return starts[k] + (t - p["start"]) / float(p.get("speed", 1.0))
    return None


def _short_text(group, starts, cfg, L):
    """Shorts on the crop34_black layout have empty black bands above and below the picture. The research pass
    (2026-11-13: gaming-editing tutorials + the playbook) agrees a Short needs a bold caption in the first 3 s and
    on-screen text that tells the story; it is also original work on top of the footage (reused-content policy).
      * HEADLINE in the top band for the whole short (the measured hook, e.g. '3 KILLS IN 9 SECONDS')
      * KILL COUNTER in the bottom band, popping in on every confirmed kill ('1 KILL', '2 KILLS', ... 'ACE')."""
    tc = cfg["style"]["short"].get("text") or {}
    if not tc.get("enabled", True) or cfg["style"]["short"]["layout"] != "crop34_black":
        return ""
    W, H = cfg["style"]["short"]["width"], cfg["style"]["short"]["height"]
    band = (H - int(W * 4 / 3) // 2 * 2) // 2
    fnt = _font(); out = ""
    head = _safe(group.get("headline", ""))
    if head and band >= 120:
        fs = int(min(band * 0.42, W * 1.6 / max(8, len(head))))
        out += (f",drawtext=fontfile='{fnt}':text='{head}':fontsize={fs}:fontcolor=white:borderw={max(3, fs // 12)}"
                f":bordercolor=black:x=(w-tw)/2:y={band // 2}-th/2:alpha='min(1,t/0.25)'")
    caps = []                                           # burned-in subtitles (cinematic moments): bottom band
    for t0, t1, txt in group.get("captions") or []:
        a = src_to_out(group, starts, t0)
        if a is None:
            continue
        caps.append((a, a + max(0.5, min(3.0, t1 - t0 + 0.15)), _safe(txt)))
    for i, (a, b, txt) in enumerate(caps):
        if i + 1 < len(caps):
            b = min(b, caps[i + 1][0])                  # never two captions at once
        if not txt or b - a < 0.2:
            continue
        fs = int(min(band * 0.34, W * 1.7 / max(8, len(txt))))
        out += (f",drawtext=fontfile='{fnt}':text='{txt}':fontsize={fs}:fontcolor=white:borderw={max(3, fs // 9)}"
                f":bordercolor=black:x=(w-tw)/2:y={H - band // 2}-th/2:enable='between(t,{a:.3f},{b:.3f})'")
    big = group.get("big_word")                         # (n, word) e.g. (4, 'SQUAD WIPE') / (5, 'ACE')
    times = [src_to_out(group, starts, t) for t in group.get("kill_times", [])]
    times = [t for t in times if t is not None]
    for i, t in enumerate(times, 1):
        end = times[i] if i < len(times) else L
        dur = max(0.6, min(1.6, end - t - 0.05))
        word = big[1] if big and i >= big[0] and i == len(times) else f"{i} KILL{'S' * (i > 1)}"
        fs = int(band * 0.5)
        y0 = H - band // 2
        # pop: scale-in look via quick fade, gold colour, thick black border
        out += (f",drawtext=fontfile='{fnt}':text='{_safe(word)}':fontsize={fs}:fontcolor=0xFFD23F"
                f":borderw={max(3, fs // 10)}:bordercolor=black:x=(w-tw)/2:y={y0}-th/2"
                f":enable='between(t,{t:.3f},{t + dur:.3f})':alpha='min(1,(t-{t:.3f})/0.12)'")
    return out


def render_intro(video, kill_t, out, cfg, info):
    """Cinematic intro (from the owner's reference 'Royalty | PUBG MOBILE Montage', shortened: a 70 s intro loses
    viewers): a dark, desaturated slow-motion teaser of the video's best kill, then the channel name glowing on
    black, then straight into the opening fight. Returns its length."""
    ic = cfg["style"]["intro"]; _, enc = pick_encoder(cfg)
    W, H = _long_size(cfg, info)
    fps = 60 if info["fps"] >= 50 else int(round(info["fps"]))
    tl, sl = float(ic.get("teaser", 3.0)), float(ic.get("slow", 0.5))
    nl = float(ic.get("name_len", 2.4))
    src = tl * sl
    name = str(ic.get("name", "")).upper().replace("'", "").replace(":", "")
    fnt = _font(); fs = int(H * 0.13)
    txt = (f"drawtext=fontfile='{fnt}':text='{name}':fontsize={fs}:fontcolor=white:x=(w-tw)/2:y=(h-th)/2"
           f":alpha='min(1,t/0.5)*min(1,({nl}-t)/0.4)'")
    glow = (f"drawtext=fontfile='{fnt}':text='{name}':fontsize={fs}:fontcolor=0x8a2be2:x=(w-tw)/2:y=(h-th)/2"
            f":alpha='min(1,t/0.5)*min(1,({nl}-t)/0.4)',gblur=sigma=18")
    fc = (f"[0:v]setpts=(PTS-STARTPTS)/{sl},scale={W}:{H}:flags=lanczos,eq=brightness=-0.06:saturation=0.55:contrast=1.1,"
          f"fps={fps},fade=t=in:st=0:d=0.6,fade=t=out:st={tl - 0.5:.2f}:d=0.5,format=yuv420p,setsar=1[t];"
          f"color=c=black:s={W}x{H}:r={fps}:d={nl},format=gbrp,{glow}[g];"
          f"color=c=black:s={W}x{H}:r={fps}:d={nl},format=gbrp,{txt}[n];"
          f"[g][n]blend=all_mode=screen,format=yuv420p,setsar=1[nm];"
          f"[0:a]asetpts=PTS-STARTPTS,atempo={sl},aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
          f"afade=t=in:d=0.6,afade=t=out:st={tl - 0.5:.2f}:d=0.5[ta];"
          f"anullsrc=r=48000:cl=stereo,atrim=0:{nl},aformat=sample_fmts=fltp:channel_layouts=stereo[na];"
          f"[t][ta][nm][na]concat=n=2:v=1:a=1[v][a]")
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0, kill_t - src * 0.7):.3f}", "-t", f"{src:.3f}", "-i", video,
         "-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
         "-r", str(fps)] + enc + [out])
    return tl + nl


def add_music(path, music, cfg, tmp):
    """Music bed under the whole montage (reference style); gunshots and kills stay audible on top."""
    mc = cfg["style"]["music_mix"]
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                               capture_output=True, text=True).stdout)
    tmpo = os.path.join(tmp, "music_mix.mp4")
    g, m = float(mc.get("game_gain", 0.55)), float(mc.get("music_gain", 0.8))
    fc = (f"[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume={m},"
          f"afade=t=in:d=1.0,afade=t=out:st={max(0, dur - 3):.2f}:d=3,atrim=0:{dur:.3f}[mu];"
          f"[0:a]volume={g}[ga];[ga][mu]amix=inputs=2:duration=first:normalize=0,"
          f"loudnorm=I={mc.get('lufs', -10)}:TP=-1.0:LRA=9[a]")
    run(["ffmpeg", "-y", "-v", "error", "-i", path, "-stream_loop", "-1", "-i", music, "-filter_complex", fc,
         "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", tmpo])
    shutil.move(tmpo, path)


def render_long(video, groups, outdir, cfg, info, name="Cinematic_Montage.mp4"):
    os.makedirs(outdir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="gf_long_")
    files, marks, t = [], [], 0.0
    ic = cfg["style"].get("intro") or {}
    cold = next((g for g in groups if g.get("cold") and g.get("kill_t") is not None), None)
    if ic.get("enabled") and cold and info["has_audio"]:
        p = os.path.join(tmp, "intro.mp4")
        t += render_intro(video, cold["kill_t"], p, cfg, info)
        files.append(p)
    for i, g in enumerate(groups):
        p = os.path.join(tmp, f"g{i:03d}.mp4")
        L = render_group(video, g, p, cfg, info)
        if not g.get("cold"):
            marks.append((t, g["title"]))
        files.append(p); t += L
    out = os.path.join(outdir, name)
    concat_and_master(files, out, cfg["style"]["long"]["loudness_lufs"], tmp, info["has_audio"])
    music = str(cfg["style"].get("music") or "none")
    if music.lower() != "none" and os.path.exists(music) and info["has_audio"]:
        add_music(out, music, cfg, tmp)
    elif music.lower() != "none":
        print(f"  NOTE: music file not found ({music}) - montage keeps game audio only")
    # YouTube chapters: first at 0:00, each >= 10 s, at least 3 - otherwise YouTube ignores them
    ch = []
    for ts, title in marks:
        if not ch or ts - ch[-1][0] >= 10:
            ch.append([ts, title])
    if ch:
        ch[0][0] = 0
    chap_name = "YouTube_Chapters.txt" if name == "Cinematic_Montage.mp4" else name[:-4] + "_Chapters.txt"
    with open(os.path.join(outdir, chap_name), "w", encoding="utf-8") as f:
        if len(ch) >= 3:
            for ts, title in ch:
                f.write(f"{fmt(ts)} {title}\n")
        else:
            f.write("(fewer than 3 sections - YouTube needs at least 3 chapters, so none are suggested)\n")
    shutil.rmtree(tmp, ignore_errors=True)
    return out, marks


def render_shorts(video, shorts, outdir, cfg, info):
    os.makedirs(outdir, exist_ok=True)
    outs = []
    for name, sp in shorts:
        tmp = tempfile.mkdtemp(prefix="gf_short_")
        p = os.path.join(tmp, "s.mp4")
        render_group(video, sp, p, cfg, info, vertical=True)
        out = os.path.join(outdir, name + ".mp4")
        concat_and_master([p], out, cfg["style"]["long"]["loudness_lufs"], tmp, info["has_audio"])
        shutil.rmtree(tmp, ignore_errors=True)
        outs.append(out)
    return outs


def extract_cutscenes(video, cuts, outdir, cfg, info):
    """Frame-accurate cut (a stream copy can only start on a keyframe, i.e. up to several seconds early).
    Re-encoded with NVENC at the configured quality, no colour grade."""
    os.makedirs(outdir, exist_ok=True)
    _, enc = pick_encoder(cfg)
    outs = []
    for i, c in enumerate(cuts, 1):
        p = os.path.join(outdir, f"Cutscene_{i:03d}.mp4")
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{c['start']:.3f}", "-t", f"{c['end'] - c['start']:.3f}",
               "-i", video, "-map", "0:v:0"]
        cmd += (["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k"] if info["has_audio"] else ["-an"])
        run(cmd + enc + ["-movflags", "+faststart", p])
        outs.append(p)
    return outs

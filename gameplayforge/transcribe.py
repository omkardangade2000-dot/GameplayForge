"""Speech -> text with word timings (faster-whisper on the GPU, models cached locally). Cached per video.

Used by cinematic mode for three things:
  * the DIALOGUE of every cutscene (what the scene is about, meme-able lines, spoiler words)
  * matching a scene to the trend radar's moment keys (character names, quoted lines people search for)
  * burned-in subtitles on moment Shorts (captions are on-screen original work and lift retention)
The stream mix has game dialogue AND the player's commentary in one track - both are transcribed together.
"""
import json, os, subprocess, tempfile

VERSION = "t1"


def _load_model(name):
    import torch
    lib = os.path.join(os.path.dirname(torch.__file__), "lib")       # cuBLAS / cuDNN shipped with torch
    if os.path.isdir(lib) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(lib)
    from faster_whisper import WhisperModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return WhisperModel(name, device=dev, compute_type="float16" if dev == "cuda" else "int8")


def transcribe(video, workdir, cfg, spans=None):
    """Return a list of segments {start, end, text, words:[(start, end, word)]} for the whole video (or only the
    given (a, b) spans, merged). Cached in workdir/transcript.json."""
    tc = cfg.get("transcribe") or {}
    model = tc.get("model", "small")
    lang = tc.get("language") or None                  # None = auto-detect (channel is global / English)
    cache = os.path.join(workdir, "transcript.json")
    key = f"{VERSION}|{model}|{lang}|{spans}"
    if os.path.exists(cache):
        c = json.load(open(cache, encoding="utf-8"))
        if c.get("key") == key:
            return c["segments"]
    tmp = tempfile.mkdtemp(prefix="gf_asr_")
    wav = os.path.join(tmp, "a.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000", wav], check=True)
    m = _load_model(model)
    clip = []
    for a, b in (spans or []):
        clip += [round(a, 2), round(b, 2)]
    segs, info = m.transcribe(wav, language=lang, word_timestamps=True, vad_filter=True,
                              clip_timestamps=clip if clip else "0", condition_on_previous_text=False)
    out = []
    for s in segs:
        out.append(dict(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip(),
                        words=[(round(w.start, 2), round(w.end, 2), w.word.strip()) for w in (s.words or [])]))
    json.dump(dict(key=key, language=info.language, segments=out), open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    try:
        os.remove(wav); os.rmdir(tmp)
    except OSError:
        pass
    print(f"  transcript: {len(out)} lines of speech ({info.language}), model {model}")
    return out


def text_between(segs, a, b):
    return " ".join(s["text"] for s in segs if s["end"] > a and s["start"] < b).strip()


def captions(segs, a, b, max_words=4):
    """Short caption chunks (<= max_words words) inside [a, b): [(t0, t1, text)] in source time."""
    out, cur = [], []
    for s in segs:
        for w0, w1, w in s.get("words") or []:
            if w1 <= a or w0 >= b or not w:
                continue
            cur.append((w0, w1, w))
            if len(cur) >= max_words or w.endswith((".", "?", "!", ",")):
                out.append((cur[0][0], cur[-1][1], " ".join(x[2] for x in cur))); cur = []
        if cur:
            out.append((cur[0][0], cur[-1][1], " ".join(x[2] for x in cur))); cur = []
    return out

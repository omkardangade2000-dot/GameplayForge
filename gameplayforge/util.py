import json, os, subprocess, sys, shutil, yaml


def load_config(path=None):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = path or os.path.join(here, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base, over):
    """Per-game overrides (config.yaml games.<profile>.style_overrides) on top of the global settings."""
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def run(cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(map(str, cmd)) + "\n" + r.stderr[-2000:])
    return r


def probe(path):
    r = run(["ffprobe", "-v", "error", "-show_entries",
             "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,sample_rate,channels:format=duration,size",
             "-of", "json", path])
    j = json.loads(r.stdout)
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise RuntimeError(f"No video stream in {path}")
    num, den = (v.get("avg_frame_rate") or v["r_frame_rate"]).split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return {"width": int(v["width"]), "height": int(v["height"]), "fps": fps,
            "duration": float(j["format"]["duration"]), "has_audio": a is not None,
            "size": int(j["format"].get("size", 0))}


def fmt(t, hours=False):
    t = max(0, int(round(t)))
    h, m, s = t // 3600, t % 3600 // 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if (h or hours) else f"{m}:{s:02d}"


_enc_cache = {}


def pick_encoder(cfg):
    """Return (name, ffmpeg args). Tests NVENC for real instead of assuming."""
    want = cfg["hardware"]["encoder"]
    if want in _enc_cache:
        return _enc_cache[want]
    cands = [want] if want != "auto" else ["h264_nvenc", "libx264"]
    cq, preset = cfg["hardware"]["cq"], cfg["hardware"]["preset"]
    for c in cands:
        if c.endswith("_nvenc"):
            test = run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
                        "-c:v", c, "-f", "null", "-"], check=False)
            if test.returncode != 0:
                continue
            args = ["-c:v", c, "-preset", preset, "-tune", "hq", "-rc", "vbr", "-cq", str(cq), "-b:v", "0",
                    "-pix_fmt", "yuv420p"]
            mr = str(cfg["hardware"].get("maxrate") or "")
            if mr:                      # quality-based, but never above YouTube's upload spec
                n = float(mr.rstrip("Mm"))
                args += ["-maxrate", mr, "-bufsize", f"{int(n * 2)}M"]
            if c == "h264_nvenc":
                args += ["-profile:v", "high"]
            if c == "hevc_nvenc":
                args += ["-tag:v", "hvc1"]
        else:
            args = ["-c:v", "libx264", "-preset", "veryfast" if cq >= 19 else "medium",
                    "-crf", str(cq), "-pix_fmt", "yuv420p"]
        _enc_cache[want] = (c, args)
        return _enc_cache[want]
    raise RuntimeError("No usable video encoder found")


def hw_decode_ok():
    r = run(["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
             "-f", "null", "-"], check=False)
    return r.returncode == 0

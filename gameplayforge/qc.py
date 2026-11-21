"""Stage 6: quality control. Every check is measured on the rendered file, nothing is assumed."""
import os, re, subprocess, json
from .util import run, probe


def _blackframes(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", path, "-vf", "blackdetect=d=0.4:pix_th=0.06", "-an",
                        "-f", "null", "-"], capture_output=True, text=True)
    return len(re.findall(r"black_start", r.stderr))


def _loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", path, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
    return float(m[-1]) if m else None


def check(path, expect, label):
    res = {"file": os.path.basename(path), "label": label, "ok": True, "notes": []}
    if not os.path.exists(path) or os.path.getsize(path) < 10_000:
        return dict(res, ok=False, notes=["file missing or empty"])
    try:
        info = probe(path)
    except RuntimeError as e:
        return dict(res, ok=False, notes=[str(e)])
    res.update(width=info["width"], height=info["height"], fps=round(info["fps"], 2),
               duration=round(info["duration"], 1))
    if "size" in expect and (info["width"], info["height"]) != tuple(expect["size"]):
        res["ok"] = False; res["notes"].append(f"resolution {info['width']}x{info['height']} != {expect['size']}")
    if "min_s" in expect and not (expect["min_s"] <= info["duration"] <= expect["max_s"]):
        res["notes"].append(f"duration {info['duration']:.0f}s outside {expect['min_s']}-{expect['max_s']}s")
        if expect.get("strict", True): res["ok"] = False
    if expect.get("audio") and not info["has_audio"]:
        res["ok"] = False; res["notes"].append("audio stream missing")
    if info["has_audio"]:
        v = float(run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=duration",
                       "-of", "csv=p=0", path]).stdout.split()[0])
        a = float(run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=duration",
                       "-of", "csv=p=0", path]).stdout.split()[0])
        res["av_drift_s"] = round(abs(v - a), 3)
        if abs(v - a) > 0.25:
            res["ok"] = False; res["notes"].append(f"A/V length drift {abs(v - a):.2f}s")
        res["loudness_lufs"] = _loudness(path)
    bf = _blackframes(path)
    res["black_segments"] = bf
    if bf: res["notes"].append(f"{bf} black segment(s) >0.4s - inspect")
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"], capture_output=True, text=True)
    if r.stderr.strip():
        res["ok"] = False; res["notes"].append("decode errors: " + r.stderr.strip()[:120])
    return res


def contact_sheet(path, out_png, n=8):
    d = probe(path)["duration"]
    run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vf",
         f"fps={n / max(d, 1):.5f},scale=240:-2,tile={min(n, 4)}x{(n + 3) // 4}", "-frames:v", "1", out_png],
        check=False)


def overlap_check(plan_clips):
    """No accidental repetition: no two selected source ranges may overlap."""
    s = sorted(plan_clips, key=lambda c: c["start"]); bad = []
    for a, b in zip(s, s[1:]):
        if b["start"] < a["end"] - 0.05: bad.append((a["start"], b["start"]))
    return bad

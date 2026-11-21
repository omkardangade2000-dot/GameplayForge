"""Stage 2: cutscene / cinematic segmentation.

Candidate cutscenes come mainly from what the vision model SEES (cutscene + conversation probability).
HUD disappearance and letterbox bars only support that evidence - on their own they can never make a
cutscene, because games also hide the HUD in restricted areas and during kill animations.
Every candidate is then checked by the Qwen-VL judge (judge.verify_cutscenes):
  story cutscene / conversation -> Cutscenes folder
  kill animation / gameplay     -> stays available for the montage
"""
import numpy as np
from scipy.ndimage import median_filter
from .vision import SCENE, MOMENT


def fused_confidence(F):
    P, M = F["P_scene"], F["P_mom"]
    if str(F["backend"]) == "heuristic":
        return 0.6 * F["hud_absent"] + 0.4 * F["letterbox"]
    cut_p = P[:, SCENE.index("cutscene")]
    talk = M[:, MOMENT.index("dialogue")]
    return 0.6 * cut_p + 0.15 * talk + 0.15 * F["hud_absent"] + 0.1 * F["letterbox"]


def find_cutscenes(F, on=0.55, off=0.35, min_len=3.0, merge_gap=3.0):
    fps = int(F["fps"])
    c = median_filter(fused_confidence(F), size=max(3, fps * 2 | 1), mode="nearest")
    state, start, segs = False, 0, []
    for i, v in enumerate(c):
        if not state and v >= on:
            state, start = True, i
        elif state and v < off:
            state = False; segs.append([start, i])
    if state:
        segs.append([start, len(c)])
    merged = []
    for s, e in segs:
        if merged and (s - merged[-1][1]) / fps <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        if (e - s) / fps >= min_len:
            out.append(dict(start=s / fps, end=e / fps, confidence=float(c[s:e].mean()), kind="unverified"))
    return out, c


def classify_without_judge(cuts, F):
    """Fallback when the Qwen-VL judge is off: a short HUD-less burst inside action is a kill animation."""
    fps = int(F["fps"]); M = F["P_mom"]
    act = M[:, [MOMENT.index(k) for k in ("combat", "stealth", "boss")]].sum(1)
    for c in cuts:
        a, b = int(c["start"] * fps), int(c["end"] * fps)
        around = np.r_[act[max(0, a - 5 * fps):a], act[b:b + 5 * fps]]
        short = c["end"] - c["start"] < 10
        c["kind"] = "kill_animation" if short and around.size and around.mean() > 0.5 else "story_cutscene"
    return cuts

"""Stage 3b: turn per-window judgements into complete EVENTS.

An event is one continuous piece of action (a whole fight, a whole climb, a stealth approach and its
kill) from the moment it starts to the moment it is over. Editing works on whole events, never on
isolated 6-second peaks - that was the cause of clips being cut mid-kill and of unrelated moments being
stitched together.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d, binary_dilation
from .vision import SCENE, MOMENT, ACTION
from .judge import ACTION_ACTS

NAMES = {"combat": "Combat", "stealth": "Stealth", "parkour": "Parkour", "boss": "Boss"}


def frozen_mask(F, thr=0.25, min_s=1.5):
    """Truly static screens (menus, maps, loading, pause): rolling median motion below `thr` for >= 1.5 s.
    Measured: menus/maps/loading have median motion 0.05-0.2, while standing still in a stealth wait is ~0.8
    and holding aim for a headshot ~5 - those must NOT be thrown away (that is where the kills are)."""
    from scipy.ndimage import median_filter
    fps = int(F["fps"]); w = max(1, int(min_s * fps))
    still = median_filter(F["motion"], size=w, mode="nearest") < thr
    return binary_dilation(still, iterations=max(1, fps // 2))


def usable_mask(F, cut_mask):
    """Footage that may appear in an edit at all (hard filters - nothing scores its way past these)."""
    Ps = F["P_scene"]; fps = int(F["fps"])
    bad = sum(Ps[:, SCENE.index(k)] for k in ("menu", "map", "loading", "black"))
    bad = uniform_filter1d(bad, fps)
    talk = uniform_filter1d(F["P_mom"][:, MOMENT.index("dialogue")], fps * 2)
    return (~cut_mask) & (bad < 0.5) & (talk < 0.85) & ~frozen_mask(F) & (F["brightness"] > 10)


def windows_from_siglip(F, usable, win):
    """Fallback labels when the Qwen-VL judge is unavailable (weaker: zero-shot image model only)."""
    fps = int(F["fps"]); M = F["P_mom"]; out = []
    # the image model's "boss" guess is unreliable (it fires on any dark scene), so the fallback calls it combat
    to_act = {"combat": "combat", "stealth": "stealth_kill", "parkour": "parkour", "boss": "combat"}
    for w in range(int(float(F["duration"]) // win)):
        a, b = int(w * win * fps), int((w + 1) * win * fps)
        if usable[a:b].mean() < 0.5:
            out.append(dict(start=w * win, end=(w + 1) * win, activity="skipped", intensity=0, kill=False,
                            enemies=False, caption="")); continue
        m = M[a:b].mean(0); act = sum(m[MOMENT.index(c)] for c in ACTION)
        top = max(ACTION, key=lambda c: m[MOMENT.index(c)])
        motion = np.clip(F["motion"][a:b].mean() / 20, 0, 1)
        inten = int(round(10 * act * (0.6 + 0.4 * motion))) if act > 0.7 else int(round(3 * act))
        out.append(dict(start=w * win, end=(w + 1) * win, activity=to_act[top] if act > 0.7 else "exploring",
                        intensity=inten, kill=False, enemies=act > 0.7, caption="(image-model guess)"))
    return out


# --------------------------------------------------------------------------------------------------------
# ENCOUNTERS - what the montage and the combat/stealth shorts are built from.
# An encounter is one complete fight or stealth chain: from the moment the player engages (lines up the first
# shot, starts stalking the first target) to the moment the last enemy is down. Stealth kills that lead into a
# fight are the same encounter. Lulls of walking INSIDE an encounter are cut out (joined with a soft
# cross-dissolve); a fight is never cut in the middle.
# --------------------------------------------------------------------------------------------------------
ENGAGE_ACTS = {"combat", "stealth_kill", "boss_fight"}
NON_PLAY = {"skipped", "dialogue", "menu_or_map", "cutscene", "loading"}


def is_engaged(w, min_int):
    if w["activity"] in ENGAGE_ACTS:
        return w["intensity"] >= min_int or w["kill"]
    return bool(w.get("enemies")) and w["activity"] not in NON_PLAY and w["intensity"] >= min_int


def impact_time(F, a, b):
    """Loudest sudden audio rise (shot, hit, scream) inside [a, b] - the actual moment of a kill."""
    fps = int(F["fps"]); i = int(a * fps); j = max(i + 1, int(b * fps))
    return (i + int(np.argmax(F["onset"][i:j]))) / fps


def _clip_to_usable(usable, fps, a, b, core_a, core_b):
    """Extend [core_a, core_b] towards [a, b] only through usable footage."""
    A, B = max(0, int(a * fps)), min(len(usable), int(b * fps))
    ca, cb = int(core_a * fps), int(core_b * fps)
    while A < ca and not usable[A:ca].all(): A += 1
    while B > cb and not usable[cb:B].all(): B -= 1
    return A / fps, B / fps


def _usable_segments(usable, fps, segs):
    """The kill guard may widen a segment; never let that widening pull in a menu, cutscene or loading screen."""
    from scipy.ndimage import binary_closing
    usable = binary_closing(usable, structure=np.ones(fps * 2, bool)) | usable   # ignore sub-2 s blips
    out = []
    for a, b in segs:
        A, B = max(0, int(a * fps)), min(len(usable), int(b * fps))
        k = A
        while k < B:
            if not usable[k]:
                k += 1; continue
            j = k
            while j < B and usable[j]:
                j += 1
            if (j - k) / fps >= 1.0:
                out.append([k / fps, j / fps])
            k = j
    return out or [list(s) for s in segs]


HARD_BREAK = {"skipped", "cutscene", "menu_or_map"}


def _breaks_fight(gap, ec):
    """Does the stretch between two engaged windows end the fight? Owner review 2026-11-18 ("full combat are not
    shown"): one 4 s 'loading' flash (white screen-flash power) or one 'dialogue' window (an enemy taunting
    mid-fight) split one battle in two and only half was shown. A short taunt/flash inside a fight no longer ends
    it; a cutscene, menu, or a longer non-play stretch still does."""
    if any(w["activity"] in HARD_BREAK for w in gap):
        return True
    soft = sum(w["end"] - w["start"] for w in gap if w["activity"] in NON_PLAY)
    return soft > float(ec.get("max_inner_talk_s", 8.0))


def build_encounters(windows, usable, F, cfg, kill_anims=()):
    ec = cfg["encounters"]; fps = int(F["fps"])
    min_int, gap_w, dead_w = int(ec["min_intensity"]), int(ec["merge_gap_windows"]), int(ec["cut_dead_windows"])
    for w in windows:                                   # scripted kill animations are part of the action
        for ka in kill_anims:
            if w["start"] < ka["end"] and w["end"] > ka["start"]:
                w.update(kill=True, intensity=max(w["intensity"], 8), enemies=True)
                if w["activity"] not in ACTION_ACTS:
                    w["activity"] = "combat"
    eng = [is_engaged(w, min_int) for w in windows]
    runs, i = [], 0
    while i < len(windows):
        if not eng[i]:
            i += 1; continue
        j = i
        while True:
            nxt = next((k for k in range(j + 1, min(len(windows), j + 2 + gap_w)) if eng[k]), None)
            if nxt is None or _breaks_fight(windows[j + 1:nxt], ec):
                break
            j = nxt
        runs.append((i, j)); i = j + 1
    encs = []
    for i, j in runs:
        ws = windows[i:j + 1]; flags = eng[i:j + 1]
        # split out internal dead stretches (>= dead_w windows of walking)
        groups, cur, dead = [], [], []
        for w, f in zip(ws, flags):
            if f:
                if dead and len(dead) < dead_w:
                    cur += dead
                elif dead:
                    groups.append(cur); cur = []
                dead = []; cur.append(w)
            else:
                dead.append(w)
        groups.append(cur)
        segs = [[g[0]["start"], g[-1]["end"]] for g in groups if g]
        # KILL GUARD (owner: "kill is not fully shown ... make sure you don't do this again"): no cut, dissolve or
        # section end may fall near a kill. Every kill keeps kill_guard_before s before its impact and
        # kill_guard_after s after it; segments are widened (and merged) until that holds.
        kws_ = [w for w, f in zip(ws, flags) if f and w["kill"]]
        gb, ga = float(ec.get("kill_guard_before", 2.0)), float(ec.get("kill_guard_after", 2.5))
        for w in kws_:
            t = w.get("elim_t") or impact_time(F, w["start"], w["end"])
            for lo_, hi_ in ((w["start"], w["end"]), (t - gb, t + ga)):
                segs.append([lo_, hi_])
        segs.sort()
        merged = []
        for a_, b_ in segs:
            if merged and a_ <= merged[-1][1] + 0.5:
                merged[-1][1] = max(merged[-1][1], b_)
            else:
                merged.append([a_, b_])
        segs = merged
        if ec.get("start_at_action"):
            # owner review 2026-11-18 + reference: a clip opens on the first real strike (a kill or an intensity-7+
            # window, stealth kills included) and stops after the last one - never 20 s of sneaking/taunts first
            hot = [w for w, f in zip(ws, flags) if f and (w["kill"] or w["intensity"] >= 7)]
            if hot:
                h0, h1 = hot[0]["start"], hot[-1]["end"]
                segs = [[max(a_, h0), min(b_, h1)] for a_, b_ in segs if b_ > h0 and a_ < h1] or segs
        s0, e0 = _clip_to_usable(usable, fps, segs[0][0] - float(ec["lead_in"]), segs[-1][1] + float(ec["post_roll"]),
                                 segs[0][0], segs[-1][1])
        segs[0][0], segs[-1][1] = s0, e0
        segs = _usable_segments(usable, fps, segs)
        ews =[w for w, f in zip(ws, flags) if f]
        kills = [w for w in ews if w["kill"]]
        stealth = [w for w in kills if w["activity"] == "stealth_kill"]
        spect = [w for w in ews if w["intensity"] >= 9 and w["kill"]]
        weight = {}
        for w in ews:
            c = ACTION_ACTS.get(w["activity"], "combat"); weight[c] = weight.get(c, 0) + max(1, w["intensity"])
        weight.pop("parkour", None)
        cat = max(weight, key=weight.get) if weight else "combat"
        if len(stealth) >= 2 and len(stealth) >= len(kills) / 2:
            cat = "stealth"
        top = sorted((w["intensity"] for w in ews), reverse=True)[:3]
        eng_s = sum(w["end"] - w["start"] for w in ews)
        score = float(np.mean(top)) + 1.0 * min(len(kills), 6) + 1.0 * min(len(spect), 3) + 1.0 * min(len(stealth), 3) \
            + 0.3 * np.log1p(eng_s / 10)
        impacts = [w.get("elim_t") or impact_time(F, w["start"], w["end"]) for w in kills]
        elim_times = [w["elim_t"] for w in kills if w.get("verified")]
        impact_db = [float(F["onset"][min(len(F["onset"]) - 1, int(t * fps))]) for t in impacts]
        encs.append(dict(start=segs[0][0], end=segs[-1][1], segments=[tuple(s) for s in segs], cat=cat,
                         score=round(score, 2), windows=ews, weights=weight, kills=[(w["start"], w["end"]) for w in kills],
                         impacts=impacts, impact_db=impact_db, elim_times=elim_times, n_stealth=len(stealth), n_spect=len(spect),
                         length=round(sum(b - a for a, b in segs), 1), title=NAMES.get(cat, "Combat"),
                         caption=max(ews, key=lambda w: w["intensity"])["caption"], hype=None, hype_why=""))
    return encs


def worth_showing(e, cfg):
    """The user's rule: a single ordinary kill is not worth the viewer's time. Keep an encounter only if it has
    >= min_kills kills, or a stealth kill, or a spectacular moment."""
    ec = cfg["encounters"]
    kc = (cfg.get("_game") or {}).get("kill_counter") or {}
    if cfg.get("_elims") is not None and kc.get("require_for_montage") and not e.get("elim_times"):
        return False                    # BGMI: no confirmed elimination in this fight = not montage material
    return len(e["kills"]) >= int(ec["min_kills"]) or e["n_stealth"] >= 1 or \
        (e["n_spect"] >= 1 and len(e["kills"]) >= 1)


def build_events(windows, usable, F, cfg, kill_anims=()):
    ec = cfg["events"]; fps = int(F["fps"])
    min_int, gap_w = int(ec["min_intensity"]), int(ec["gap_windows"])
    for w in windows:                                   # scripted kill animations are part of the action
        for ka in kill_anims:
            if w["start"] < ka["end"] and w["end"] > ka["start"]:
                w.update(kill=True, intensity=max(w["intensity"], 7))
                if w["activity"] not in ACTION_ACTS:
                    w["activity"] = "combat"
    is_act = [w["activity"] in ACTION_ACTS and w["intensity"] >= min_int for w in windows]
    runs, i = [], 0
    while i < len(windows):
        if not is_act[i]:
            i += 1; continue
        j = i
        while True:
            nxt = next((k for k in range(j + 1, min(len(windows), j + 2 + gap_w)) if is_act[k]), None)
            if nxt is None or any(windows[k]["activity"] == "skipped" for k in range(j + 1, nxt)):
                break
            j = nxt
        runs.append((i, j)); i = j + 1
    events = []
    for i, j in runs:
        ws = windows[i:j + 1]
        start, end = ws[0]["start"] - float(ec["pre_roll"]), ws[-1]["end"] + float(ec["post_roll"])
        a, b = max(0, int(start * fps)), min(len(usable), int(end * fps))     # never run into a cutscene/menu
        core_a, core_b = int(ws[0]["start"] * fps), int(ws[-1]["end"] * fps)
        while a < core_a and not usable[a:core_a].all(): a += 1
        while b > core_b and not usable[core_b:b].all(): b -= 1
        start, end = a / fps, b / fps
        kills = [w for w in ws if w["kill"]]
        if end - start < float(ec["min_len"]) and not kills:
            continue
        weight = {}
        for w in ws:
            if w["activity"] in ACTION_ACTS:
                c = ACTION_ACTS[w["activity"]]; weight[c] = weight.get(c, 0) + w["intensity"]
        cat = max(weight, key=weight.get)
        top = sorted((w["intensity"] for w in ws), reverse=True)[:3]
        score = float(np.mean(top)) + 1.0 * min(3, len(kills)) + 0.5 * np.log1p((end - start) / 10)
        events.append(dict(start=start, end=end, cat=cat, score=round(score, 2), windows=ws, weights=weight,
                           kills=[(w["start"], w["end"]) for w in kills], title=NAMES[cat],
                           caption=max(ws, key=lambda w: w["intensity"])["caption"]))
    return events

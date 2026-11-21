"""KILL-CENTRED EDITING (games with an on-screen kill counter, e.g. BGMI / PUBG Mobile).

Why (owner's review 2026-11-11 of the encounter-based BGMI edit): a battle-royale "fight" is mostly scanning,
waiting, looting and healing around a few seconds of shooting, so whole-fight sections were full of
"unnecessary clips"; the player's own knock-down was shown; the shorts ran on after nothing happened.

Rules:
  * Every edit is built from CONFIRMED eliminations only (HUD counter, judge.read_kill_counter).
  * Around each kill: the shooting that leads to it and the kill landing - nothing else.
  * Every second of every planned clip is checked by the AI (judge.review_seconds): only shooting / aiming /
    kill seconds are kept; 2 s or more of anything else is cut out (soft dissolve); a clip ENDS before the
    player's own character gets knocked or dies, and that stretch is never shown.
  * Montage: the best clip's final kill as a 5 s hook (taken out of its place later - never shown twice), then
    every kill clip in match order; dissolves inside a match, dip to black between matches.
  * Shorts: exactly 3 kills, then end (1.2 s after the 3rd kill lands, slow motion on it).
"""
import numpy as np
from .planner import _piece, _slowmo_split, _out_len


def match_ids(F, min_gap=6.0):
    """Clips of different matches are joined with black gaps; number the matches."""
    fps = int(F["fps"]); dark = F["brightness"] < 20      # video black is 16 (limited range), not 0
    starts, i = [], 0
    while i < len(dark):
        if dark[i]:
            j = i
            while j < len(dark) and dark[j]:
                j += 1
            if (j - i) / fps >= min_gap:
                starts.append(j / fps)
            i = j
        else:
            i += 1
    return lambda t: sum(1 for s in starts if t >= s) + 1


def candidate_spans(elims, usable, F, cfg):
    """[kill - back, kill + after] around every confirmed elimination, merged when they touch, never crossing
    unusable footage (menus, spectating, the black gap between matches)."""
    kc = cfg["kill_edit"]; fps = int(F["fps"])
    back, after = float(kc.get("search_back", 8.0)), float(kc.get("search_after", 2.5))
    spans = []
    for t in sorted(elims):
        a, b = t - back, t + after
        A, T = max(0, int(a * fps)), int(t * fps)
        bad = np.where(~usable[A:T])[0]
        if len(bad):
            a = (A + bad[-1] + 1) / fps
        B = min(len(usable), int(b * fps))
        bad = np.where(~usable[T:B])[0]
        if len(bad):
            b = (T + bad[0]) / fps
        if b - a < 1.0:
            continue
        if spans and a <= spans[-1][1] + 1.0:
            spans[-1][1] = max(spans[-1][1], b); spans[-1][2].append(t)
        else:
            spans.append([a, b, [t]])
    return spans


KEEP = ("shooting", "aiming", "kill")


def tighten(span, labels, cfg):
    """Turn one candidate span into the pieces worth showing, using the per-second AI labels."""
    kc = cfg["kill_edit"]
    a, b, kills = span
    secs = list(range(int(np.floor(a)), int(np.ceil(b))))
    lab = {s: labels.get(s, "dead_time") for s in secs}
    down = [s for s in secs if lab[s] == "player_down"]
    gb, ga = float(kc.get("kill_before", 1.5)), float(kc.get("kill_after", 1.5))
    D = set(down)
    keep = {}
    hot = {s for s in secs if lab[s] in ("shooting", "kill")}
    for s in secs:
        # "aiming" alone was the AI's word for walking around with a gun up (87 of 162 s on 2026-11-11): it only
        # counts in the 2 s right before the player actually fires
        aim_ok = lab[s] == "aiming" and ((s + 1) in hot or (s + 2) in hot)
        keep[s] = s in hot or aim_ok or any(t - gb <= s + 0.5 <= t + ga for t in kills)
        if s in D or (s + 1) in D:          # never the player's own knock / death, nor the hits just before it
            keep[s] = False
    # once the player is down, nothing after it is shown - unless he is back up and gets another kill later
    if down:
        first, last = down[0], down[-1]
        for s in secs:
            if s >= first - 1:
                revived_kill = s > last and any(t - 6 <= s + 0.5 <= t + ga for t in kills if t > last + 1)
                if not revived_kill:
                    keep[s] = False
    # fill 1-second holes (choppy otherwise); runs of >= 2 dead seconds are cut
    for i in range(1, len(secs) - 1):
        if not keep[secs[i]] and keep[secs[i - 1]] and keep[secs[i + 1]] and lab[secs[i]] != "player_down":
            keep[secs[i]] = True
    runs, cur = [], None
    for s in secs:
        if keep[s]:
            cur = [s, s + 1] if cur is None else [cur[0], s + 1]
        elif cur:
            runs.append(cur); cur = None
    if cur:
        runs.append(cur)
    # a run must contain a kill, or lead into one (the build-up) within 3 s; the build-up is at most `max_buildup` s
    mb = float(kc.get("max_buildup", 6.0))
    out = []
    for i, (x, y) in enumerate(runs):
        has = [t for t in kills if x - 0.5 <= t <= y + 0.5]
        nxt = runs[i + 1] if i + 1 < len(runs) else None
        lead = nxt and nxt[0] - y <= 3 and any(nxt[0] - 0.5 <= t <= nxt[1] + 0.5 for t in kills)
        if has:
            x = max(x, min(has) - mb)
            out.append([max(a, float(x)), min(b, float(y)), has])
        elif lead:
            nk = min(t for t in kills if nxt[0] - 0.5 <= t <= nxt[1] + 0.5)
            x = max(x, nk - mb)
            if y - x >= 1.0:
                out.append([max(a, float(x)), min(b, float(y)), []])
    # merge pieces into clips: pieces whose gaps are short belong to the same clip (dissolve between them)
    pieces = [p for p in out if p[1] - p[0] >= float(kc.get("min_piece", 1.5)) or p[2]]
    return pieces


def build_clips(spans, labels, cfg, match_of):
    clips = []
    for sp in spans:
        ps = tighten(sp, labels, cfg)
        kills = sorted({t for p in ps for t in p[2]})
        if not kills:
            continue
        clips.append(dict(pieces=[(p[0], p[1]) for p in ps], kills=kills, start=ps[0][0], end=ps[-1][1],
                          match=match_of(kills[0]), wow=None))
    return clips


def _pieces_for(clip, a=None, b=None, join_first="xfade"):
    out = []
    for x, y in clip["pieces"]:
        x2, y2 = max(x, a) if a is not None else x, min(y, b) if b is not None else y
        if y2 - x2 >= 1.0:
            out.append(_piece(x2, y2, join="xfade"))
    if out:
        out[0]["join"] = join_first
    return out


def plan_long(clips, cfg):
    lc = cfg["style"]["long"]; kc = cfg["kill_edit"]; dip = float(lc["dip"])
    if not clips:
        return []
    groups = []
    best = max(clips, key=lambda c: ((c["wow"] or 5) + 2 * len(c["kills"]), len(c["kills"])))
    hook = None
    if lc.get("cold_open"):
        t = best["kills"][-1]
        s = max(best["start"], t - float(kc.get("hook_before", 3.0))); e = min(best["end"], t + float(kc.get("hook_after", 1.5)))
        ps = _pieces_for(best, s, e, join_first="cut")
        for p in ps[1:]:
            p["join"] = "cut" if p["start"] - ps[0]["end"] < 0.05 else "xfade"
        ps = _slowmo_split(ps, t, cfg)
        groups.append(dict(title="Cold open", pieces=ps, fade_in=0.0, fade_out=dip, cold=True))
        hook = (best, s, e)
    by_match = {}
    for c in clips:
        by_match.setdefault(c["match"], []).append(c)
    for m in sorted(by_match):
        pieces = []
        for c in sorted(by_match[m], key=lambda c: c["start"]):
            ps = _pieces_for(c)
            if hook and c is hook[0] and lc.get("cold_open_no_repeat", True):
                h1, h2 = hook[1], hook[2]          # the hook moment is never shown a second time
                cut = []
                for p in ps:
                    if p["end"] <= h1 or p["start"] >= h2:
                        cut.append(p); continue
                    for x, y in ((p["start"], h1), (h2, p["end"])):
                        if y - x >= 1.5:
                            cut.append(_piece(x, y))
                ps = cut
            pieces += ps
        if not pieces:
            continue
        pieces[0]["join"] = "cut"
        groups.append(dict(title=f"Match {m}", pieces=pieces, fade_in=dip if groups else 0.0, fade_out=dip))
    return groups


def plan_shorts(clips, cfg):
    """Exactly 3 kills, then end. Candidates = every run of 3 consecutive kills in one match; the best
    non-overlapping ones (AI rating of their clips + how tightly the kills follow each other) up to max_total."""
    sc = cfg["style"]["short"]; kc = cfg["kill_edit"]; xd = float(cfg["style"]["long"]["xfade"])
    n_k = int(kc.get("kills_per_short", 3)); tail = float(kc.get("short_after", 1.2))
    kills = []
    for c in clips:
        for t in c["kills"]:
            kills.append((t, c))
    kills.sort(key=lambda k: k[0])
    # owner: "combine them and show player 3 kills and then end" - the 3 kills may come from different fights.
    # Chronological groups of 3; when the count does not divide by 3, the weakest kills are left out.
    while len(kills) % n_k and len(kills) > n_k:
        kills.remove(min(kills, key=lambda k: (k[1]["wow"] or 5, -k[0])))
    groups = [kills[i:i + n_k] for i in range(0, len(kills) - n_k + 1, n_k)]
    groups = sorted(groups, key=lambda g: -sum((c["wow"] or 5) for _, c in g))[:int(sc.get("max_total", 6))]
    groups.sort(key=lambda g: g[0][0])
    bu = float(kc.get("short_buildup", 4.0))
    res = []
    for i, grp in enumerate(groups, 1):
        pieces = []
        for j, (t, c) in enumerate(grp):
            # each kill: its own build-up (max short_buildup s) to the kill; the last one ends `tail` s after it
            end = t + (tail if j == len(grp) - 1 else float(kc.get("kill_after", 1.5)))
            prev_t = grp[j - 1][0] if j else -1e9
            ps = _pieces_for(c, a=max(t - bu, prev_t + float(kc.get("kill_after", 1.5))), b=end)
            ps = [p for p in ps if p["end"] > t - bu]
            if ps:
                ps[0]["join"] = "xfade"
                if j and pieces and abs(ps[0]["start"] - pieces[-1]["end"]) <= 0.05:
                    ps[0]["join"] = "cut"           # same continuous footage: seamless
                elif j:
                    ps[0].update(trans=kc.get("short_transition", "smoothleft"), trans_d=0.2)   # whip between fights
            pieces += ps
        if not pieces:
            continue
        pieces[0]["join"] = "cut"
        if cfg["style"]["slowmo"].get("enabled", True):
            pieces = _slowmo_split(pieces, grp[-1][0], cfg)
        res.append((f"Triple_Kill_Short_{i:02d}", dict(theme="Triple Kill", title=f"3 Kills #{i}", pieces=pieces,
                                                       fade_in=0.0, fade_out=float(sc["fade_out"]),
                                                       length=_out_len(pieces, xd))))
    return res

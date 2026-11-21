"""FIGHT-CENTRED EDITING for shooters (BGMI / PUBG Mobile). Replaces killedit.py for `edit_mode: fights`.

Why (owner review 2026-11-11 of the kill-only edit): "shorts are still shitty", "full combat is not shown".
The kill-only edit stitched 3-second kill fragments from different fights into 11-second shorts, and the
"after the kill" seconds were mostly looting. Root cause: the AI judge sees 4 still frames per 4 s and never
hears the audio, so it cannot tell shooting from looting.

New ground truth, both measured, not guessed:
  * the player's OWN gunfire, found in the audio track: own shots are loud, sharp transients close to the mic
    (automatic fire = regular bursts 50-220 ms apart; snipers / shotguns = very loud single shots). Distant
    gunfire from other squads is much quieter and is ignored (checked frame-by-frame on BGMI_2026-11-03).
  * confirmed eliminations from the HUD kill counter (judge.read_kill_counter).

A FIGHT = a cluster of own gunfire (gaps < cluster_gap s) that contains at least one confirmed elimination.
It starts `lead_in` s before the first shot (the player lining up the shot) and ends `after_kill` s after the
last kill. It is never cut in the middle; only quiet stretches INSIDE it (no own shot, no kill for >= lull_cut s:
looting, healing, waiting) are removed with a soft dissolve. It ends before the player's own knock-down.

Montage: opens with the BEST fight in full (engagement in the first seconds, final kill in slow motion), then
every other worthy fight in timeline order, dip to black between fights. Nothing is ever shown twice.
Shorts: ONE complete fight per short (up to max_len s, ending on the last kill in slow motion).
"""
import json, os, subprocess
import numpy as np
from .planner import _piece, _slowmo_split, _out_len

GUN_VERSION = "g1"


# ------------------------------------------------------------------------------------------------ gunfire
def gunfire(video, cfg, workdir):
    """Times (s) of the player's own gunshots, from the audio track. Cached."""
    gc = cfg["fight_edit"]["gunfire"]
    key = f"{GUN_VERSION}|{gc}"
    cache = os.path.join(workdir, "gunfire.json")
    if os.path.exists(cache):
        c = json.load(open(cache, encoding="utf-8"))
        if c.get("key") == key:
            return np.array(c["shots"])
    from scipy.signal import butter, sosfiltfilt
    from scipy.ndimage import median_filter, maximum_filter
    SR = 16000
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    if len(x) < SR:
        return np.array([])
    hop = SR // 100                                                   # 10 ms frames

    def env(lo, hi):
        h = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), x)
        n = len(h) // hop
        return 20 * np.log10(np.sqrt((h[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-6)

    hi, lo = env(1500, 7000), env(80, 600)
    bg = median_filter(hi, size=150)
    rise = hi - np.concatenate([[hi[0]] * 3, hi[:-3]])
    idx = np.where((hi - bg > 10) & (rise > 8) & (hi == maximum_filter(hi, size=5)))[0]
    t = idx / 100.0
    lvl = hi[idx] - np.median(hi); low = lo[idx] - np.median(lo)
    burst = np.zeros(len(t), bool)                                    # automatic fire: regular rapid onsets
    i = 0
    while i < len(t) - 2:
        j = i
        while j + 1 < len(t) and 0.05 <= t[j + 1] - t[j] <= 0.22:
            j += 1
        if j - i >= 2 and np.diff(t[i:j + 1]).std() < 0.05:
            burst[i:j + 1] = True
        i = j + 1 if j > i else i + 1
    own = lvl > float(gc.get("own_level_db", 32))                     # own gun = loud (close to the mic)
    single = (lvl > np.percentile(lvl, 97)) & (low > np.percentile(low, 90))   # sniper / shotgun
    shots = t[(burst & own) | single]
    json.dump(dict(key=key, shots=shots.tolist()), open(cache, "w", encoding="utf-8"))
    print(f"  gunfire: {len(shots)} own shots found in the audio")
    return shots


# ------------------------------------------------------------------------------------------------ fights
def find_fights(shots, elims, usable, F, cfg, match_of):
    fc = cfg["fight_edit"]; fps = int(F["fps"])
    gap, merge = float(fc["cluster_gap"]), float(fc["merge_gap"])
    cl = []
    for s in shots:
        if not usable[min(len(usable) - 1, int(s * fps))]:
            continue
        if cl and s - cl[-1][1] <= gap:
            cl[-1][1] = s; cl[-1][2].append(s)
        else:
            cl.append([s, s, [s]])
    # clusters a short pause apart (reload, reposition) are the same fight
    m = []
    for c in cl:
        if m and c[0] - m[-1][1] <= merge:
            m[-1][1] = c[1]; m[-1][2] += c[2]
        else:
            m.append(c)
    fights = []
    for a, b, sh in m:
        ks = [t for t in elims if a - 3 <= t <= b + float(fc["kill_late"])]
        if not ks:
            continue
        # the HUD counter is read every 2 s, so its time is late (the player is often looting by then): the kill
        # landed on the player's last own shot before the reading (review 2026-11-11: short ended on looting)
        sh_arr = np.array(sh)
        snapped = []
        for t in ks:
            before = sh_arr[(sh_arr <= t + 0.3) & (sh_arr >= t - float(fc["kill_late"]) - 0.5)]
            snapped.append(float(before[-1]) + 0.15 if len(before) else t)
        ks = sorted(snapped)
        start = a - float(fc["lead_in"])
        end = max(ks[-1] + float(fc["after_kill"]), b + 0.8)
        A, B = int(max(0, start) * fps), int(min(len(usable) / fps, end) * fps)
        bad = np.where(~usable[A:int(a * fps)])[0]
        if len(bad):
            start = (A + bad[-1] + 1) / fps
        bad = np.where(~usable[int(ks[-1] * fps):B])[0]
        if len(bad):
            end = (int(ks[-1] * fps) + bad[0]) / fps
        fights.append(dict(start=max(0.0, start), end=end, shots=sh, kills=ks, match=match_of(ks[0]), wow=None))
    return fights


def fights_from_kills(kills, shots, usable, F, cfg, match_of):
    """Games whose own kills come from a reliable kill feed (CS2): the KILLS define the fights. The audio gunfire
    detector was tuned on BGMI phone audio; on a stream mix (game + voice) it found only half the fights
    (2026-11-12: 42 feed kills, 20 inside gunfire clusters). Kills <= kill_cluster s apart are one fight; the span
    reaches max_build s before the first kill and is then trimmed by tighten() using the per-second AI review."""
    fc = cfg["fight_edit"]; fps = int(F["fps"])
    gap = float(fc.get("kill_cluster", 15.0)); build = float(fc.get("max_build", 10.0))
    sh = np.array(shots) if len(shots) else np.array([])
    groups = []
    for k in sorted(kills):
        if groups and k - groups[-1][-1] <= gap:
            groups[-1].append(k)
        else:
            groups.append([k])
    out = []
    for ks in groups:
        snapped = []                 # the feed is read every second: the kill landed on the last shot just before
        for t in ks:
            before = sh[(sh <= t + 0.2) & (sh >= t - 1.5)] if len(sh) else []
            snapped.append(float(before[-1]) + 0.15 if len(before) else t - 0.5)
        ks = sorted(snapped)
        start, end = ks[0] - build, ks[-1] + float(fc["after_kill"])
        A, B = int(max(0, start) * fps), int(min(len(usable) / fps, end) * fps)
        bad = np.where(~usable[A:int(ks[0] * fps)])[0]
        if len(bad):
            start = (A + bad[-1] + 1) / fps
        bad = np.where(~usable[int(ks[-1] * fps):B])[0]
        if len(bad):
            end = (int(ks[-1] * fps) + bad[0]) / fps
        s_in = [float(s) for s in sh if start <= s <= end] if len(sh) else []
        out.append(dict(start=max(0.0, start), end=end, shots=s_in, kills=ks, match=match_of(ks[0]), wow=None))
    return out


def add_review_shots(f, labels):
    """Seconds the AI saw the player SHOOTING count as shots (the audio misses some on a stream mix)."""
    extra = [s + 0.5 for s in range(int(np.floor(f["start"])), int(np.ceil(f["end"]))) if labels.get(s) == "shooting"]
    return dict(f, shots=sorted(set(f["shots"]) | set(extra)))


def tighten(f, labels, cfg, loot=None):
    """Pieces of one fight: cut before the player's own knock-down; remove quiet stretches inside it."""
    fc = cfg["fight_edit"]
    a, b, shots, kills = f["start"], f["end"], np.array(f["shots"]), f["kills"]
    # the player's own knock-down: two 'player_down' seconds in a row (a single one is often a misread)
    secs = range(int(np.floor(a)), int(np.ceil(b)))
    down = [s for s in secs if labels.get(s) == "player_down" and labels.get(s + 1) == "player_down"]
    if down:
        d = down[0] - 1.0
        if any(k <= d for k in kills):
            b = min(b, d); kills = [k for k in kills if k <= b]
    # start and end on ACTION: the lead-in is kept only where the AI saw aiming / shooting (no parachute, no loot
    # menu); after the last kill, the first second that is not action ends the fight (review 2026-11-11)
    act = ("shooting", "aiming", "kill")
    # anchor the START on the first kill (review 2026-11-12: stray pickup / footstep sounds 10-20 s earlier pulled
    # the start back into looting and walking): walk back from the first kill through shots that are at most
    # `chain_gap` s apart; at most `max_build` s of shooting before the first kill
    if kills and len(shots):
        k0, cg = kills[0], float(fc.get("chain_gap", 4.0))
        pre = [float(s) for s in shots if s <= k0 + 0.3]
        if pre:
            c = pre[-1]
            for s in reversed(pre[:-1]):
                if c - s > cg or k0 - s > float(fc.get("max_build", 10.0)):
                    break
                c = s
            shots = shots[shots >= c - 0.01]
            a = max(a, c - float(fc["lead_in"]))
    first = float(shots[0]) if len(shots) else a
    s0 = int(np.floor(first)) - 1
    while s0 >= a and labels.get(s0) in act and first - s0 <= float(fc["lead_in"]):
        s0 -= 1
    a = max(a, min(first - 0.6, s0 + 1.0))
    if kills:
        k = kills[-1]; e = int(np.floor(k)) + 1
        while e < b and labels.get(e) in act and e - k < float(fc["after_kill"]):
            e += 1
        b = min(b, max(k + 1.0, float(e)))
    # quiet = no own shot within 2 s and no kill within 2.5 s
    step = 0.25
    ts = np.arange(a, b, step)
    hot = np.zeros(len(ts), bool)
    for s in shots:
        hot |= np.abs(ts - s) <= 2.0
    for k in kills:
        hot |= np.abs(ts - k) <= 2.5
    hot[: int(max(0.0, first - a) / step) + 1] = True               # the aim-in before the first shot stays
    for s in range(int(np.floor(a)), int(np.ceil(b))):              # AI-confirmed looting / walking is never "hot"
        if labels.get(s) == "dead_time":
            near = (np.abs(shots - (s + 0.5)) <= 0.8).any() or any(abs(k - (s + 0.5)) <= 1.2 for k in kills)
            if not near:
                hot[(ts >= s) & (ts < s + 1)] = False
    looting = np.zeros(len(ts), bool)                               # 'Picked up' / loot list on screen
    loot = set(loot or ())
    for s in {x + d for x in loot for d in (-2, -1, 0, 1, 2)}:      # walking to / between pickups is looting too
        m = (ts >= s) & (ts < s + 1)
        before = any(s + d in loot for d in (1, 2))                 # the pickup click precedes the text
        if m.any() and not any(abs(k - (s + 0.5)) <= 1.5 for k in kills) \
                and not (s not in loot and not before and len(shots) and (np.abs(shots - (s + 0.5)) <= 0.8).any()):
            looting |= m
    hot &= ~looting
    keep, lull, i = [], float(fc["lull_cut"]), 0
    runs = []
    while i < len(ts):
        j = i
        while j < len(ts) and hot[j] == hot[i]:
            j += 1
        runs.append((hot[i], ts[i], ts[j - 1] + step, bool(looting[i:j].any())))
        i = j
    if runs and not runs[0][0] and runs[0][3]:                      # never OPEN on looting
        a = runs[0][2]; runs = runs[1:]
    pieces = []
    for h, x, y, lt in runs:
        if lt and not h and y - x >= 1.5:                           # looting inside a fight: cut it out completely
            if pieces:
                pieces[-1][1] = x + 0.3
            pieces.append([y - 0.3, y])
            continue
        if h or y - x < lull:
            if pieces and abs(pieces[-1][1] - x) < 1e-6:
                pieces[-1][1] = y
            else:
                pieces.append([x, y])
        else:                                                       # long quiet stretch: keep 0.8 s each side
            if pieces:
                pieces[-1][1] = x + 0.8
            pieces.append([y - 0.8, y])
    pieces = [[x, min(y, b)] for x, y in pieces if min(y, b) - x >= 1.0]
    f = dict(f, end=b, kills=kills, pieces=[tuple(p) for p in pieces])
    f["length"] = sum(y - x for x, y in f["pieces"])
    return f


def score(f):
    return 3 * len(f["kills"]) + (f["wow"] or 5) + min(len(f["shots"]), 60) / 15


def worthy(f, cfg):
    fc = cfg["fight_edit"]
    return len(f["kills"]) >= int(fc["min_kills"]) or (f["wow"] or 0) >= int(fc["single_kill_wow"])


def _pieces(f, a=None, b=None):
    out = []
    for x, y in f["pieces"]:
        x2 = max(x, a) if a is not None else x
        y2 = min(y, b) if b is not None else y
        if y2 - x2 >= 1.0:
            out.append(_piece(x2, y2, join="xfade"))
    if out:
        out[0]["join"] = "cut"
    return out


# ------------------------------------------------------------------------------------------------ plans
def plan_long(fights, cfg):
    dip = float(cfg["style"]["long"]["dip"])
    pool = [f for f in fights if worthy(f, cfg) and f["pieces"]]
    if not pool:
        return []
    best = max(pool, key=score)
    groups = []
    ps = _pieces(best)
    if cfg["style"]["slowmo"].get("enabled", True) and best["kills"]:
        ps = _slowmo_split(ps, best["kills"][-1], cfg)
    groups.append(dict(title=f"Opening - {len(best['kills'])} kills", pieces=ps, fade_in=0.0, fade_out=dip, cold=True,
                       kill_t=best["kills"][-1]))
    n = 0
    for f in sorted(pool, key=lambda f: f["start"]):
        if f is best:
            continue
        n += 1
        groups.append(dict(title=f"Match {f['match']} - fight {n} ({len(f['kills'])} kill{'s' * (len(f['kills']) > 1)})",
                           pieces=_pieces(f), fade_in=dip, fade_out=dip, kill_t=f["kills"][-1]))
    return groups


def plan_shorts(fights, cfg):
    fc = cfg["fight_edit"]; sc = cfg["style"]["short"]; xd = float(cfg["style"]["long"]["xfade"])
    mx = float(fc["short_max"]); target = float(fc.get("short_target", 24))
    # a short fight (e.g. 3 kills in 10 s) is CHAINED with the next good fight of the same match (owner: shorts
    # must be longer) - whole fights only, soft dissolve between them, never padded with non-fights
    good = sorted([f for f in fights if worthy(f, cfg) and f["pieces"]], key=lambda f: f["start"])
    chains, used = [], set()
    for i, f in enumerate(good):
        if i in used:
            continue
        ch = [f]; used.add(i); L = f["length"]
        for j in range(i + 1, len(good)):
            g = good[j]
            if L >= target or g["match"] != f["match"] or g["start"] - ch[-1]["end"] > float(fc.get("chain_within", 150)) \
                    or L + g["length"] > mx:
                break
            ch.append(g); used.add(j); L += g["length"]
        if L >= float(fc["short_min"]):
            chains.append(ch)
    chains = sorted(chains, key=lambda c: -sum(score(f) for f in c))[: int(sc.get("max_total", 6))]
    res = []
    for i, ch in enumerate(chains, 1):
        ps = []
        for f in ch:
            a = None
            if f["length"] > mx:            # too long: keep the last `mx` s of action, which end on the final kill
                acc, a = 0.0, f["pieces"][0][0]
                for x, y in reversed(f["pieces"]):
                    if acc + (y - x) >= mx:
                        a = y - (mx - acc); break
                    acc += y - x
            q = _pieces(f, a=a)
            if q and ps:
                q[0]["join"] = "xfade"
            ps += q
        if not ps:
            continue
        last = ch[-1]
        if cfg["style"]["slowmo"].get("enabled", True):
            ps = _slowmo_split(ps, last["kills"][-1], cfg)
        k = sum(len(f["kills"]) for f in ch)
        best = max(ch, key=lambda f: len(f["kills"]))
        res.append((f"Fight_Short_{i:02d}", dict(theme="Fight", title=f"{k} kill{'s' * (k > 1)} fight", pieces=ps,
                                                 fade_in=0.0, fade_out=float(sc["fade_out"]), length=_out_len(ps, xd),
                                                 kills=k, wow=max(f["wow"] or 0 for f in ch),
                                                 kill_times=sorted(t for f in ch for t in f["kills"]),
                                                 src=(ch[0]["start"], last["end"]),
                                                 kill_span=(last["kills"][-1] - ch[0]["kills"][0]) if len(ch) == 1
                                                 else 999, best_burst=(len(best["kills"]),
                                                                       best["kills"][-1] - best["kills"][0]))))
    return res

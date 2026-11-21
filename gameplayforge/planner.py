"""Stage 4: edit plans.

Every plan is a list of GROUPS; a group is rendered as one continuous piece of video:
    {"title", "pieces": [{"start", "end", "speed", "zoom", "join"}], "fade_in", "fade_out"}
  join  = how a piece joins the previous one inside the group: "xfade" (soft 0.3 s dissolve, used where dead
          walking time was cut out of a fight) or "cut" (used for the slow-motion ramp on a kill).
  fade  = dip to/from black at the group edges (how the references move between sections, ~0.4 s).

LONG MONTAGE (rules from the references + the channel owner's review - see EDITING_RULES.md)
  * Cold open: the single best kill of the video (in slow motion) as a 5-6 s hook, then the montage.
  * Only encounters: whole fights and stealth chains, from the moment the player engages to the last kill.
    No stand-alone traversal or parkour (a few seconds of approach right before a fight is allowed).
  * An encounter with a single ordinary kill is dropped; so is one the judge rates as not exciting.
  * Timeline order, sections joined with a dip to black, never a hard cut.

SHORTS
  * One encounter per short (parkour shorts: one climb/run). Starts ON the action (1.5 s lead-in at most),
    lulls cut out, ends on the final kill, which is played in slow motion with a slight punch-in.
"""
import numpy as np
from .highlights import NAMES, worth_showing


def _piece(a, b, speed=1.0, zoom=1.0, join="xfade"):
    return dict(start=float(a), end=float(b), speed=speed, zoom=zoom, join=join)


def _slowmo_split(pieces, t, cfg, join_after="cut"):
    """Split the piece containing source time t into normal / slow-motion punch-in / normal."""
    sm = cfg["style"]["slowmo"]
    a0, a1 = t - float(sm["before"]), t + float(sm["after"])
    out = []
    for p in pieces:
        if p["start"] <= t <= p["end"] and p["speed"] == 1.0:
            b0, b1 = max(a0, p["start"]), min(a1, p["end"])
            parts = [(p["start"], b0, 1.0, 1.0, p["join"]), (b0, b1, float(sm["speed"]), float(sm["zoom"]), "cut"),
                     (b1, p["end"], 1.0, 1.0, join_after)]
            parts = [q for q in parts if q[1] - q[0] >= 0.1]
            if parts and parts[0][4] == "cut" and p["join"] != "cut":
                parts[0] = parts[0][:4] + (p["join"],)       # first part keeps how the piece joined before
            out += [_piece(*q) for q in parts]
            t = float("inf")                                  # only once
        else:
            out.append(p)
    return out


def _cut_out(segs, a, b, min_keep=3.0):
    out = []
    for s, e in segs:
        for x, y in ((s, min(e, a)), (max(s, b), e)):
            if y - x >= min_keep:
                out.append([x, y])
    return out


def _punch_pieces(pieces, e, cfg):
    """Kills that happen far away / small on screen get a smooth punch-in towards them (zoom eases in and out, so
    the joins are invisible) - the owner wants every kill clearly SEEN. e["punch"] comes from judge.locate_kills."""
    pc = cfg["style"].get("punch_in", {})
    if not pc.get("enabled", True):
        return pieces
    for pu in e.get("punch", []):
        a0, a1 = pu["t"] - float(pc.get("before", 1.6)), pu["t"] + float(pc.get("after", 1.8))
        out = []
        for p in pieces:
            if p["speed"] != 1.0 or p.get("zoom", 1.0) != 1.0 or p["end"] <= a0 or p["start"] >= a1:
                out.append(p); continue
            b0, b1 = max(a0, p["start"]), min(a1, p["end"])
            if b1 - b0 < 1.2:
                out.append(p); continue
            if b0 - p["start"] < 0.6:              # never leave a sliver of a clip: zoom covers it instead
                b0 = p["start"]
            if p["end"] - b1 < 0.6:
                b1 = p["end"]
            if b0 - p["start"] >= 0.1:
                out.append(_piece(p["start"], b0, join=p["join"]))
            q = _piece(b0, b1, zoom=pu["z"], join="cut" if b0 - p["start"] >= 0.1 else p["join"])
            q.update(cx=pu["cx"], cy=pu["cy"], ramp=True)
            out.append(q)
            if p["end"] - b1 >= 0.1:
                out.append(_piece(b1, p["end"], join="cut"))
        pieces = out
    return pieces


def _out_len(pieces, xd):
    L = 0.0
    for k, p in enumerate(pieces):
        L += (p["end"] - p["start"]) / p["speed"] - (float(p.get("trans_d", xd)) if k and p["join"] == "xfade" else 0)
    return L


# ------------------------------------------------------------------------------------------------- long
def plan_long(encs, cfg):
    lc = cfg["style"]["long"]; target = float(lc["target_seconds"]); dip = float(lc["dip"])
    pool = [e for e in encs if worth_showing(e, cfg) and (e["hype"] is None or e["hype"] >= int(lc["min_hype"]))]
    if pool:
        best = max(e["score"] for e in pool)
        pool = [e for e in pool if e["score"] >= float(lc["min_rel_score"]) * best]
    chosen, total = [], 0.0
    for e in sorted(pool, key=lambda e: -(e["score"] + (e["hype"] or 5))):
        if total + e["length"] <= target * 1.1 or not chosen:
            chosen.append(e); total += e["length"]
    chosen.sort(key=lambda e: e["start"])
    if lc.get("open_with_best") and chosen:
        # reference (Bloodlines 2 "brutal combat and air combos"): no teaser - the video opens straight into the
        # best fight, shown whole; the rest follow in timeline order; nothing is shown twice
        top = max(chosen, key=lambda e: (e["score"] + (e["hype"] or 5), len(e["kills"])))
        chosen = [top] + [e for e in chosen if e is not top]
    groups, cold = [], None
    if lc.get("cold_open") and chosen:
        cands = []
        for e in chosen:
            kws = [w for w in e["windows"] if w["kill"]]
            for w, t in zip(kws, e["impacts"]):
                cands.append((w["intensity"], e["hype"] or 0, e["score"], t, e))
        if cands:
            *_, t, e = max(cands, key=lambda c: c[:3])
            s = max(e["start"], t - float(lc["cold_open_before"])); en = min(e["end"], t + float(lc["cold_open_after"]))
            pieces = _slowmo_split([_piece(s, en, join="cut")], t, cfg)
            pu = next((p for p in e.get("punch", []) if abs(p["t"] - t) < 0.5), None)
            for p in pieces:                      # the slow-motion punch-in aims at the kill, not the centre
                if pu and p["speed"] != 1.0:
                    p.update(zoom=max(p["zoom"], pu["z"]), cx=pu["cx"], cy=pu["cy"])
            groups.append(dict(title="Cold open", pieces=pieces, fade_in=0.0, fade_out=dip, cold=True))
            cold = (e, s, en)
    for k, e in enumerate(chosen):
        segs = [list(x) for x in e["segments"]]
        if cold and cold[0] is e and lc.get("cold_open_no_repeat", True):
            # owner: the hook moment must not appear a second time -> cut it out of its fight (soft dissolve)
            segs = _cut_out(segs, cold[1], cold[2])
            if not segs:
                continue
        pieces = _punch_pieces([_piece(a, b) for a, b in segs], e, cfg)
        fi = dip if groups else 0.0          # the very first frame of the video is never black
        groups.append(dict(title=e["title"], pieces=pieces, fade_in=fi, fade_out=dip, cat=e["cat"], enc=e))
    return groups


# ------------------------------------------------------------------------------------------------- shorts
def _trim(segs, u0, u1):
    """Kept-time span [u0, u1] -> source segments."""
    out, u = [], 0.0
    for a, b in segs:
        d = b - a; s, e = max(u0, u), min(u1, u + d)
        if e - s > 0.05:
            out.append([a + (s - u), a + (e - u)])
        u += d
    return out


def _value(e, segs, impacts):
    v = 0.0
    for a, b in segs:
        for w in e["windows"]:
            v += w["intensity"] * max(0.0, min(b, w["end"]) - max(a, w["start"]))
        for t in impacts:
            if a <= t <= b:
                v += 15
    if impacts and segs:
        # the ending matters most: the short should finish on the hardest-hitting kill (loudest impact)
        a0, last_end = segs[0][0], segs[-1][1]
        inside = [(t, db) for t, db in zip(impacts, e.get("impact_db", [8] * len(impacts))) if a0 <= t <= last_end]
        if inside:
            t, db = max(inside)
            v += 10 + 4 * db
    return v


def short_from_encounter(e, cfg):
    sc = cfg["style"]["short"]; lo, hi = float(sc["length_min"]), float(sc["length_max"])
    tail = float(sc["after_last_kill"])
    segs = [list(s) for s in e["segments"]]
    segs[0][0] = max(segs[0][0], e["windows"][0]["start"] - float(sc["lead_in"]))
    L = sum(b - a for a, b in segs)
    if L < lo * 0.8:
        return None
    full = [list(s) for s in segs]
    if L > hi:
        # try every kill as the ending: the short ends tail-seconds after it and reaches back up to `hi`.
        def kept(t):
            u = 0.0
            for a, b in segs:
                if a <= t <= b:
                    return u + (t - a)
                u += b - a
            return None
        best, bv = None, -1
        dbs = e.get("impact_db", [8] * len(e["impacts"]))
        for t, db in zip(e["impacts"], dbs):
            ut = kept(t)
            if ut is None:
                continue
            u1 = min(L, ut + tail); u0 = max(0.0, u1 - hi)
            if u1 - u0 < lo:
                continue
            cand = _trim(segs, u0, u1)
            # each jump (removed lull) inside a short costs: continuous action reads better than fragments
            # (a jump to a short fragment of under 8 s is what makes an edit feel choppy)
            frag = sum(60 if b - a < 8 else 20 for a, b in cand[1:])
            v = _value(e, cand, [x for x in e["impacts"] if x <= t]) + 6 * db - frag
            if v > bv:
                best, bv = cand, v
        if best is None:
            best = _trim(segs, 0, hi)
        segs = best
    inside = [t for t in e["impacts"] if segs[0][0] <= t <= segs[-1][1]]
    if inside:                                     # end on the final kill, let it land
        t = max(inside); new_end = t + tail
        host = next((s for s in full if s[0] <= segs[-1][1] <= s[1] + 1e-6), None)
        if host and new_end > segs[-1][1]:
            segs[-1][1] = min(host[1], new_end)
        cut = _trim(segs, 0, sum(b - a for a, b in segs))   # copy
        if new_end < segs[-1][1]:
            keep = [s for s in cut if s[0] < new_end]; keep[-1][1] = min(keep[-1][1], new_end)
            if sum(b - a for a, b in keep) >= lo:
                segs = keep
    pieces = [_piece(a, b, join="xfade") for a, b in segs]
    if inside and cfg["style"]["slowmo"].get("enabled", True):
        pieces = _slowmo_split(pieces, max(inside), cfg)
    return pieces


def parkour_candidates(events, cfg):
    """Parkour shorts: the best continuous 15-30 s of a climb / rooftop run (the owner liked these as they are)."""
    from .judge import ACTION_ACTS
    sc = cfg["style"]["short"]; lo, hi = float(sc["length_min"]), float(sc.get("parkour_max", 30))
    cands = []
    for e in events:
        pw = [w for w in e["windows"] if ACTION_ACTS.get(w["activity"]) == "parkour"]
        if not pw:
            continue
        s, t = pw[0]["start"] - 0.5, pw[-1]["end"] + 0.5
        best, bv = None, -1
        if t - s <= hi:
            best = (s, t)
            bv = sum(w["intensity"] * (w["end"] - w["start"]) for w in pw)
        else:
            x = s
            while x + hi <= t + 1e-6:
                v = sum(w["intensity"] * max(0.0, min(x + hi, w["end"]) - max(x, w["start"])) for w in pw)
                if v > bv:
                    best, bv = (x, x + hi), v
                x += 0.5
        a, b = best
        cover = sum(max(0.0, min(b, w["end"]) - max(a, w["start"])) for w in pw)
        if b - a >= lo * 0.8 and cover >= 0.6 * (b - a):
            cands.append((bv, e["title"], [_piece(a, b, join="cut")]))
    cands.sort(key=lambda c: -c[0])
    return [(t, p) for _, t, p in cands]


def _chain(items, cfg, cc, n_min, n_max):
    """Chain the best moments, each its own clip, joined by a fast whip (references: Valorant 'lalala' / 'Beggin').
    Opens on the 2nd-best (hook in second 1), timeline order in between, ends on the best one in slow motion."""
    pre, post = float(cc.get("before", 1.5)), float(cc.get("after", 1.2))
    if len(items) < n_min:
        return None, []
    chosen = []
    for k in sorted(items, key=lambda k: -k["v"]):     # no two clips overlapping
        if all(abs(k["t"] - c["t"]) >= pre + post for c in chosen):
            chosen.append(k)
        if len(chosen) >= n_max:
            break
    if len(chosen) < n_min:
        return None, []
    top = chosen[0]
    rest = sorted(chosen[1:], key=lambda k: k["t"])
    opener = max(rest, key=lambda k: k["v"]) if rest else None
    order = ([opener] if opener else []) + [k for k in rest if k is not opener] + [top]
    pieces = []
    for i, k in enumerate(order):
        p = _piece(k["t"] - pre, k["t"] + post, join="xfade")
        p.update(trans=cc.get("transition", "smoothleft"), trans_d=float(cc.get("trans_d", 0.18)))
        if k.get("pu"):
            p.update(cx=k["pu"]["cx"], cy=k["pu"]["cy"])
        if i == len(order) - 1:
            p["end"] = k["t"] + float(cfg["style"]["short"]["after_last_kill"])
        pieces.append(p)
    pieces[0]["join"] = "cut"
    last = pieces[-1]
    head = pieces[:-1]
    tail = _slowmo_split([last], top["t"], cfg, join_after="cut")
    if head and tail:
        tail[0].update(join="xfade", trans=last["trans"], trans_d=last["trans_d"])
    for q in tail:
        if "cx" in last:
            q.setdefault("cx", last["cx"]); q.setdefault("cy", last["cy"])
    return head + tail, [k["t"] for k in chosen]


def _punch_at(e, t):
    return next((p for p in e.get("punch", []) if abs(p["t"] - t) < 0.5), None)


def movement_short(encs, cfg):
    """MOVEMENT-KILL short (owner approved 2026-11-11): kills that came straight out of a move (drop, jump, glide,
    dash, slide, vault, vehicle) - found by judge.find_movement_kills - each from the start of the move to the body
    dropping, best ones chained with whips, the most insane one last in slow motion."""
    mc = cfg["style"].get("movement", {})
    if not mc.get("enabled", True):
        return None, []
    items = [dict(t=m["t"], v=m["wow"] * 3 + _hype(e) / 3, pu=_punch_at(e, m["t"]), caption=m.get("caption", ""))
             for e in encs for m in e.get("moves", [])]
    clip = float(mc.get("before", 3.0)) + float(mc.get("after", 1.2)) - float(mc.get("trans_d", 0.18))
    n_max = int(min(int(mc.get("max_moves", 6)), max(1, float(mc.get("length", 30)) // clip)))
    return _chain(items, cfg, mc, int(mc.get("min_moves", 3)), n_max)


def _hype(e):
    return e["hype"] if e.get("hype") is not None else 5


def killstreak_short(encs, cfg, exclude=()):
    """Valorant/CS frag-montage style: the best kills of the WHOLE recording, ~2.7 s each (the shot that kills
    plus the body dropping), whip transitions, ends on the strongest kill in slow motion. Kills already used by
    the movement short are skipped so the two shorts never show the same moment."""
    kc = cfg["style"].get("killstreak", {})
    if not kc.get("enabled", True):
        return None
    pre, post = float(kc.get("before", 1.5)), float(kc.get("after", 1.2))
    kills = []
    for e in encs:
        if not worth_showing(e, cfg):
            continue
        dbs = e.get("impact_db", [8] * len(e["impacts"]))
        kws = [w for w in e["windows"] if w["kill"]]
        req = ((cfg.get("_game") or {}).get("kill_counter") or {}).get("require_for_shorts", False)
        for w, t, db in zip(kws, e["impacts"], dbs):
            if any(abs(t - x) < pre + post + 2 for x in exclude):
                continue
            if req and cfg.get("_elims") is not None and not w.get("verified"):
                continue
            v = w["intensity"] * 2 + db / 2 + (3 if w["activity"] == "stealth_kill" else 0)
            kills.append(dict(t=t, v=v, pu=_punch_at(e, t)))
    clip = pre + post - float(kc.get("trans_d", 0.18))
    n = int(max(1, float(kc.get("length", 28)) // clip))
    pieces, _ = _chain(kills, cfg, kc, int(kc.get("min_kills", 5)), n)
    return pieces


def plan_shorts(encs, events, cfg):
    sc = cfg["style"]["short"]; xd = float(cfg["style"]["long"]["xfade"])
    themes = sc.get("themes") or list(NAMES)
    out, used = [], []

    def free(a, b):
        return not any(a < y and b > x for x, y in used)

    move_ts = []
    for cat in themes:
        n = 0
        if len(out) >= int(sc.get("max_total", 6)):
            break
        if cat == "movement":
            mv, move_ts = movement_short(encs, cfg)
            if mv:
                out.append(("Movement_Kill_Short_01", dict(theme="Movement", title="Movement Kills", pieces=mv,
                                                           fade_in=0.0, fade_out=float(sc["fade_out"]),
                                                           length=_out_len(mv, xd))))
            continue
        if cat == "killstreak":
            ks = killstreak_short(encs, cfg, exclude=move_ts)
            if ks:
                out.append(("Killstreak_Short_01", dict(theme="Killstreak", title="Killstreak", pieces=ks, fade_in=0.0,
                                                        fade_out=float(sc["fade_out"]), length=_out_len(ks, xd))))
            continue
        if cat == "parkour":
            cands = parkour_candidates(events, cfg)
        else:
            pool = [e for e in encs if worth_showing(e, cfg) and (e["hype"] is None or e["hype"] >= int(sc["min_hype"]))]
            if cat == "stealth":
                pool = [e for e in pool if e["n_stealth"] >= 1]
            else:
                pool = [e for e in pool if e["cat"] == cat or (cat == "combat" and e["cat"] == "stealth" and len(e["kills"]) > e["n_stealth"])]
            cands = []
            for e in sorted(pool, key=lambda e: -(e["score"] + (e["hype"] or 5))):
                p = short_from_encounter(e, cfg)
                if p:
                    cands.append((e["title"], p, e))
        for title, pieces, *src in cands:
            if n >= int(sc["count_per_theme"]) or len(out) >= int(sc.get("max_total", 6)):
                break
            a, b = pieces[0]["start"], pieces[-1]["end"]
            if not free(a, b):
                continue
            used.append((a, b)); n += 1
            e = src[0] if src else {}
            out.append((f"{NAMES[cat]}_Short_{n:02d}", dict(theme=NAMES[cat], title=title, pieces=pieces,
                                                            fade_in=0.0, fade_out=float(sc["fade_out"]),
                                                            length=_out_len(pieces, xd),
                                                            # facts for the on-screen hook (seo._story_hook)
                                                            kills_est=len(e.get("kills", [])), n_stealth=e.get("n_stealth", 0),
                                                            about=f"{e.get('caption', '')} {e.get('hype_why', '')}")))
    return out

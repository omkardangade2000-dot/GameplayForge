"""CINEMATIC MODE - story games (GTA 5 / GTA 6, Bloodlines 2, Phantom Blade Zero ...): find the cutscenes and story
moments people will click on, rank them, and turn them into videos the PUBLISHER allows.

Why it is built this way (research 2026-11-16, details in GTA_CINEMATICS.md):
  * Rockstar's video policy (support.rockstargames.com, updated 2025-01-10): cutscenes are fine "as part of a larger
    play-through (or in a narrated 'Let's Play')", but "isolated videos of the game's cinematics will be removed";
    an "edited combination of cutscenes and/or in-game plot dialogue" and "the ending" count as spoilers.
  * GTA 5 data: 'all cutscenes' movies median 212K views, cinematic edits 38K - the player's VOICE over the story
    (first-time reactions, walkthrough parts, funny moments) is what goes viral, and it is also what is allowed.
So nothing here depends on GTA 6's HUD (unknown until launch): cutscenes are found by the existing detector
(letterbox / HUD gone / camera language) + the AI judge, moments by dialogue + AI rating + the trend radar.

Outputs (per recording):
  Reports/Moments.csv + Moments_Report.txt   every cutscene ranked: viral score, why, trend keys matched,
                                             spoiler level, POLICY STATUS (OK / REVIEW / DO NOT POST)
  Long_Videos/Episode_Part_XX.mp4           the playthrough in order, cutscenes kept IN CONTEXT, dead time cut
  YouTube_Shorts/Moment_Short_XX.mp4        best moments with burned-in subtitles - only where the policy allows
                                             (for GTA: only with the player's own reaction on screen, never raw)
"""
import json, os, re
import numpy as np

MOMENT_PROMPT = (
    "These 4 frames are from ONE story cutscene of the video game {game}. {notes}"
    "Speech heard during it (game dialogue, maybe also the player's own commentary): \"{text}\"\n"
    "Rate this scene as a YouTube moment for gamers. Reply with JSON only: "
    '{{"title": a TRUE 3-7 word title naming who/what (no clickbait lies), "hook": 0-10 how strongly the first seconds '
    'grab a stranger, "humor": 0-10, "shock": 0-10, "emotion": 0-10, "action": 0-10, '
    '"meme_line": the single most quotable line from the speech or "", "characters": [names you can identify], '
    '"spoiler": one of ["none", "minor", "major", "ending"] (major = a death / betrayal / big twist; ending = final '
    'mission or credits), "facecam": true if a webcam image of a real person is overlaid on the game}}'
)


def _num(d, k):
    try:
        return max(0.0, min(10.0, float(d.get(k, 0))))
    except Exception:
        return 0.0


def rate(video, cuts, segs, jd, cfg, workdir, hw):
    """AI rating of every story cutscene (cached). Adds title/hook/.../spoiler/facecam to each cut."""
    from . import judge as J
    from .transcribe import text_between
    cache = os.path.join(workdir, "judge_moments.json")
    known = json.load(open(cache, encoding="utf-8")) if os.path.exists(cache) else {}
    W = int(cfg["judge"]["frame_width"]); H = int(round(W * 9 / 16 / 2)) * 2
    game = (cfg.get("_game") or {}).get("title") or (cfg.get("_game") or {}).get("name", "game")
    todo = []
    for c in cuts:
        c["dialogue"] = text_between(segs, c["start"], c["end"])
        k = f"{J.JUDGE_VERSION}|m1|{c['start']:.2f}|{c['end']:.2f}"
        if k in known:
            c.update(known[k])
        else:
            todo.append((k, c))
    bs = int(cfg["judge"].get("batch", 4))
    for i in range(0, len(todo), bs):
        chunk = todo[i:i + bs]
        items = []
        for k, c in chunk:
            ts = np.linspace(c["start"] + 0.4, c["end"] - 0.4, 4)
            items.append((J.grab(video, ts, W, H, hw),
                          MOMENT_PROMPT.format(game=game, notes=J.notes(cfg), text=c["dialogue"][:700] or "(no speech)")))
        for (k, c), r in zip(chunk, jd.ask(items, 170)):
            d = J.parse_json(r) or {}
            known[k] = dict(m_title=str(d.get("title", ""))[:70], hook=_num(d, "hook"), humor=_num(d, "humor"),
                            shock=_num(d, "shock"), emotion=_num(d, "emotion"), action=_num(d, "action"),
                            meme_line=str(d.get("meme_line", ""))[:140],
                            characters=[str(x)[:30] for x in (d.get("characters") or [])][:6],
                            spoiler=str(d.get("spoiler", "none")).lower() if str(d.get("spoiler", "none")).lower()
                            in ("none", "minor", "major", "ending") else "minor",
                            facecam=str(d.get("facecam", "")).lower() == "true")
            c.update(known[k])
        json.dump(known, open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return cuts


def trend_match(c, keys):
    """Moment keys from the trend radar found in this scene's dialogue / AI title / characters."""
    hay = " ".join([c.get("dialogue", ""), c.get("m_title", ""), c.get("meme_line", ""),
                    " ".join(c.get("characters", []))]).lower()
    hits = [k for k in keys if len(k["key"]) >= 4 and re.search(rf"\b{re.escape(k['key'])}\b", hay)]
    top = max([k["score"] for k in keys] or [1])
    return hits, min(10.0, 10 * sum(k["score"] for k in hits) / (top * 2))


def score(c, keys, cc):
    hits, tr = trend_match(c, keys)
    peak = max(c.get("humor", 0), c.get("shock", 0), c.get("emotion", 0), c.get("action", 0))
    L = c["end"] - c["start"]
    fit = 1.0 if 12 <= L <= 75 else (0.6 if L <= 150 else 0.3)       # a moment, not a 6-minute scene
    meme = 1.0 if len(c.get("meme_line", "")) > 8 else 0.0
    w = cc.get("weights") or {}
    s = (w.get("hook", 0.30) * c.get("hook", 0) + w.get("peak", 0.30) * peak + w.get("trend", 0.25) * tr
         + w.get("meme", 0.5) * meme + w.get("fit", 1.0) * fit)
    c.update(trend_keys=[k["key"] for k in hits], trend=round(tr, 1), viral=round(s, 2))
    return c


def policy(c, cfg, commentary):
    """OK / REVIEW / DO NOT POST + reason, from the game's publisher rules (config games.<g>.policy)."""
    pol = ((cfg.get("_game") or {}).get("policy") or {})
    import datetime as dt
    rel = pol.get("release_date")
    days = None
    if rel:
        try:
            days = (dt.date.today() - dt.date.fromisoformat(str(rel))).days
        except Exception:
            days = None
    hold = int(pol.get("spoiler_hold_days", 30))
    if days is not None and days < 0:
        return "DO NOT POST", "game not released yet - pre-release footage is taken down"
    if c.get("spoiler") == "ending":
        return "DO NOT POST", "the ending - publisher policy treats it as a spoiler (keep it inside a full episode only)"
    if c.get("spoiler") == "major" and days is not None and days < hold:
        return "DO NOT POST", f"major spoiler inside the first {hold} days after release - keep it in the episode only"
    if pol.get("isolated_cinematics") == "forbidden":
        if not commentary:
            return "DO NOT POST", ("publisher removes isolated cinematics - use it inside a narrated episode, or record "
                                   "your reaction (facecam + voice) and post the reaction")
        return "REVIEW", "reaction short: your voice/face must be the point of the video, not the cutscene"
    if c.get("spoiler") == "major":
        return "REVIEW", "major story moment - title it without spoiling"
    return "OK", ""


def episodes(windows, cuts, F, usable, cfg):
    """The playthrough in order: every cutscene in full, gameplay that has something happening, long walks / menus /
    loading cut down. Split into parts of ~episode_minutes at a quiet point. Returns list of parts (lists of groups)."""
    from .planner import _piece
    cc = cfg.get("cinematic") or {}
    keep_explore = float(cc.get("explore_keep", 6.0))
    part_len = float(cc.get("episode_minutes", 25)) * 60
    dip = float(cfg["style"]["long"]["dip"])
    win = float(cfg["judge"]["window_s"])
    spans = []
    for c in cuts:
        spans.append([c["start"], c["end"], "cut", c])
    in_cut = lambda a, b: any(a < c["end"] and b > c["start"] for c in cuts)
    run = []
    for w in windows:
        a, b = w["start"], w["start"] + win
        if in_cut(a, b) or w["activity"] in ("menu_or_map", "loading", "skipped"):
            if run: spans.append(run); run = []
            continue
        if w["activity"] == "exploring" and w["intensity"] <= 3:
            if run and run[2] == "explore":
                run[1] = b
            else:
                if run: spans.append(run)
                run = [a, b, "explore", None]
        else:
            if run and run[2] == "play" and a - run[1] < 0.1:
                run[1] = b
            else:
                if run: spans.append(run)
                run = [a, b, "play", None]
    if run: spans.append(run)
    spans.sort(key=lambda s: s[0])
    pieces = []
    for a, b, kind, c in spans:
        if kind == "explore":
            if b - a > 2 * keep_explore + 2:      # long walk: keep its start and end, dissolve over the middle
                pieces += [(a, a + keep_explore, None), (b - keep_explore, b, None)]
            else:
                pieces.append((a, b, None))
        else:
            pieces.append((a, b, c if kind == "cut" else None))
    parts, cur, L = [], [], 0.0
    for a, b, c in pieces:
        if b - a < 1.0:
            continue
        if L >= part_len and c is None and cur:
            parts.append(cur); cur, L = [], 0.0
        title = (c.get("m_title") or c.get("caption") or "Story")[:60] if c else None
        if c is not None or not cur:
            cur.append(dict(title=title or "Gameplay", pieces=[], fade_in=dip if cur else 0.0, fade_out=dip,
                            cut=c))
        cur[-1]["pieces"].append(_piece(a, b, join="xfade"))
        L += b - a
    if cur:
        parts.append(cur)
    for p in parts:                              # first piece of every group joins with a cut (it is a new section)
        for g in p:
            if g["pieces"]:
                g["pieces"][0]["join"] = "cut"
    return parts


def shorts(cuts, segs, cfg, commentary):
    """Moment Shorts from the top-ranked scenes whose policy status allows it. Burned-in subtitles."""
    from .planner import _piece, _out_len
    from .transcribe import captions
    cc = cfg.get("cinematic") or {}; sc = cfg["style"]["short"]
    mx = float(sc.get("length_max", 58)); n = int(cc.get("max_shorts", 4))
    xd = float(cfg["style"]["long"]["xfade"])
    out = []
    for c in sorted(cuts, key=lambda c: -c.get("viral", 0)):
        if len(out) >= n:
            break
        if c.get("status") == "DO NOT POST" or c.get("viral", 0) < float(cc.get("min_viral", 4.0)):
            continue
        a, b = c["start"], c["end"]
        if b - a > mx:                         # long scene: centre the window on the quotable line if we heard it
            t = None
            ml = (c.get("meme_line") or "").lower()[:24]
            if ml:
                for s in segs:
                    if ml[:12] and ml[:12] in s["text"].lower() and a <= s["start"] <= b:
                        t = s["start"]; break
            st = max(a, min(b - mx, (t - mx * 0.4) if t else a))
            a, b = st, st + mx
        if b - a < 10:
            continue
        ps = [_piece(a, b, join="cut")]
        name = f"Moment_Short_{len(out) + 1:02d}"
        out.append((name, dict(theme="Moment", title=c.get("m_title") or "Story moment", pieces=ps, fade_in=0.0,
                               fade_out=float(sc["fade_out"]), length=_out_len(ps, xd), cut=c,
                               headline=(c.get("m_title") or "")[:40],
                               captions=captions(segs, a, b) if cc.get("subtitles", True) else [],
                               status=c.get("status"), reason=c.get("reason", ""))))
    return out


def write_report(cuts, path_csv, path_txt, game, keys_date):
    import csv
    from .util import fmt
    rows = sorted(cuts, key=lambda c: -c.get("viral", 0))
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Rank", "Start", "End", "Length_s", "Viral", "Status", "Why status", "Title idea", "Hook", "Humor",
                    "Shock", "Emotion", "Action", "Trend", "Trend keys", "Spoiler", "Quotable line", "Characters"])
        for i, c in enumerate(rows, 1):
            w.writerow([i, fmt(c["start"], True), fmt(c["end"], True), round(c["end"] - c["start"], 1), c.get("viral"),
                        c.get("status"), c.get("reason"), c.get("m_title"), c.get("hook"), c.get("humor"),
                        c.get("shock"), c.get("emotion"), c.get("action"), c.get("trend"),
                        "; ".join(c.get("trend_keys", [])), c.get("spoiler"), c.get("meme_line"),
                        ", ".join(c.get("characters", []))])
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(f"STORY MOMENTS - {game}   (trend radar: {keys_date or 'not run - run: gameplayforge trends --game <g>'})\n")
        f.write("Ranked by viral score = hook + strongest emotion (humor/shock/emotion/action) + trend match + quotable line"
                " + good length.\nSTATUS: OK = fine to use | REVIEW = check before posting | DO NOT POST = keep it "
                "inside a full episode only.\n\n")
        for i, c in enumerate(rows[:25], 1):
            f.write(f"{i:>2}. [{c.get('status')}] {fmt(c['start'], True)}-{fmt(c['end'], True)} viral {c.get('viral')}  "
                    f"\"{c.get('m_title')}\"\n")
            if c.get("meme_line"):
                f.write(f"      line: \"{c['meme_line']}\"\n")
            if c.get("trend_keys"):
                f.write(f"      trending keys: {', '.join(c['trend_keys'])}\n")
            if c.get("reason"):
                f.write(f"      why: {c['reason']}\n")

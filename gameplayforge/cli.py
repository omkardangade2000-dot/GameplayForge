import argparse, csv, json, os, shutil, sys, time
import numpy as np
from .util import load_config, probe, fmt, pick_encoder, hw_decode_ok
from . import analyze, segments, highlights, planner, render, qc, judge as J


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gameplayforge")
    ap.add_argument("command", choices=["run", "analyze", "doctor", "trends", "thumbs"])
    ap.add_argument("--title", help="thumbs: the video title (thumbnail words never repeat it)")
    ap.add_argument("--part", help="thumbs: series label, e.g. 'PART 1'")
    ap.add_argument("--mode", choices=["auto", "cinematic"], default="auto",
                    help="cinematic = story moments, episodes and subtitle shorts (story games, GTA)")
    ap.add_argument("--commentary", action="store_true",
                    help="the recording has your own voice/facecam reaction (enables reaction shorts)")
    ap.add_argument("video", nargs="?", help="file name inside paths.input_dir (or a full path)")
    ap.add_argument("--config"); ap.add_argument("--backend", choices=["siglip2", "heuristic"])
    ap.add_argument("--encoder"); ap.add_argument("--force", action="store_true", help="redo analysis")
    ap.add_argument("--long-target", type=float); ap.add_argument("--no-shorts", action="store_true")
    ap.add_argument("--no-long", action="store_true"); ap.add_argument("--no-judge", action="store_true",
                                                                       help="skip the Qwen-VL judge (much less accurate)")
    ap.add_argument("--themes", help="comma list, e.g. combat,stealth")
    ap.add_argument("--game", help="game profile from config.yaml games: (default: auto-detect)")
    ap.add_argument("--layout", choices=["crop34_black", "blur", "full_crop"])
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.backend: cfg["analysis"]["vision_backend"] = a.backend
    if a.encoder: cfg["hardware"]["encoder"] = a.encoder
    if a.long_target: cfg["style"]["long"]["target_seconds"] = a.long_target
    if a.layout: cfg["style"]["short"]["layout"] = a.layout
    if a.themes: cfg["style"]["short"]["themes"] = a.themes.split(",")
    if a.no_judge: cfg["judge"]["enabled"] = False
    if a.game: cfg["game"] = a.game
    if a.command == "doctor":
        from .doctor import main as d; return d(cfg)
    if a.command == "trends":            # trend radar: YouTube titles / views only, nothing is downloaded
        if not a.game: ap.error("trends needs --game, e.g. --game gta6")
        from . import trends; trends.run(a.game, cfg); return
    if not a.video: ap.error("give the video file name")
    video = a.video if os.path.isabs(a.video) else os.path.join(cfg["paths"]["input_dir"], a.video)
    if not os.path.exists(video): sys.exit(f"Video not found: {video}")
    if a.command == "thumbs":              # thumbnails only, in the owner's style (thumbstudio.py)
        from . import thumbstudio as TS
        stem_ = os.path.splitext(os.path.basename(video))[0]
        out_ = os.path.join(cfg["paths"]["output_dir"], stem_, "Long_Videos"); os.makedirs(out_, exist_ok=True)
        work_ = os.path.join(cfg["paths"]["output_dir"], stem_, "_work"); os.makedirs(work_, exist_ok=True)
        gname = a.game or "generic"
        cfg["_game"] = dict((cfg.get("games") or {}).get(gname, {}), name=gname)
        mc = os.path.join(cfg["paths"]["output_dir"], stem_, "Reports", "Moments.csv")
        times, mtitles = [], []
        if os.path.exists(mc):                # story moments from a previous run, best first
            for r in csv.DictReader(open(mc, encoding="utf-8")):
                h, m, s_ = [float(x) for x in r["Start"].split(":")] if r["Start"].count(":") == 2 else [0] + [float(x) for x in r["Start"].split(":")]
                st_ = h * 3600 + m * 60 + s_; L_ = float(r["Length_s"])
                times += [st_ + L_ * 0.3, st_ + L_ * 0.6]
                if r.get("Title idea"): mtitles.append(r["Title idea"])
        dur_ = probe(video)["duration"]
        times += list(np.linspace(dur_ * 0.05, dur_ * 0.95, 24))
        paths = TS.make(video, cfg, work_, out_, "Thumbnail", a.title or "", a.part or "", times,
                        "; ".join(mtitles[:4]))
        print("\n".join(paths) if paths else "no thumbnails made")
        return
    stem = os.path.splitext(os.path.basename(video))[0]
    root = os.path.join(cfg["paths"]["output_dir"], stem)
    work = os.path.join(root, "_work")
    dirs = {k: os.path.join(root, k) for k in ["Cutscenes", "Long_Videos", "YouTube_Shorts", "Reports",
                                               "Original_Project"]}
    for d in dirs.values(): os.makedirs(d, exist_ok=True)
    t0 = time.time(); times = {}; info = probe(video)
    enc = pick_encoder(cfg)[0]; hw = cfg["hardware"]["decode"] in ("auto", "cuda") and hw_decode_ok()
    print(f"Video: {video}\n  {info['width']}x{info['height']} @ {info['fps']:.2f} fps, {fmt(info['duration'], True)}, "
          f"audio={'yes' if info['has_audio'] else 'NO'}  encoder={enc}")

    print("[1/6] Analysing footage ...")
    F = analyze.run(video, cfg, work, force=a.force); times["analyse"] = time.time() - t0
    if a.command == "analyze": return
    backend = str(F["backend"]); fps = int(F["fps"])

    print(f"[2/6] Finding cutscene candidates ...")
    cuts, _ = segments.find_cutscenes(F)
    use_judge = cfg["judge"]["enabled"] and backend == "siglip2"
    jd = None
    if use_judge:
        name = J.pick_model(cfg)
        if not J.model_cached(name):
            print(f"  downloading the judge model {name} (one time, several GB) ...")
        t = time.time(); jd = J.Judge(name); print(f"  judge model {name} loaded ({time.time()-t:.0f}s)")
        J.detect_game(video, info["duration"], jd, cfg, hw)
        ov = (cfg.get("_game") or {}).get("style_overrides")
        if ov:                      # this game's own editing style (e.g. Bloodlines 2 = whole uncut fights)
            from .util import deep_merge
            deep_merge(cfg, ov)
            print(f"  style: {cfg['_game'].get('name')} overrides applied ({', '.join(ov)})")
        if (cfg.get("_game") or {}).get("has_cutscenes", True):
            cuts = J.verify_cutscenes(video, cuts, jd, cfg, work, hw)
        else:      # e.g. BGMI: no story scenes; HUD-less stretches (plane, lobby) are judged window by window
            for c in cuts: c.update(kind="gameplay", caption="")
    else:
        print("  NOTE: Qwen-VL judge is OFF - cutscene/kill-animation split and highlight labels are guesses")
        cuts = segments.classify_without_judge(cuts, F)
    story = [c for c in cuts if c["kind"] in ("story_cutscene", "conversation")]
    kill_anims = [c for c in cuts if c["kind"] == "kill_animation"]
    cut_mask = np.zeros(int(F["n"]), bool)
    for c in story: cut_mask[int(c["start"] * fps):int(c["end"] * fps)] = True
    times["cutscenes"] = time.time() - t0 - sum(times.values())

    print("[3/6] Understanding the gameplay (what happens in every 4-second window) ...")
    usable = highlights.usable_mask(F, cut_mask)
    if jd:
        windows = J.label_windows(video, F, usable, cfg, jd, work, hw)
        windows = J.hunt_kills(video, F, windows, jd, cfg, work, hw)
        J._SMALL["judge"] = jd
        elims, spect = J.read_kill_counter(video, F, cfg, work, hw)
        cfg["_elims"] = elims
        for sa, sb in spect:            # player dead / watching a teammate: never part of any edit
            usable[int(sa * fps):int(sb * fps)] = False
            for w in windows:
                if w["start"] < sb and w["end"] > sa:
                    w.update(activity="menu_or_map", intensity=0, kill=False, enemies=False,
                             caption="spectating teammate (player dead)")
        if elims is not None:                       # the HUD kill count is ground truth
            win = float(cfg["judge"]["window_s"])
            byidx = {int(w["start"] // win): w for w in windows}
            for t in elims:
                # the counter is read every `step` s: the elimination happened in the 2.5 s before the reading
                lo = t - float(cfg["_game"]["kill_counter"].get("step", 2.0)) - 0.5
                fpsF = int(F["fps"]); i0, i1 = int(max(0, lo) * fpsF), int((t + 0.3) * fpsF)
                te = (i0 + int(np.argmax(F["onset"][i0:i1]))) / fpsF if i1 > i0 else t
                w = byidx.get(int(te // win))
                if w is None or w["activity"] in ("skipped", "menu_or_map", "loading"):
                    continue
                w.update(kill=True, verified=True, elim_t=round(te, 2), enemies=True, intensity=max(w["intensity"], 7))
                if w["activity"] not in ("combat", "stealth_kill", "boss_fight"):
                    w["activity"] = "combat"
    else:
        windows = highlights.windows_from_siglip(F, usable, float(cfg["judge"]["window_s"]))
    # Safety net: a short "scene" glued to a fight is a finisher / feeding animation, not a story scene.
    fight = [w for w in windows if w["activity"] in ("combat", "stealth_kill", "boss_fight")]
    for c in list(story):
        if c["end"] - c["start"] < 8 and any(w["end"] > c["start"] - 6 and w["start"] < c["end"] + 6 for w in fight):
            c["kind"] = "kill_animation"; story.remove(c); kill_anims.append(c)
    with open(os.path.join(dirs["Cutscenes"], "cutscenes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Cutscene number", "Start timestamp", "End timestamp", "Duration", "Type", "Detection confidence", "What it shows"])
        for i, c in enumerate(story, 1):
            w.writerow([i, fmt(c["start"], True), fmt(c["end"], True), round(c["end"] - c["start"], 1), c["kind"], round(c["confidence"], 2), c.get("caption", "")])
    for old in os.listdir(dirs["Cutscenes"]):
        if old.startswith("Cutscene_") and old.endswith(".mp4"): os.remove(os.path.join(dirs["Cutscenes"], old))
    render.extract_cutscenes(video, story, dirs["Cutscenes"], cfg, info)
    print(f"  {len(story)} cutscene(s) kept, {len(kill_anims)} kill animation(s) and "
          f"{len(cuts) - len(story) - len(kill_anims)} hidden-HUD gameplay stretch(es) returned to the montage")
    import copy
    encs = highlights.build_encounters(copy.deepcopy(windows), usable, F, cfg, kill_anims)
    events = highlights.build_events(copy.deepcopy(windows), usable, F, cfg, kill_anims)   # parkour shorts
    if jd:
        encs = J.rate_encounters(video, encs, jd, cfg, hw)
        encs = J.title_events(video, encs, jd, cfg, hw)
        encs = J.locate_kills(video, encs, jd, cfg, work, hw)
        encs = J.find_movement_kills(video, F, encs, jd, cfg, work, hw)
        if elims is not None and (cfg.get("_game") or {}).get("edit_mode") == "fights":
            from . import fightedit, killedit
            te_all = sorted(w["elim_t"] for w in windows if w.get("verified"))
            shots = fightedit.gunfire(video, cfg, work)
            # knocks + finishes from the kill feed, read only around the player's own gunfire
            near = []
            for s in shots:
                if near and s - 6 <= near[-1][1]:
                    near[-1][1] = s + 6
                else:
                    near.append([max(0.0, s - 6), s + 6])
            if (cfg["_game"].get("kill_feed") or {}).get("mode") == "name":     # CS2: known name, no counter
                feed = J.read_own_kills(video, cfg, work, hw, spect)
            else:
                feed = J.read_killfeed(video, near, F, te_all, cfg, work, hw)
            kills_all = sorted(te_all)
            for t in feed:                     # a counter rise right after a feed line is the same kill
                if not any(-1.0 <= k - t <= 4.5 for k in te_all):
                    kills_all.append(t)
            kills_all = sorted(kills_all)
            # looting: pickup clicks sound like shots - drop 'shots' fired while the loot text / list is on screen
            loot = J.read_loot(video, near, cfg, work, hw)
            n0 = len(shots)
            # the pickup CLICK comes up to 2 s before the 'Picked up' text appears
            shots = [s for s in shots if not any(int(np.floor(s)) + d in loot for d in (0, 1, 2))]
            print(f"  gunfire: {n0 - len(shots)} of {n0} 'shots' were loot pickups, removed")
            by_feed = (cfg["_game"].get("kill_feed") or {}).get("mode") == "name"
            if by_feed:                        # CS2: the kill feed is complete, the audio is not -> kills drive fights
                fights = fightedit.fights_from_kills(kills_all, shots, usable, F, cfg, killedit.match_ids(F))
            else:
                fights = fightedit.find_fights(shots, kills_all, usable, F, cfg, killedit.match_ids(F))
            labels = J.review_seconds(video, [(f["start"], f["end"]) for f in fights], jd, cfg, work, hw)
            if by_feed:
                ammo = J.read_ammo_shots(video, [(f["start"], f["end"]) for f in fights], cfg, work, hw)
                if ammo is not None:           # measured own shots replace audio + AI guesses
                    ammo = np.array(ammo)
                    fights = [dict(f, shots=[float(s) for s in ammo if f["start"] <= s <= f["end"]]) for f in fights]
                else:
                    fights = [fightedit.add_review_shots(f, labels) for f in fights]
            fights = [fightedit.tighten(f, labels, cfg, loot) for f in fights]
            fights = [f for f in fights if f["kills"] and f["pieces"]]
            for f in fights:
                f["pieces"] = list(f["pieces"])
            fights = J.rate_clips(video, fights, jd, cfg, hw)
            cfg["_fights"] = fights
            print(f"  fights: {len(fights)} with confirmed kills, {sum(len(f['kills']) for f in fights)} kills, "
                  f"{sum(f['length'] for f in fights):.0f} s of combat")
        elif elims is not None and (cfg.get("_game") or {}).get("edit_mode") == "kills":
            from . import killedit
            te_all = sorted(w["elim_t"] for w in windows if w.get("verified"))
            spans = killedit.candidate_spans(te_all, usable, F, cfg)
            labels = J.review_seconds(video, spans, jd, cfg, work, hw)
            kclips = killedit.build_clips(spans, labels, cfg, killedit.match_ids(F))
            kclips = J.rate_clips(video, kclips, jd, cfg, hw)
            cfg["_kclips"] = kclips
            print(f"  kill clips: {len(kclips)} clips, {sum(len(c['kills']) for c in kclips)} kills, "
                  f"{sum(y - x for c in kclips for x, y in c['pieces']):.0f} s kept after self-review")
        cine = a.mode == "cinematic" or (cfg.get("_game") or {}).get("edit_mode") == "cinematic"
        if cine and story:                 # story moments: dialogue + AI rating + trend radar + publisher policy
            from . import cinematic, transcribe as TR, trends as TRD
            segs = TR.transcribe(video, work, cfg)
            story = cinematic.rate(video, story, segs, jd, cfg, work, hw)
            gp = cfg.get("_game") or {}
            keys = TRD.load_keys(gp.get("name", ""), cfg) or \
                TRD.load_keys((gp.get("trend") or {}).get("keys_from", ""), cfg)
            commentary = bool(a.commentary) or sum(bool(c.get("facecam")) for c in story) >= max(1, len(story) // 3)
            for c in story:
                cinematic.score(c, keys, cfg.get("cinematic") or {})
                c["status"], c["reason"] = cinematic.policy(c, cfg, commentary)
            cfg["_cine"] = dict(cuts=story, segs=segs, commentary=commentary, keys=keys)
            print(f"  story moments: {len(story)} rated, commentary={'yes' if commentary else 'no'}, "
                  f"trend keys {len(keys)}; status " + ", ".join(
                      f"{s} {sum(c['status'] == s for c in story)}" for s in ("OK", "REVIEW", "DO NOT POST")))
        events = J.title_events(video, [e for e in events if e["cat"] == "parkour"], jd, cfg, hw); jd.close()
    print(f"  {len(encs)} encounter(s): " + ", ".join(
        f"{e['title']} {fmt(e['start'])}-{fmt(e['end'])} k{len(e['kills'])} h{e['hype']}" for e in encs))
    times["understand"] = time.time() - t0 - sum(times.values())

    print("[4/6] Planning the edits ...")
    parts = []
    if cfg.get("_cine") is not None:            # story game: episodes (cutscenes in context) + moment shorts
        from . import cinematic
        cn = cfg["_cine"]
        parts = [] if a.no_long else cinematic.episodes(windows, cn["cuts"], F, usable, cfg)
        groups = []
        shorts = [] if a.no_shorts else cinematic.shorts(cn["cuts"], cn["segs"], cfg, cn["commentary"])
        if not a.no_shorts and (cfg.get("cinematic") or {}).get("gameplay_shorts", True):
            shorts += planner.plan_shorts(encs, events, cfg)       # action / chaos from gameplay = always allowed
        from .trends import load_keys
        gp = cfg.get("_game") or {}
        kd = os.path.join(cfg["paths"]["output_dir"], "_Trends", gp.get("name", ""), "latest_moment_keys.json")
        cinematic.write_report(cn["cuts"], os.path.join(dirs["Reports"], "Moments.csv"),
                               os.path.join(dirs["Reports"], "Moments_Report.txt"), gp.get("title") or gp.get("name"),
                               json.load(open(kd, encoding="utf-8")).get("date") if os.path.exists(kd) else
                               ("borrowed from " + (gp.get("trend") or {}).get("keys_from", "") if cn["keys"] else None))
        with open(os.path.join(dirs["Cutscenes"], "README_DO_NOT_UPLOAD.txt"), "w", encoding="utf-8") as f:
            f.write("These cutscene files are for YOUR reference (finding moments, checking spoilers).\n"
                    "Do not upload them on their own: e.g. Rockstar removes 'isolated videos of the game's cinematics'.\n"
                    "Use Long_Videos/Episode_Part_XX.mp4 (cutscenes inside the playthrough) and the Shorts instead.\n")
    elif cfg.get("_fights") is not None:          # fight-centred edit (BGMI etc.)
        from . import fightedit
        groups = [] if a.no_long else fightedit.plan_long(cfg["_fights"], cfg)
        shorts = [] if a.no_shorts else fightedit.plan_shorts(cfg["_fights"], cfg)
        with open(os.path.join(dirs["Reports"], "Fights.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["Start", "End", "Kept_s", "Kills", "Own shots", "AI wow", "Score", "Worthy"])
            for x in cfg["_fights"]:
                w.writerow([fmt(x["start"], True), fmt(x["end"], True), round(x["length"], 1), len(x["kills"]),
                            len(x["shots"]), x["wow"], round(fightedit.score(x), 1), fightedit.worthy(x, cfg)])
    elif cfg.get("_kclips") is not None:          # kill-centred edit (BGMI etc.)
        from . import killedit
        groups = [] if a.no_long else killedit.plan_long(cfg["_kclips"], cfg)
        shorts = [] if a.no_shorts else killedit.plan_shorts(cfg["_kclips"], cfg)
    else:
        groups = [] if a.no_long else planner.plan_long(encs, cfg)
        shorts = [] if a.no_shorts else planner.plan_shorts(encs, events, cfg)
    shown = {id(g["enc"]) for g in groups if "enc" in g}
    with open(os.path.join(dirs["Reports"], "Encounters.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Encounter", "Title", "Category", "Start", "End", "Kept_s", "Kills", "Stealth kills",
                                       "Score", "Hype", "In montage", "Why (judge)", "Best moment"])
        for i, e in enumerate(encs, 1):
            why = "yes" if id(e) in shown else ("no - single ordinary kill" if not highlights.worth_showing(e, cfg) else "no - not exciting enough")
            w.writerow([i, e["title"], e["cat"], fmt(e["start"], True), fmt(e["end"], True), e["length"], len(e["kills"]),
                        e["n_stealth"], e["score"], e["hype"], why, e["hype_why"], e["caption"]])
    with open(os.path.join(dirs["Reports"], "Highlights.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Output", "Section", "Piece start", "Piece end", "Speed", "Join"])
        for g in groups:
            for p in g["pieces"]: w.writerow(["Long", g["title"], fmt(p["start"], True), fmt(p["end"], True), p["speed"], p["join"]])
        for n, sp in shorts:
            for p in sp["pieces"]: w.writerow([n, sp["title"], fmt(p["start"], True), fmt(p["end"], True), p["speed"], p["join"]])
    with open(os.path.join(dirs["Reports"], "Window_Labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Start", "End", "Activity", "Intensity", "Kill", "Enemies", "Caption"])
        for x in windows: w.writerow([fmt(x["start"], True), fmt(x["end"], True), x["activity"], x["intensity"], x["kill"],
                                      x.get("enemies", ""), x["caption"]])

    print("[5/6] Rendering ...")
    for d in ("Long_Videos", "YouTube_Shorts"):
        for old in os.listdir(dirs[d]):
            if old.endswith((".mp4", "_UPLOAD.txt")): os.remove(os.path.join(dirs[d], old))
    long_out, marks, short_outs = None, [], []
    from . import seo                      # title / description / tags for every video (no API)
    for n, sp in shorts:                   # the short's on-screen headline = its title hook (never over-claims)
        sp["headline"], sp["big_word"] = seo.short_hook(sp, cfg)
    for old in os.listdir(dirs["Long_Videos"]):
        if "Thumbnail_" in old and old.endswith(".jpg"): os.remove(os.path.join(dirs["Long_Videos"], old))
    if groups: long_out, marks = render.render_long(video, groups, dirs["Long_Videos"], cfg, info)
    ep_outs = []
    for i, p in enumerate(parts, 1):
        o, _ = render.render_long(video, p, dirs["Long_Videos"], cfg, info, name=f"Episode_Part_{i:02d}.mp4")
        ep_outs.append((o, p))
    if shorts: short_outs = render.render_shorts(video, shorts, dirs["YouTube_Shorts"], cfg, info)
    seo.write_all(long_out, groups, short_outs, shorts, cfg)
    for i, (o, p) in enumerate(ep_outs, 1):
        seo.write_episode(o, p, i, cfg)
        try:                               # owner's style first (face cut-out + moment), simple frames as fallback
            cuts_ = sorted([g["cut"] for g in p if g.get("cut")], key=lambda c: -c.get("viral", 0))
            tms = [c["start"] + (c["end"] - c["start"]) * f for c in cuts_ for f in (0.3, 0.6)]
            tms += [(x["start"] + x["end"]) / 2 for g in p for x in g["pieces"]]
            if not _studio(video, cfg, work, dirs["Long_Videos"], f"Episode_Part_{i:02d}_Thumbnail", o, f"PART {i}",
                           tms, cuts_[0].get("m_title", "") if cuts_ else ""):
                from . import thumbs
                thumbs.make(video, p, seo.episode_thumb_words(p, i, cfg), dirs["Long_Videos"], cfg,
                            prefix=f"Episode_Part_{i:02d}_Thumbnail")
        except Exception as e:
            print(f"  WARNING: episode thumbnails not written: {e!r}")
    if long_out:
        try:
            tms = [g["kill_t"] - 0.3 for g in groups if g.get("kill_t") is not None]
            tms += [(x["start"] + x["end"]) / 2 for g in groups for x in g["pieces"]]
            if not _studio(video, cfg, work, dirs["Long_Videos"], "Thumbnail", long_out, "", tms, ""):
                from . import thumbs
                thumbs.make(video, groups, seo.thumb_words(groups, cfg), dirs["Long_Videos"], cfg)
        except Exception as e:             # thumbnails must never cost the rendered videos their QC
            print(f"  WARNING: thumbnails not written: {e!r}")
    times["render"] = time.time() - t0 - sum(times.values())

    print("[6/6] Quality control ...")
    results = []
    if long_out:
        tgt = cfg["style"]["long"]["target_seconds"]
        r = qc.check(long_out, {"size": render._long_size(cfg, info), "min_s": 0.75 * tgt, "max_s": 1.25 * tgt, "audio": info["has_audio"], "strict": False}, "long")
        if not 540 <= r.get("duration", 0) <= 1200:
            r["notes"].append("outside the 9-20 min goal (needs more action in the recording)")
        results.append(r); qc.contact_sheet(long_out, os.path.join(dirs["Reports"], "Montage_ContactSheet.png"), 24)
        allc = [dict(start=p["start"], end=p["end"]) for g in groups if not g.get("cold") for p in g["pieces"]]
        ov = qc.overlap_check(allc)
        if ov: results[-1]["ok"] = False; results[-1]["notes"].append(f"overlapping source ranges: {ov}")
    for o, p in ep_outs:
        r = qc.check(o, {"size": render._long_size(cfg, info), "min_s": 30, "max_s": 6 * 3600, "audio": info["has_audio"],
                         "strict": False}, "long")
        results.append(r); qc.contact_sheet(o, os.path.join(dirs["Reports"], os.path.basename(o)[:-4] + "_ContactSheet.png"), 24)
    for p in short_outs:
        results.append(qc.check(p, {"size": (cfg["style"]["short"]["width"], cfg["style"]["short"]["height"]), "min_s": 10, "max_s": float(cfg["style"]["short"]["length_max"]) + 3, "audio": info["has_audio"]}, "short"))
        qc.contact_sheet(p, os.path.join(dirs["Reports"], os.path.basename(p)[:-4] + "_ContactSheet.png"), 12)
    sec = results[0]["duration"] if long_out and results else 0
    with open(os.path.join(dirs["Reports"], "Editing_Report.txt"), "w", encoding="utf-8") as f:
        f.write(f"GameplayForge report - {os.path.basename(video)}\n")
        f.write(f"Source: {info['width']}x{info['height']} {info['fps']:.2f}fps {fmt(info['duration'], True)}\n")
        f.write(f"Image model: {backend}   Judge: {cfg['judge']['model'] if jd else 'OFF (labels are guesses)'}\n")
        f.write(f"Encoder: {enc}\nCutscenes kept: {len(story)}   kill animations returned to montage: {len(kill_anims)}\n")
        f.write(f"Encounters found: {len(encs)}\n")
        for e in encs:
            f.write(f"  {fmt(e['start'])}-{fmt(e['end'])}  {e['cat']:<8} kills {len(e['kills'])} (stealth {e['n_stealth']})  "
                    f"hype {e['hype']}  score {e['score']:<5} {'IN MONTAGE' if id(e) in shown else 'left out'}  "
                    f"{e['title']} - {e['hype_why']}\n")
        f.write(f"\nLong montage: {sec:.0f}s (target {cfg['style']['long']['target_seconds']}s)\n")
        if sec < 0.9 * cfg["style"]["long"]["target_seconds"]:
            f.write("  NOTE: the recording does not contain enough strong action for the requested length - nothing was padded.\n")
        for g in groups: f.write(f"  section '{g['title']}': {len(g['pieces'])} piece(s), "
                                 f"{sum((p['end']-p['start'])/p['speed'] for p in g['pieces']):.0f}s\n")
        f.write(f"Shorts: {len(shorts)}  " + ", ".join(f"{n} ({sp['title']}, {sp['length']:.0f}s)" for n, sp in shorts) + "\n")
        missing = [t for t in (cfg['style']['short'].get('themes') or []) if not any(sp['theme'].lower() == t for _, sp in shorts)]
        if missing: f.write(f"  no strong enough event for: {', '.join(missing)} (skipped instead of padding)\n")
        f.write("\nQC results:\n")
        for r in results: f.write(json.dumps(r) + "\n")
        f.write("\nTime per stage: " + ", ".join(f"{k} {v/60:.1f} min" for k, v in times.items()))
        f.write(f"\nTotal time: {(time.time()-t0)/60:.1f} min\n")
    shutil.copy(cfg_path(a.config), os.path.join(dirs["Original_Project"], "config.yaml"))
    print("Done ->", root)
    for r in results: print(("  OK   " if r["ok"] else "  FAIL ") + r["file"], r.get("duration"), "s", r["notes"])


def cfg_path(p):
    return p or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def _studio(video, cfg, work, outdir, prefix, out_video, part, times, moment):
    """Thumbnail studio (owner's style). Title = first title option of the video's upload file."""
    from . import thumbstudio as TS
    title = ""
    up = out_video[:-4] + "_UPLOAD.txt"
    if os.path.exists(up):
        for ln in open(up, encoding="utf-8"):
            if ln.strip().startswith("1."):
                title = ln.strip()[2:].strip(); break
    return TS.make(video, cfg, work, outdir, prefix, title, part, times, moment)

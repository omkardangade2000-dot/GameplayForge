"""Stage 3: the visual judge.

A local vision-language model (Qwen3-VL, Apache-2.0, runs on the RTX GPU) looks at the footage and
decides what the player is actually doing - this is the step that understands the scene instead of
guessing from loudness or motion.
  * label_windows   : every usable 4-second window of gameplay (4 frames, 1 s apart) gets
                      activity / intensity / kill / caption.
  * verify_cutscenes: each cutscene candidate is classified as story cutscene, conversation,
                      kill animation or gameplay.
  * title_events    : a short YouTube-chapter title for each selected event.
Results are cached in _work/ so re-running with other edit settings does not re-judge.
"""
import json, os, re, subprocess, sys
import numpy as np

JUDGE_VERSION = "j5"
ACTION_ACTS = {"combat": "combat", "stealth_kill": "stealth", "parkour": "parkour", "boss_fight": "boss"}
ACTIVITIES = list(ACTION_ACTS) + ["exploring", "dialogue", "menu_or_map", "cutscene", "loading"]

WINDOW_PROMPT = (
    "These {k} frames are {step:g} second(s) apart, in time order, from a recording of someone playing a "
    "{view} video game. Decide what the PLAYER is doing in this moment.\n"
    "Activities:\n"
    "- combat: shooting, aiming a weapon at enemies (also from long range), melee fighting, using powers or "
    "grenades on enemies, being attacked, finishing an enemy during a fight.\n"
    "- stealth_kill: sneaking up on unaware enemies - crouching, hiding, tracking them - and taking them down "
    "silently.\n"
    "- parkour: climbing a wall or ledge, jumping or gliding between rooftops, dropping from a height.\n"
    "- boss_fight: fighting one large or powerful named enemy with a big health bar.\n"
    "- exploring: walking, running, driving, looting or looking around with NO enemies nearby.\n"
    "- dialogue: the player stands facing a talking character, or dialogue choices are shown. Subtitles "
    "alone do NOT mean dialogue; if a weapon is raised or enemies are being shot or hit, it is combat.\n"
    "- menu_or_map: a menu, map or inventory screen. cutscene: a scripted cinematic. loading: a loading or "
    "title screen.\n"
    "{notes}"
    "Reply with JSON only, no other text:\n"
    '{{"activity": one of {acts}, '
    '"intensity": integer 0-10, how good this moment is for an action highlight video: walking or talking 0-1, '
    "traversal 2-3, sneaking or aiming at enemies 4-5, ordinary fighting 6-7, a spectacular moment "
    "(headshot, several enemies down quickly, explosion, finisher, clutch against several enemies) 8-10, "
    '"kill": true if an enemy dies or is knocked down in these frames, else false, '
    '"enemies": true if hostile enemies are visible or being targeted, else false, '
    '"caption": what happens, at most 12 words}}'
)
HYPE_PROMPT = (
    "These frames are from ONE fight or stealth sequence in a video game, in time order. Rate how impressive "
    "it is for a gaming highlight montage that must keep viewers watching. 9-10: clutch or cinematic - player "
    "outnumbered, many enemies down quickly, headshots, explosions, brutal finishers, flashy powers, squad wipes. "
    "6-8: a solid fight with several kills. 3-5: an ordinary fight, one or two enemies, little variety. 1-2: "
    "almost nothing happens. Reply with JSON only: "
    '{"hype": integer 1-10, "why": at most 12 words}'
)
CUTSCENE_PROMPT = (
    "These frames are from a video game recording. They come from a stretch where the game's HUD may be hidden. "
    "Which is it? Reply with JSON only: "
    '{"type": one of ["story_cutscene", "conversation", "kill_animation", "gameplay"], "caption": at most 12 words}. '
    "story_cutscene = a scripted cinematic scene; conversation = talking with a character; "
    "kill_animation = a short scripted takedown or finisher on an enemy during gameplay (a close-up, often "
    "right after a fight); "
    "gameplay = the player is simply playing."
)
TITLE_PROMPT = (
    "These frames show one action sequence from a video game. Write a short, punchy YouTube chapter title "
    "for it, 2 to 4 words, no quotes, no game name, no numbering. Name WHAT HAPPENS or WHERE (the fight, the "
    "place, the weapon, the outcome), never the player's own character, outfit, hair or colours. Examples: "
    "Rooftop Ambush, Park Shootout, Silent Takedown, Bridge Squad Wipe, Sniper Duel, 1v3 Clutch. "
    "Reply with the title only."
)


def notes(cfg):
    """Game-specific hints for every AI question (config.yaml: games.<profile>.notes)."""
    g = cfg.get("_game") or {}
    return ("Notes about this game: " + g["notes"].strip() + "\n") if g.get("notes") else ""


GAME_PROMPT = ("Which video game is this? Read any game logo or text on screen (HUD labels, kill feed, watermark). "
               "Choose from this list: {opts}. Reply with JSON only: "
               '{{"choice": one item from the list, "game": the game name you see}}')


def _set_game(cfg, name, games, **extra):
    global JUDGE_VERSION
    cfg["_game"] = dict(games.get(name, {}), name=name, **extra)
    JUDGE_VERSION = f"j5-{name}"            # every cached AI answer is per game profile


def detect_game(video, duration, judge, cfg, hw):
    """config game: auto -> 1) a profile `match` word in the file or folder name (e.g. 'BGMI_...mp4', '...\\pubg\\'),
    2) else the AI picks from the list of known games after reading on-screen logos/text (8 frames). Free naming
    proved unreliable: it called BGMI 'Garena Free Fire'. Unknown game = generic prompts, fewer hints."""
    want = str(cfg.get("game", "auto")).lower()
    games = cfg.get("games") or {}
    if want != "auto":
        _set_game(cfg, want, games)
        return want
    path = video.lower()
    name = next((k for k, g in games.items() if any(m in path for m in g.get("match", []))), None)
    how = "file name"
    if not name:
        labels = {k: " / ".join(g.get("match", [k])) for k, g in games.items()}
        opts = json.dumps(list(labels.values()) + ["other"])
        ts = list(np.linspace(duration * 0.1, duration * 0.9, 8))
        d = parse_json(judge.ask([(grab(video, ts, 960, 540, hw), GAME_PROMPT.format(opts=opts))], 40)[0]) or {}
        said = str(d.get("choice", "")).lower()
        name = next((k for k, lab in labels.items() if lab.lower() == said or
                     any(m in said for m in games[k].get("match", []))), None)
        how = f"AI chose '{said}' (sees '{d.get('game', '')}')"
    _set_game(cfg, name or "generic", games)
    if not name:
        cfg["_game"]["view"] = "first-person or third-person"
    print(f"  game: {how} -> profile {name or 'generic'}")
    return name or "generic"


def pick_model(cfg):
    """First configured judge model that is already downloaded; else the first one (it will download)."""
    names = cfg["judge"].get("models") or [cfg["judge"]["model"]]
    chosen = next((n for n in names if model_cached(n)), names[0])
    cfg["judge"]["model"] = chosen
    return chosen


def model_cached(name):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(name, local_files_only=True)
        return True
    except Exception:
        return False


def _decode_cmd(video, rate, w, h, hw, ss=None):
    c = ["ffmpeg", "-v", "error"] + (["-hwaccel", "cuda"] if hw else [])
    if ss is not None:
        c += ["-ss", f"{ss:.3f}"]
    return c + ["-i", video, "-an", "-vf", f"fps={rate},scale={w}:{h}:flags=area", "-pix_fmt", "rgb24",
                "-f", "rawvideo", "-"]


def grab(video, times, w, h, hw):
    """Single frames at the given timestamps (fast keyframe seek, then exact decode)."""
    from PIL import Image
    out = []
    for t in times:
        c = ["ffmpeg", "-v", "error"] + (["-hwaccel", "cuda"] if hw else []) + \
            ["-ss", f"{max(0.0, t):.3f}", "-i", video, "-frames:v", "1", "-vf", f"scale={w}:{h}:flags=area",
             "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
        raw = subprocess.run(c, capture_output=True).stdout
        if len(raw) >= w * h * 3:
            out.append(Image.fromarray(np.frombuffer(raw[:w * h * 3], np.uint8).reshape(h, w, 3)))
    return out


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace("'", '"').replace("True", "true").replace("False", "false"))
        except Exception:
            return None


class Judge:
    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else "cpu").eval()
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.proc.tokenizer.padding_side = "left"

    def ask(self, items, max_new_tokens=80):
        """items: list of (list of PIL images, prompt text) -> list of reply strings (one batch)."""
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im} for im in ims] +
                  [{"type": "text", "text": txt}]}] for ims, txt in items]
        inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


def _clean_window(d):
    if not isinstance(d, dict):
        return None
    act = str(d.get("activity", "")).strip().lower().replace(" ", "_")
    if act not in ACTIVITIES:
        act = "exploring"
    try:
        inten = int(float(d.get("intensity", 0)))
    except Exception:
        inten = 0
    kill = d.get("kill", False)
    kill = kill if isinstance(kill, bool) else str(kill).lower() == "true"
    en = d.get("enemies", False)
    en = en if isinstance(en, bool) else str(en).lower() == "true"
    return dict(activity=act, intensity=max(0, min(10, inten)), kill=bool(kill), enemies=bool(en),
                caption=str(d.get("caption", ""))[:120])


def label_windows(video, F, usable, cfg, judge, workdir, hw):
    """Stream the video once at (frames_per_window / window_s) fps and judge every usable window."""
    jc = cfg["judge"]
    win, k, W = float(jc["window_s"]), int(jc["frames_per_window"]), int(jc["frame_width"])
    H = int(round(W * 9 / 16 / 2)) * 2
    key = f"{JUDGE_VERSION}|{(cfg.get('_game') or {}).get('name')}|{jc['model']}|{win}|{k}|{W}"
    cache = os.path.join(workdir, "judge_windows.json")
    if os.path.exists(cache):
        c = json.load(open(cache, encoding="utf-8"))
        if c.get("key") == key and abs(c.get("duration", 0) - float(F["duration"])) < 1:
            print("  using cached judge labels:", cache)
            return c["windows"]
    fps = int(F["fps"]); step = win / k
    prompt = WINDOW_PROMPT.format(k=k, step=step, acts=json.dumps(ACTIVITIES), notes=notes(cfg),
                                  view=(cfg.get("_game") or {}).get("view", "first-person or third-person"))
    from PIL import Image
    p = subprocess.Popen(_decode_cmd(video, 1.0 / step, W, H, hw), stdout=subprocess.PIPE, bufsize=10 ** 8)
    size = W * H * 3
    windows, frames, batch, i = [], [], [], 0
    nwin_total = int(float(F["duration"]) // win)
    todo = sum(1 for w in range(nwin_total) if usable[int(w * win * fps):int((w + 1) * win * fps)].mean() >= 0.5)
    done = 0

    def flush():
        nonlocal done
        if not batch:
            return
        replies = judge.ask([(b["imgs"], prompt) for b in batch])
        for b, r in zip(batch, replies):
            d = _clean_window(parse_json(r)) or dict(activity="exploring", intensity=0, kill=False, enemies=False,
                                                     caption="(unreadable)")
            windows.append(dict(start=b["start"], end=b["end"], **d))
        done += len(batch); batch.clear()
        print(f"    judged {done}/{todo} windows", file=sys.stderr, flush=True)

    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        frames.append(Image.fromarray(np.frombuffer(buf, np.uint8).reshape(H, W, 3)))
        i += 1
        if len(frames) == k:
            w = (i - 1) // k
            a, b = w * win, (w + 1) * win
            seg = usable[int(a * fps):int(b * fps)]
            if seg.size and seg.mean() >= 0.5:
                batch.append(dict(start=a, end=b, imgs=frames))
                if len(batch) >= int(jc["batch"]):
                    flush()
            else:
                windows.append(dict(start=a, end=b, activity="skipped", intensity=0, kill=False, enemies=False,
                                    caption=""))
            frames = []
    p.wait(); flush()
    windows.sort(key=lambda d: d["start"])
    json.dump(dict(key=key, duration=float(F["duration"]), windows=windows), open(cache, "w", encoding="utf-8"), indent=1)
    return windows


KILL_PROMPT = (
    "These 4 frames are 0.8 seconds apart, in time order, around a sudden loud sound in a video game recording "
    "(first-person or third-person). What did the PLAYER just do? {notes}Reply with JSON only: "
    '{"attack": true if the player shot, hit, slashed or used a power on an enemy, '
    '"kill": true if an enemy went down, was knocked down or died, '
    '"stealth": true if it was a silent takedown of an unaware enemy, '
    '"caption": at most 10 words}. Footsteps, doors, jumping or walking are NOT attacks.'
)


def hunt_kills(video, F, windows, judge, cfg, workdir, hw):
    """Second pass: the 4-frames-per-4-s window labels can step over a 1-second kill (a headshot from range, a
    quick stealth takedown). Every sudden loud sound (gunshot, hit, feeding) outside an already-detected fight is
    checked with 4 frames tightly around it, and the window is upgraded if it really was an attack."""
    from scipy.signal import find_peaks
    jc = cfg["judge"]; fps = int(F["fps"]); W = int(jc["frame_width"]); H = int(round(W * 9 / 16 / 2)) * 2
    win = float(jc["window_s"])
    pk, _ = find_peaks(F["onset"], height=float(jc.get("hunt_min_db", 7)), distance=fps)
    byidx = {int(w["start"] // win): w for w in windows}
    cands = []
    for p in pk:
        t = p / fps; w = byidx.get(int(t // win))
        if w is None or w["activity"] in ("skipped", "menu_or_map", "cutscene", "loading") or w["kill"]:
            continue
        cands.append(round(t, 2))
    cache = os.path.join(workdir, "judge_kills.json")
    known = json.load(open(cache, encoding="utf-8")) if os.path.exists(cache) else {}
    todo = [t for t in cands if f"{JUDGE_VERSION}|{jc['model']}|{t}" not in known]
    for i in range(0, len(todo), int(jc["batch"])):
        chunk = todo[i:i + int(jc["batch"])]
        replies = judge.ask([(grab(video, [t - 1.2, t - 0.4, t + 0.4, t + 1.2], W, H, hw),
                              KILL_PROMPT.replace("{notes}", notes(cfg))) for t in chunk], 60)
        for t, r in zip(chunk, replies):
            d = parse_json(r) or {}
            known[f"{JUDGE_VERSION}|{jc['model']}|{t}"] = {k: (str(d.get(k, "")).lower() == "true" if k != "caption" else
                                                              str(d.get(k, ""))[:100]) for k in ("attack", "kill", "stealth", "caption")}
    json.dump(known, open(cache, "w", encoding="utf-8"), indent=1)
    found = 0
    for t in cands:
        d = known.get(f"{JUDGE_VERSION}|{jc['model']}|{t}", {})
        if not (d.get("attack") or d.get("kill")):
            continue
        w = byidx[int(t // win)]
        act = "stealth_kill" if d.get("stealth") and d.get("kill") else ("combat" if w["activity"] not in ("combat", "stealth_kill", "boss_fight") else w["activity"])
        w.update(activity=act, intensity=max(w["intensity"], 8 if d.get("kill") else 6), kill=w["kill"] or bool(d.get("kill")),
                 enemies=True, caption=(d.get("caption") or w["caption"]) + " [audio+vision check]")
        found += 1
    print(f"  kill hunter: {len(cands)} loud moments checked, {found} were attacks/kills the first pass missed")
    return windows


def verify_cutscenes(video, cuts, judge, cfg, workdir, hw):
    W = int(cfg["judge"]["frame_width"]); H = int(round(W * 9 / 16 / 2)) * 2
    cache = os.path.join(workdir, "judge_cutscenes.json")
    known = {}
    if os.path.exists(cache):
        known = json.load(open(cache, encoding="utf-8"))
    for c in cuts:
        k = f"{JUDGE_VERSION}|{c['start']:.2f}|{c['end']:.2f}"
        if k not in known:
            ts = np.linspace(c["start"] + 0.3, c["end"] - 0.3, 4)
            d = parse_json(judge.ask([(grab(video, ts, W, H, hw), notes(cfg) + CUTSCENE_PROMPT)])[0]) or {}
            t = str(d.get("type", "story_cutscene")).strip().lower()
            known[k] = dict(kind=t if t in ("story_cutscene", "conversation", "kill_animation", "gameplay") else "story_cutscene",
                            caption=str(d.get("caption", ""))[:120])
        c.update(known[k])
    json.dump(known, open(cache, "w", encoding="utf-8"), indent=1)
    return cuts


def _key_times(e, n):
    """Middle of the n most intense windows of an event/encounter, in time order."""
    ws = sorted(e["windows"], key=lambda w: -w["intensity"])[:n]
    return sorted((w["start"] + w["end"]) / 2 for w in ws)


def rate_encounters(video, encs, judge, cfg, hw):
    """Encounter-level 'hype' rating: is this fight worth a viewer's time (clutch / cinematic) or ordinary?"""
    W = int(cfg["judge"]["frame_width"]); H = int(round(W * 9 / 16 / 2)) * 2
    for e in encs:
        d = parse_json(judge.ask([(grab(video, _key_times(e, 8), W, H, hw), notes(cfg) + HYPE_PROMPT)], 40)[0]) or {}
        try:
            e["hype"] = max(1, min(10, int(float(d.get("hype", 5)))))
        except Exception:
            e["hype"] = 5
        e["hype_why"] = str(d.get("why", ""))[:100]
    return encs


def title_events(video, events, judge, cfg, hw):
    W = int(cfg["judge"]["frame_width"]); H = int(round(W * 9 / 16 / 2)) * 2
    for e in events:
        ts = _key_times(e, 4)
        t = judge.ask([(grab(video, ts, W, H, hw), notes(cfg) + TITLE_PROMPT)], max_new_tokens=16)[0]
        t = re.sub(r"[^\w '&-]", "", t.strip().splitlines()[0] if t.strip() else "").strip()
        e["title"] = " ".join(t.split()[:5]).title() or e["title"]
    return events


LOCATE_PROMPT = (
    "Frame from an action video game. Find the fight or kill happening in this frame: the enemy being shot, "
    "hit, stabbed, fed on or falling dead, together with whoever attacks it (ignore the player's own hands/weapon "
    'and the HUD). Answer JSON only: {"found": true/false, "bbox_2d": [x1, y1, x2, y2]} with coordinates 0-1000 '
    "relative to the image, the box tightly around the victim and attacker.")


def locate_kills(video, encs, judge, cfg, workdir, hw):
    """Owner: 'kill is not fully shown' - a kill far away is a few pixels on screen. Find WHERE each kill happens;
    if it is small and stays in one place, the planner punches in on it (smooth zoom, see planner._punch_pieces)."""
    pc = cfg["style"].get("punch_in", {})
    if not pc.get("enabled", True):
        return encs
    W = int(pc.get("frame_width", 960)); H = int(round(W * 9 / 16 / 2)) * 2
    cache = os.path.join(workdir, "judge_locate.json")
    known = json.load(open(cache, encoding="utf-8")) if os.path.exists(cache) else {}
    offs = (-1.2, -0.4, 0.4, 1.2)
    todo = sorted({round(t + o, 2) for e in encs for t in e["impacts"] for o in offs} -
                  {float(k.split("|")[1]) for k in known if k.startswith(JUDGE_VERSION + "|")})
    bs = int(cfg["judge"]["batch"])
    for i in range(0, len(todo), bs):
        chunk = todo[i:i + bs]
        ims = grab(video, chunk, W, H, hw)
        if len(ims) != len(chunk):
            continue
        for t, r in zip(chunk, judge.ask([([im], notes(cfg) + LOCATE_PROMPT) for im in ims], 60)):
            d = parse_json(r) or {}
            b = d.get("bbox_2d")
            ok = str(d.get("found", "")).lower() == "true" and isinstance(b, list) and len(b) == 4
            known[f"{JUDGE_VERSION}|{t}"] = [float(x) / 1000 for x in b] if ok else None
    json.dump(known, open(cache, "w", encoding="utf-8"), indent=1)
    max_w, max_h = float(pc.get("small_width", 0.2)), float(pc.get("small_height", 0.35))
    n = 0
    for e in encs:
        e["punch"] = []
        for t in e["impacts"]:
            boxes = [known.get(f"{JUDGE_VERSION}|{round(t + o, 2)}") for o in offs]
            boxes = [b for b in boxes if b and b[2] > b[0] and b[3] > b[1]]
            if len(boxes) < 2:
                continue
            ws = [b[2] - b[0] for b in boxes]; hs = [b[3] - b[1] for b in boxes]
            cxs = [(b[0] + b[2]) / 2 for b in boxes]; cys = [(b[1] + b[3]) / 2 for b in boxes]
            if np.median(ws) > max_w or np.median(hs) > max_h:
                continue                                  # already big on screen - no zoom needed
            if max(cxs) - min(cxs) > 0.25 or max(cys) - min(cys) > 0.25:
                continue                                  # camera swinging around - a fixed zoom would miss it
            z = float(np.clip(float(pc.get("fill", 0.45)) / max(np.median(ws), 0.05), 1.3, float(pc.get("max_zoom", 1.6))))
            e["punch"].append(dict(t=t, cx=float(np.median(cxs)), cy=float(np.median(cys)), z=round(z, 2)))
            n += 1
    print(f"  kill locator: {sum(len(e['impacts']) for e in encs)} kills checked, {n} far away -> punch-in")
    return encs


MOVE_PROMPT = (
    "These 4 frames are 0.8 seconds apart, in time order, ending on the moment an enemy is killed or knocked "
    "down in a video game. {notes}Did the kill come straight out of a MOVE by the player - dropping or jumping "
    "from a height onto/at the enemy, gliding, dashing, sliding, vaulting, a jump-shot, or a vehicle - rather "
    "than standing still and shooting? Reply with JSON only: "
    '{{"movement_kill": true/false, "move": one of ["drop", "jump", "glide", "dash", "slide", "vault", "vehicle", '
    '"none"], "wow": integer 1-10 how impressive and shareable the move plus kill is (10 = insane, viral), '
    '"caption": at most 10 words}}')


def find_movement_kills(video, F, encs, judge, cfg, workdir, hw):
    """MOVEMENT-KILL short (owner approved 2026-11-11). A kill qualifies when the seconds before it show a burst of
    movement (fast camera / player motion, or a window the first pass labelled as parkour) and the AI confirms,
    on 4 frames from take-off to impact, that the kill came out of that move. Result: e["moves"]."""
    mc = cfg["style"].get("movement", {})
    if not mc.get("enabled", True):
        return encs
    fps = int(F["fps"]); mot = F["motion"]; look = float(mc.get("lookback", 3.0))
    hi = float(np.percentile(mot[mot > 0], 85)) if (mot > 0).any() else 1e9
    W = int(cfg["judge"]["frame_width"]); H = int(round(W * 9 / 16 / 2)) * 2
    cache = os.path.join(workdir, "judge_moves.json")
    known = json.load(open(cache, encoding="utf-8")) if os.path.exists(cache) else {}
    cands = []
    req = ((cfg.get("_game") or {}).get("kill_counter") or {}).get("require_for_shorts", False)
    for e in encs:
        e["moves"] = []
        for t in (e.get("elim_times", []) if req and cfg.get("_elims") is not None else e["impacts"]):
            a, b = max(0, int((t - look) * fps)), max(1, int((t - 0.2) * fps))
            burst = mot[a:b].max() if b > a else 0
            parkour_before = any(w["activity"] == "parkour" and w["end"] > t - look - 2 and w["start"] < t
                                 for w in e["windows"])
            if burst >= hi or parkour_before:
                cands.append((e, round(t, 2)))
    prompt = MOVE_PROMPT.format(notes=notes(cfg))
    todo = [t for _, t in cands if f"{JUDGE_VERSION}|{t}" not in known]
    bs = int(cfg["judge"]["batch"])
    for i in range(0, len(todo), bs):
        chunk = todo[i:i + bs]
        replies = judge.ask([(grab(video, [t - 2.4, t - 1.6, t - 0.8, t + 0.3], W, H, hw), prompt) for t in chunk], 60)
        for t, r in zip(chunk, replies):
            d = parse_json(r) or {}
            try:
                wow = int(float(d.get("wow", 0)))
            except Exception:
                wow = 0
            known[f"{JUDGE_VERSION}|{t}"] = dict(ok=str(d.get("movement_kill", "")).lower() == "true",
                                                 move=str(d.get("move", "none")), wow=wow,
                                                 caption=str(d.get("caption", ""))[:100])
    json.dump(known, open(cache, "w", encoding="utf-8"), indent=1)
    n = 0
    for e, t in cands:
        d = known.get(f"{JUDGE_VERSION}|{t}")
        if d and d["ok"] and d["move"] != "none" and d["wow"] >= int(mc.get("min_wow", 6)):
            e["moves"].append(dict(t=t, **d)); n += 1
    print(f"  movement kills: {len(cands)} kills right after a burst of movement checked, {n} confirmed")
    return encs


COUNTER_PROMPT = ('This is a corner of a game screen. Read the number right after the word "{label}" (the '
                  "player's own kill count). Reply with JSON only: {{\"count\": integer or null}}")
SPECTATE_PROMPT = ('This is a strip from the bottom of a game screen. Does it show the word "{word}" (the player '
                   'died and is watching a teammate)? Reply with JSON only: {{"spectating": true/false}}')


def _crop_iter(video, step, box, hw, zoom=None):
    """Stream crops one by one (a 2 h video read every second would not fit in RAM as a list).
    `box` is in 2560x1440 coordinates whatever the source resolution (BGMI was recorded at 1440p, the
    2026-11 stream is 1080p): the frame is scaled to 2560x1440 first."""
    from PIL import Image
    x, y, w, h = [int(v) for v in box]
    z = zoom or (3 if w < 400 else 2)
    W2, H2 = w * z, h * z
    cmd = ["ffmpeg", "-v", "error"] + (["-hwaccel", "cuda"] if hw else []) + ["-i", video, "-an", "-vf",
           f"fps={1 / step},scale=2560:1440,crop={w}:{h}:{x}:{y},scale={W2}:{H2}:flags=lanczos",
           "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    while True:
        buf = p.stdout.read(W2 * H2 * 3)
        if len(buf) < W2 * H2 * 3:
            break
        yield Image.fromarray(np.frombuffer(buf, np.uint8).reshape(H2, W2, 3))
    p.wait()


def _crops(video, step, box, hw):
    return list(_crop_iter(video, step, box, hw))


def _ask_stream(video, step, box, hw, prompt, tokens, zoom=None):
    """Ask the judge about every crop, 32 at a time, without holding the whole video in memory."""
    out, buf = [], []
    for im in _crop_iter(video, step, box, hw, zoom):
        buf.append(im)
        if len(buf) == 32:
            out += _SMALL["judge"].ask([([b], prompt) for b in buf], tokens); buf = []
    if buf:
        out += _SMALL["judge"].ask([([b], prompt) for b in buf], tokens)
    return out


def read_kill_counter(video, F, cfg, workdir, hw):
    """Games with an on-screen kill count (BGMI: 'Eliminations N'): read it every `step` s across the whole video.
    Every rise of the number is a CONFIRMED elimination - ground truth the window labels cannot give (2026-11-11:
    two 'movement kills' were bat flights; the counter never moved).
    The same pass reads the 'Spectating' label: after the player dies the HUD shows the TEAMMATE's count (2026-11-11:
    it jumped 2 -> 5 and the teammate's kills were credited) and the footage is someone else's game. Spectating
    stretches (plus the death fade before them) are returned so the editor never uses them.
    Returns (elimination_times, spectate_spans) or (None, []) when the game has no counter."""
    kc = (cfg.get("_game") or {}).get("kill_counter")
    if not kc:
        return None, []
    step = float(kc.get("step", 2.0))
    sbox = kc.get("spectate_crop")
    cache = os.path.join(workdir, "judge_counter.json")
    key = f"{JUDGE_VERSION}|{step}|{kc.get('crop')}|{sbox}"
    c = json.load(open(cache, encoding="utf-8")) if os.path.exists(cache) else {}
    if c.get("key") == key:
        reads, spec = c["reads"], c["spec"]
    else:
        reads = []
        if kc.get("crop"):         # CS2 has no own-kill counter: only the spectate check runs (kills = kill feed)
            prompt = COUNTER_PROMPT.format(label=kc.get("label", "Eliminations"))
            for r in _ask_stream(video, step, kc["crop"], hw, prompt, 16):
                try:
                    reads.append(int((parse_json(r) or {}).get("count")))
                except Exception:
                    reads.append(None)
        spec = [False] * len(reads)
        if sbox:
            sp = kc.get("spectate_prompt") or SPECTATE_PROMPT.format(word=kc.get("spectate_word", "Spectating"))
            spec = [str((parse_json(r) or {}).get("spectating", "")).lower() == "true"
                    for r in _ask_stream(video, step, sbox, hw, sp, 16)]
            if reads:
                spec = (spec + [False] * len(reads))[:len(reads)]
            else:
                reads = [None] * len(spec)
        json.dump(dict(key=key, reads=reads, spec=spec), open(cache, "w", encoding="utf-8"))
    # spectating spans: 2 agreeing readings in a row; reach back over the death fade before it.
    # A single missed reading between two spectating ones is a misread, not a respawn (CS2: S.S.S patterns)
    spec = [s or (0 < i < len(spec) - 1 and spec[i - 1] and spec[i + 1]) for i, s in enumerate(spec)]
    spans, i = [], 0
    while i < len(spec):
        if spec[i] and (i + 1 < len(spec) and spec[i + 1] or i > 0 and spec[i - 1]):
            j = i
            while j + 1 < len(spec) and (spec[j + 1] or j + 2 < len(spec) and spec[j + 2]):
                j += 1
            spans.append((max(0.0, i * step - float(kc.get("death_fade", 5.0))), (j + 1) * step))
            i = j + 1
        else:
            i += 1
    spect = lambda t: any(a <= t <= b for a, b in spans)
    # counter: a reading counts only if the next one agrees (kills a single misread); a drop = new match / reset;
    # readings while spectating are the teammate's and are skipped, and the next own reading starts fresh
    elims, prev = [], None
    for i in range(len(reads) - 1):
        r, nxt = reads[i], reads[i + 1]
        if spect(i * step):
            prev = None
            continue
        if r is None or nxt != r:
            continue
        if prev is not None and prev < r <= prev + 3:
            elims += [i * step] * (r - prev)
        prev = r
    print(f"  kill counter: {sum(x is not None for x in reads)}/{len(reads)} readings, {len(elims)} confirmed "
          f"eliminations, {len(spans)} spectating stretch(es) removed ({sum(b - a for a, b in spans):.0f} s)")
    return sorted(elims), spans


FEED_PROMPT = (
    "This is a crop of a mobile battle-royale game screen. A kill message line looks like: a small number box, "
    "a player name, a weapon icon, a small action icon, another player name (sometimes followed by a number). "
    "Is such a line visible? Reply with JSON only: "
    '{"line": true or false, "killer": the FIRST name exactly as written or "", "victim": the SECOND name exactly as '
    'written or ""}'
)


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def read_killfeed(video, spans, F, elims, cfg, workdir, hw):
    """Knock-downs AND finishes from the on-screen kill feed (owner review 2026-11-11: knocks were not counted, so
    most real fights looked like '1 kill' and were dropped). The feed shows the whole squad's kills, so the player's
    own name is LEARNED per recording: the killer name whose lines match the HUD 'Eliminations' counter rises
    (games.<g>.player_names in config, if given, is used as well). Returns sorted event times."""
    kf = (cfg.get("_game") or {}).get("kill_feed")
    if not kf:
        return []
    box = kf["crop"]; step = float(kf.get("step", 1.0))
    cache = os.path.join(workdir, "judge_feed2.json")
    known = json.load(open(cache, encoding="utf-8")) if os.path.exists(cache) else {}
    secs = sorted({round(s * step, 2) for a, b in spans for s in range(int(np.floor(a / step)), int(np.ceil(b / step)) + 1)})
    todo = [s for s in secs if f"{JUDGE_VERSION}|{box}|{s}" not in known]
    from PIL import Image
    x, y, w, h = [int(v) for v in box]
    W0, H0 = 2560, 1440
    for i in range(0, len(todo), 32):
        chunk = todo[i:i + 32]
        ims = []
        for s in chunk:
            fr = grab(video, [s], W0, H0, hw)
            ims.append(fr[0].crop((x, y, x + w, y + h)) if fr else Image.new("RGB", (w, h)))
        for s, r in zip(chunk, _SMALL["judge"].ask([([im], FEED_PROMPT) for im in ims], 60)):   # 16 tokens cut the JSON
            d = parse_json(r) or {}
            known[f"{JUDGE_VERSION}|{box}|{s}"] = [_norm(d.get("killer", "")), _norm(d.get("victim", ""))] \
                if str(d.get("line", "")).lower() == "true" else None
        json.dump(known, open(cache, "w", encoding="utf-8"))
    lines = [(s, known[f"{JUDGE_VERSION}|{box}|{s}"]) for s in secs if known.get(f"{JUDGE_VERSION}|{box}|{s}")]
    # events: a (killer, victim) pair that was not on screen in the previous 6 s
    ev, last = [], {}
    for s, (k, v) in lines:
        if not k or not v:
            continue
        if s - last.get((k, v), -99) > 6:
            ev.append((s, k, v))
        last[(k, v)] = s
    # learn the player's name(s): killers whose events land within 4 s before an Eliminations rise
    score, total = {}, {}
    for s, k, v in ev:
        total[k] = total.get(k, 0) + 1
        if any(-1 <= e - s <= 5 for e in elims):
            score[k] = score.get(k, 0) + 1
    names = set()
    if score:
        top = max(score.values())
        names |= {k for k, c in score.items() if c >= max(1, top // 2)}
    # a name from config is trusted only if ITS kill lines match the counter (e.g. an ENEMY whose name matches yours
    # was in the lobby and his kills were counted as the player's)
    for n in {_norm(n) for n in (cfg.get("_game") or {}).get("player_names", []) if n}:
        hits = sum(c for k, c in score.items() if n in k or k in n)
        allk = sum(c for k, c in total.items() if n in k or k in n)
        if allk and hits >= 0.5 * allk:
            names.add(n)
    own, seen = [], {}
    for s, k, v in ev:                 # knock + finish of the SAME enemy = one kill (titles must never over-claim)
        if any(n and (n in k or k in n) and min(len(n), len(k)) >= 4 for n in names):
            if s - seen.get(v, -99) > 45:
                own.append(s)
            seen[v] = s
    print(f"  kill feed: {len(secs)} s read, {len(ev)} kill messages, player name(s) {sorted(names)}, "
          f"{len(own)} own knocks/finishes")
    return sorted(own)


OWN_FEED_PROMPT = (
    "This is the kill feed from the top-right corner of a {game} screen. Every line reads: the killer name "
    '(sometimes "killer + assister"), a weapon icon, sometimes small headshot / wallbang / smoke icons, then the '
    'victim name. Find every line whose KILLER (the first name, before any "+") is "{name}". Reply with JSON only: '
    '{{"victims": [the victim name of each such line]}} - an empty list if there is none.'
)


def read_own_kills(video, cfg, workdir, hw, spect=()):
    """Own kills straight from the kill feed, for games where the player's name is known and there is no kill
    counter (e.g. a CS2 player name). Read every `step` s over the whole video (streamed, the
    feed keeps a line on screen ~5 s). A kill = a victim listed under the player's name in >= 2 readings within
    4 s (one misread is not a kill); time = the first reading. Kills while spectating are dropped."""
    kf = (cfg.get("_game") or {}).get("kill_feed") or {}
    names = [n for n in (cfg.get("_game") or {}).get("player_names", []) if n]
    if not kf or not names:
        return []
    import difflib
    step = float(kf.get("step", 1.0)); box = kf["crop"]
    cache = os.path.join(workdir, "judge_ownfeed.json")
    key = f"{JUDGE_VERSION}|{box}|{step}|{names}"
    c = json.load(open(cache, encoding="utf-8")) if os.path.exists(cache) else {}
    if c.get("key") == key:
        reads = c["reads"]
    else:
        prompt = OWN_FEED_PROMPT.format(game=kf.get("game_name", "game"), name=names[0])
        reads = []
        for r in _ask_stream(video, step, box, hw, prompt, 60, zoom=int(kf.get("zoom", 1))):
            v = (parse_json(r) or {}).get("victims") or []
            reads.append(sorted({_norm(x) for x in v if isinstance(x, str) and len(_norm(x)) >= 2} - {_norm(n) for n in names}))
        json.dump(dict(key=key, reads=reads), open(cache, "w", encoding="utf-8"))
    same = lambda a, b: a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.75
    gap = float(kf.get("same_victim_s", 20))
    win = max(1, int(round(4 / step)))
    own, last = [], []                         # last: [(victim, time)]
    for i, vs in enumerate(reads):
        t = i * step
        if any(a <= t <= b for a, b in spect):
            continue
        for v in vs:
            if any(same(v, u) and t - s <= gap for u, s in last):
                last = [(u, s) if not same(v, u) else (u, t) for u, s in last]
                continue
            if not any(any(same(v, u) for u in reads[j]) for j in range(i + 1, min(len(reads), i + 1 + win))):
                continue                      # seen once only = misread
            own.append(t); last.append((v, t))
    print(f"  kill feed: {len(reads)} readings, {sum(bool(r) for r in reads)} with a line by {names[0]}, "
          f"{len(own)} own kills")
    return sorted(own)


AMMO_PROMPT = ('This is a crop of the bottom HUD of a shooter game. Read the big number (bullets left in the '
               'magazine). Reply with JSON only: {"ammo": integer or null if there is no number}')


def read_ammo_shots(video, spans, cfg, workdir, hw):
    """Own shots from the AMMO COUNTER (CS2): the magazine number drops on every shot the player fires; a knife,
    the bomb or a grenade shows no number. Measured, not guessed (2026-11-12: the per-second AI review called
    planting the bomb and walking with the knife 'aiming' / 'shooting'). Read twice a second inside `spans`.
    Returns shot times (s)."""
    box = ((cfg.get("_game") or {}).get("kill_feed") or {}).get("ammo_crop")
    if not box:
        return None
    step = 0.5
    cache = os.path.join(workdir, "judge_ammo.json")
    known = json.load(open(cache, encoding="utf-8")) if os.path.exists(cache) else {}
    ts = sorted({round(i * step, 1) for a, b in spans for i in range(int(a / step), int(b / step) + 2)})
    key = lambda t: f"{JUDGE_VERSION}|{box}|{t}"
    todo = [t for t in ts if key(t) not in known]
    x, y, w, h = [int(v) for v in box]
    for i in range(0, len(todo), 32):
        chunk = todo[i:i + 32]
        ims = []
        for t in chunk:
            fr = grab(video, [t], 2560, 1440, hw)
            from PIL import Image
            im = fr[0].crop((x, y, x + w, y + h)) if fr else Image.new("RGB", (w, h))
            ims.append(im.resize((w * 3, h * 3)))
        for t, r in zip(chunk, _SMALL["judge"].ask([([im], AMMO_PROMPT) for im in ims], 16)):
            try:
                known[key(t)] = int((parse_json(r) or {}).get("ammo"))
            except Exception:
                known[key(t)] = None
        json.dump(known, open(cache, "w", encoding="utf-8"))
    shots = []
    for t0, t1 in zip(ts, ts[1:]):
        a, b = known.get(key(t0)), known.get(key(t1))
        if t1 - t0 <= step + 0.01 and a is not None and b is not None and 0 < a - b <= 12:
            shots += [t0 + step * (j + 1) / (a - b + 1) for j in range(a - b)]   # spread the shots fired in between
    print(f"  ammo counter: {len(ts)} readings, {len(shots)} own shots (magazine drops)")
    return shots


LOOT_PROMPT = (
    "Two crops of a mobile battle-royale game screen stacked on top of each other. Does the image show the text "
    "'Picked up' followed by an item name, OR a 'Nearby:' loot list with item boxes? Reply with JSON only: "
    '{"loot": true or false}'
)


def read_loot(video, spans, cfg, workdir, hw):
    """Seconds where the player is LOOTING (owner review: 'cut looting'). Pickup clicks are loud and were read as own
    gunfire, so a 'fight' grew 20 s of looting in front of the real fight. Reads the 'Picked up X' line and the
    'Nearby:' loot list once per second. Returns a set of whole seconds."""
    lc = (cfg.get("_game") or {}).get("loot")
    if not lc:
        return set()
    p, q = lc["picked_crop"], lc["panel_crop"]
    cache = os.path.join(workdir, "judge_loot.json")
    known = json.load(open(cache, encoding="utf-8")) if os.path.exists(cache) else {}
    secs = sorted({s for a, b in spans for s in range(int(np.floor(a)), int(np.ceil(b)) + 1)})
    key = lambda s: f"{JUDGE_VERSION}|{p}|{q}|{s}"
    todo = [s for s in secs if key(s) not in known]
    from PIL import Image
    for i in range(0, len(todo), 32):
        chunk = todo[i:i + 32]
        ims = []
        for s in chunk:
            fr = grab(video, [s], 2560, 1440, hw)
            W = max(p[2], q[2])
            im = Image.new("RGB", (W, p[3] + q[3]))
            if fr:
                im.paste(fr[0].crop((p[0], p[1], p[0] + p[2], p[1] + p[3])), (0, 0))
                im.paste(fr[0].crop((q[0], q[1], q[0] + q[2], q[1] + q[3])), (0, p[3]))
            ims.append(im)
        for s, r in zip(chunk, _SMALL["judge"].ask([([im], LOOT_PROMPT) for im in ims], 20)):
            d = parse_json(r) or {}
            known[key(s)] = str(d.get("loot", "")).lower() == "true"
        json.dump(known, open(cache, "w", encoding="utf-8"))
    loot = {s for s in secs if known.get(key(s))}
    print(f"  looting: {len(loot)} of {len(secs)} s around own shots are looting (pickup sounds ignored)")
    return loot


_SMALL = {}


def judge_ask_small(ims, prompt):
    return _SMALL["judge"].ask([([im], prompt) for im in ims], 16)


REVIEW_PROMPT = (
    "These 3 frames are 0.4 seconds apart from a {view} video game. {notes}What is the PLAYER doing in this exact "
    "second? Reply with JSON only: "
    '{{"state": one of ["shooting", "aiming", "kill", "player_down", "dead_time"]}}. '
    "shooting = firing at an enemy (muzzle flash, tracers, hit markers, damage numbers, enemy reacting) or throwing "
    "a grenade at enemies; aiming = scoped in or aiming at a visible enemy, peeking to shoot; kill = an enemy is "
    "knocked down or eliminated in these frames (kill banner / kill feed); player_down = the PLAYER'S OWN character "
    "is knocked down (crawling, red or grey screen, revive / bleed-out bar), dying, dead or spectating; "
    "dead_time = anything else: running, walking, looting, healing, hiding, waiting, looking around, driving, menus."
)


def review_seconds(video, spans, judge, cfg, workdir, hw):
    """SELF-REVIEW (owner 2026-11-11: 'figure this out by yourself'): every second of every planned clip is
    judged on 3 frames; killedit.tighten keeps only shooting / aiming / kill seconds and stops before the player's
    own knock-down. Returns {second: state}."""
    W = int(cfg.get("kill_edit", {}).get("review_width", 960)); H = int(round(W * 9 / 16 / 2)) * 2
    cache = os.path.join(workdir, "judge_review.json")
    known = json.load(open(cache, encoding="utf-8")) if os.path.exists(cache) else {}
    secs = sorted({sec for a, b, *_ in spans for sec in range(int(np.floor(a)), int(np.ceil(b)))})
    todo = [x for x in secs if f"{JUDGE_VERSION}|{x}" not in known]
    prompt = REVIEW_PROMPT.format(view=(cfg.get("_game") or {}).get("view", "first-person or third-person"),
                                  notes=notes(cfg))
    bs = int(cfg["judge"]["batch"]) * 2
    for i in range(0, len(todo), bs):
        chunk = todo[i:i + bs]
        items = []
        for x in chunk:
            ims = grab(video, [x + 0.1, x + 0.5, x + 0.9], W, H, hw)
            items.append((ims, prompt))
        for x, r in zip(chunk, judge.ask(items, 20)):
            st = str((parse_json(r) or {}).get("state", "dead_time")).strip().lower()
            known[f"{JUDGE_VERSION}|{x}"] = st if st in ("shooting", "aiming", "kill", "player_down", "dead_time") else "dead_time"
        json.dump(known, open(cache, "w", encoding="utf-8"))
    out = {x: known[f"{JUDGE_VERSION}|{x}"] for x in secs}
    import collections
    print(f"  self-review: {len(secs)} seconds checked: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(out.values()).most_common()))
    return out


CLIP_PROMPT = (
    "These frames are from ONE short kill clip of a {view} video game, in time order. {notes}Rate how impressive "
    "it is for a highlight video people will share: 9-10 = multi-kill, clutch against several enemies, long-range "
    "scoped kill, insane reaction; 6-8 = clean solid kill; 1-5 = ordinary or hard to see. Reply with JSON only: "
    '{{"wow": integer 1-10}}'
)


def rate_clips(video, clips, judge, cfg, hw):
    W = int(cfg["judge"]["frame_width"]); H = int(round(W * 9 / 16 / 2)) * 2
    prompt = CLIP_PROMPT.format(view=(cfg.get("_game") or {}).get("view", ""), notes=notes(cfg))
    for c in clips:
        ts = []
        for x, y in c["pieces"]:
            ts += list(np.linspace(x + 0.2, y - 0.2, 3))
        ts = sorted(ts)[:: max(1, len(ts) // 6)][:6]
        try:
            c["wow"] = max(1, min(10, int(float((parse_json(judge.ask([(grab(video, ts, W, H, hw), prompt)], 20)[0]) or {}).get("wow", 5)))))
        except Exception:
            c["wow"] = 5
    return clips

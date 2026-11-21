"""Upload text for every rendered video - no API, no internet.

Writes <video>_UPLOAD.txt next to each montage / short: 3 title options, a description, hashtags and a tag list.
Everything is built from FACTS the edit measured (kill count, how fast the kills came), so a title never promises
something the clip does not show (YouTube punishes misleading titles with low retention).
Kill numbers are used only when they are ground truth (HUD counter / kill feed: `edit_mode: fights`). Story / action
games are titled from the AI's encounter name instead (no invented kill counts).
Never claims "no commentary" - streams have the player's voice.
"""
import os
import random
import re

GAMES = {
    "bgmi": dict(name="BGMI", alt="PUBG Mobile", big=(4, "Squad Wipe"),
                 tags=["bgmi", "bgmi montage", "bgmi highlights", "pubg mobile", "pubg mobile montage",
                       "bgmi gameplay", "bgmi best moments", "bgmi india",
                       "battlegrounds mobile india", "pubg mobile highlights", "bgmi new update", "bgmi short",
                       "bgmi kills"],
                 hashtags=["#bgmi", "#pubgmobile", "#bgmimontage", "#battlegroundsmobileindia", "#gaming"]),
    "cs2": dict(name="CS2", alt="Counter-Strike 2", big=(5, "ACE"),
                tags=["cs2", "counter strike 2", "cs2 highlights", "cs2 montage", "cs2 gameplay", "cs2 frags",
                      "cs2 clips", "cs2 premier", "cs2 best moments", "counter strike 2 highlights", "cs2 shorts",
                      "cs2 kills", "csgo"],
                hashtags=["#cs2", "#counterstrike2", "#cs2highlights", "#counterstrike", "#gaming"]),
    "phantom_blade_zero": dict(name="Phantom Blade Zero", alt="Phantom Blade 0", big=None,
                               tags=["phantom blade zero", "phantom blade 0", "phantom blade zero gameplay",
                                     "phantom blade zero boss fight", "phantom blade zero combat",
                                     "phantom blade zero walkthrough", "phantom blade zero highlights",
                                     "kung fu punk", "wuxia game", "soulslike", "action rpg", "s-game"],
                               hashtags=["#phantombladezero", "#phantomblade0", "#soulslike", "#gaming", "#bossfight"]),
    "gta6": dict(name="GTA 6", alt="Grand Theft Auto VI", big=None,
                 # order = vidIQ keyword research 2026-11-16 (volume / competition): exact phrases people type
                 tags=["gta 6 gameplay", "gta 6 gameplay walkthrough part 1", "gta 6 walkthrough part 1", "gta 6 story",
                       "gta vi", "grand theft auto 6", "gta 6 jason and lucia", "gta 6 walkthrough", "gta 6 ps5",
                       "gta 6 vice city", "gta 6 reaction", "gta 6 funny moments", "gta 6 first time"],
                 hashtags=["#gta6", "#gtavi", "#grandtheftauto6", "#gaming", "#gta"]),
    "gta5": dict(name="GTA 5", alt="Grand Theft Auto V", big=None,
                 tags=["gta 5", "gta v", "grand theft auto 5", "gta 5 story", "gta 5 walkthrough", "gta 5 funny moments",
                       "gta 5 trevor", "gta 5 franklin", "gta 5 lamar", "gta 5 michael", "gta 5 reaction",
                       "gta 5 story mode", "gta 5 gameplay"],
                 hashtags=["#gta5", "#gtav", "#grandtheftauto5", "#gaming", "#gta"]),
    "vtm_bloodlines2": dict(name="Bloodlines 2", alt="Vampire The Masquerade", big=None,
                            tags=["bloodlines 2", "vampire the masquerade bloodlines 2", "bloodlines 2 gameplay",
                                  "bloodlines 2 walkthrough", "bloodlines 2 story", "vampire game", "vtm bloodlines 2"],
                            hashtags=["#bloodlines2", "#vampirethemasquerade", "#gaming", "#vtm"]),
}
DEFAULT = dict(name="Gameplay", alt="", big=None, tags=["gameplay", "gaming", "highlights", "montage"],
               hashtags=["#gaming", "#gameplay"])


def _channel(cfg):
    """Channel name for titles / descriptions - OWNER RULE 2026-11-13: only when the intro card is explicitly
    enabled AND named (never add a name on our own)."""
    ic = cfg["style"].get("intro") or {}
    return str(ic.get("name") or "").strip() if ic.get("enabled") else ""


def _game(cfg):
    key = (cfg.get("_game") or {}).get("name") or cfg.get("_game_key") or ""
    return GAMES.get(key, dict(DEFAULT, name=key.replace("_", " ").title() or "Gameplay"))


def _verified(cfg):
    return cfg.get("_fights") is not None or cfg.get("_kclips") is not None


def _clean(title):
    """AI encounter names over-claim ('Squad Wipe' with 1 kill, 'Clutch' with a team alive): keep them neutral."""
    t = re.sub(r"\b(squad wipe|clutch|killstreak|kill storm)\b", "Fight", str(title), flags=re.I)
    t = re.sub(r"\b(Fight)(\s+Fight)+\b", r"\1", t)
    return t.strip() or "Fight"


def _speed_phrase(k, span):
    if k >= 2 and span <= 12:
        return f"{k} Kills in {max(2, int(round(span)))} Seconds"
    if k >= 4:
        return f"{k} Kill Fight"
    return f"{k} Kills"


def _tags(g, extra):
    out, n = [], 0
    for t in extra + g["tags"]:
        if t.lower() not in out and n + len(t) + 1 <= 480:
            out.append(t.lower()); n += len(t) + 1
    return ", ".join(out)


def _alt(g):
    return f" ({g['alt']})" if g.get("alt") else ""


STORY_HOOKS = [   # owner 2026-11-18: "write something catchy that keeps the viewer watching". Each is an open loop or
    # a bold claim, picked from what the fight really shows (never a kill number the HUD did not confirm)
    (lambda k, st, a: st >= 2, "THEY NEVER SAW ME COMING"),
    (lambda k, st, a: bool(re.search(r"juggl|launch|\bair\b", a)), "NOBODY TOUCHES THE GROUND"),
    (lambda k, st, a: "fire" in a or "fiery" in a or "explos" in a, "WATCH THEM BURN"),
    (lambda k, st, a: k >= 8, "ONE VAMPIRE VS THE WHOLE GANG"),
    (lambda k, st, a: k >= 5, "THEY PICKED THE WRONG VAMPIRE"),
    (lambda k, st, a: "boss" in a or "giant" in a, "THIS ONE FOUGHT BACK"),
    (lambda k, st, a: True, "WAIT FOR THE LAST KILL"),
    (lambda k, st, a: True, "THIS GOT BRUTAL FAST"),
    (lambda k, st, a: True, "NO MERCY TONIGHT"),
    (lambda k, st, a: True, "DONT BLINK"),
]


def _story_hook(sp, cfg):
    used = cfg.setdefault("_hooks_used", set())
    k, st, a = int(sp.get("kills_est", 0)), int(sp.get("n_stealth", 0)), sp.get("about", "").lower()
    for ok, h in STORY_HOOKS:
        if h not in used and ok(k, st, a):
            used.add(h); return h
    return "WAIT FOR THE LAST KILL"


def short_hook(sp, cfg):
    """(hook, big_word) for a short - the SAME text is burned into the short's headline band and used in the
    title, so the video always delivers what the title says. Kill numbers only when they are measured."""
    g = _game(cfg)
    if "kills_est" in sp and not _verified(cfg):          # story game: a hook that keeps people watching
        return _story_hook(sp, cfg), None
    if "kills" not in sp or not _verified(cfg):          # story / action game: no measured kill count
        hook = _clean(sp.get("title", "Fight"))
        if hook.lower() in ("fight", "movement kills"):
            hook = "Insane Combat" if hook.lower() == "fight" else "Movement Kills"
        return hook, None
    k, span = int(sp.get("kills", 0)), float(sp.get("kill_span", 99))
    hook = _speed_phrase(k, span)
    bn, bs = sp.get("best_burst") or (0, 99)
    if span > 12 and bn >= 2 and bs <= 12:          # several fights: lead with the fastest burst (still true)
        hook = _speed_phrase(int(bn), float(bs))
    big = g.get("big")
    word = big[1] if big and max(bn, k if span <= 30 else 0) >= big[0] else None
    return hook, ((big[0], word) if word else None)


def _fit(t, n=70):
    """Titles: keyword first, <= ~60 visible chars (YouTube truncates on mobile; playbook section 4)."""
    return t if len(t) <= n else t[: n - 1].rstrip(" -|") + "…"


def short_text(name, sp, cfg, seed):
    g = _game(cfg); ch = _channel(cfg)
    r = random.Random(seed)
    hook, big = (sp["headline"], sp.get("big_word")) if sp.get("headline") else short_hook(sp, cfg)
    if "kills_est" in sp and not _verified(cfg):          # title = the same hook the short shows on screen
        hook = hook.title()
    if big is None and ("kills" not in sp or not _verified(cfg)):
        titles = [
            f"{g['name']}: {hook} 🔥 #shorts",
            f"{hook} in {g['name']} Was Insane 😳 #shorts",
            f"{g['name']} {hook} | {g['alt'] or 'Gameplay'} #shorts",
        ]
        desc = (f"{hook} in {g['name']}{_alt(g)}{' - ' + ch if ch else ''}.\n\n"
                f"Full fights and boss battles on the channel. Subscribe for more {g['name']}.\n\n"
                + " ".join(["#shorts"] + g["hashtags"][:3]))
        tags = _tags(g, [f"{g['name']} {hook}", f"{g['name']} shorts", "shorts"])
    else:
        k = int(sp.get("kills", 0))
        titles = [
            f"{g['name']}: {hook} 🔥 #shorts",
            f"{g['name']} {big[1] if big else 'Fight'} - {hook} #shorts",
            f"{hook} 😳 {g['name']} {g['alt']} #shorts" if g["alt"] else f"{hook} 😳 {g['name']} #shorts",
        ]
        desc = (f"{hook} in {g['name']} - the full fight, no cuts in the action{' - ' + ch if ch else ''}.\n\n"
                f"Watch the full montage on the channel. Subscribe for more {g['name']} highlights.\n\n"
                + " ".join(["#shorts"] + g["hashtags"][:3]))
        tags = _tags(g, [f"{g['name']} {k} kills", f"{g['name']} shorts", "shorts"])
    r.shuffle(titles)
    return [_fit(t) for t in titles], desc, tags


def _kills_of(grp):
    m = re.search(r"(\d+)\s*kills?\b", grp.get("title", ""), re.I)
    return int(m.group(1)) if m else 0


def long_text(groups, cfg, chapters_path):
    g = _game(cfg); ch = _channel(cfg)
    fights = [x for x in groups if not x.get("cold")]
    best = next((x for x in groups if x.get("cold")), None)
    chap = open(chapters_path, encoding="utf-8").read().strip() if os.path.exists(chapters_path) else ""
    if chap.startswith("("):
        chap = ""
    pre = ch.upper() + " | " if ch else ""
    if _verified(cfg):
        bk = _kills_of(best) if best else 0       # the fight edit's opening is a fight NOT repeated later
        total = sum(_kills_of(x) for x in fights) + bk
        n = len(fights) + (1 if bk else 0)
        titles = [
            f"{pre}{g['name']} Montage - {total} Kills, Zero Mercy 🔥",
            f"Every Fight Ends The Same Way 😈 | {g['name']} Montage{' | ' + ch if ch else ''}",
            f"{g['name']} Highlights - {n} Fights, {total} Kills{' | ' + g['alt'] + ' Montage' if g['alt'] else ''}",
        ]
        what = f"{total} kills across {n} full fights"
    else:
        chap = "\n".join(_clean(l) if re.match(r"^\d", l) else l for l in chap.splitlines())
        n = len(fights)
        titles = [
            f"{pre}{g['name']} - {n} Brutal Fights & Boss Battles 🔥",
            f"{g['name']} Combat Is Insane 😳 | Best Fights{' | ' + ch if ch else ''}",
            f"{g['name']} Highlights - {n} Fights, No Filler{' | ' + g['alt'] if g['alt'] else ''}",
        ]
        what = f"{n} fights"
    # line 1 is what shows above "...more" (~100 chars): keyword + the measured promise, nothing else
    desc = (f"{g['name']} montage: {what}, no filler{' - ' + ch if ch else ''}.\n"
            f"Every fight is shown from the first hit to the last{' (' + g['alt'] + ')' if g['alt'] else ''}.\n\n"
            + (f"Chapters:\n{chap}\n\n" if chap else "")
            + f"Subscribe for more {g['name']} montages and Shorts.\n\n" + " ".join(g["hashtags"][:3]))
    return [_fit(t) for t in titles], desc, _tags(g, [f"{g['name']} montage {ch}".strip(), f"{ch} {g['name']}".strip()])


POLICY = {     # verified 2026-11-13 (sources in the playbook) - shown in every upload file for that game
    "gta6": ["Rockstar video policy (checked 2026-11-16): cutscenes only inside a play-through or narrated Let's Play; 'isolated videos of the game's cinematics will be removed'; no endings, no edited cutscene/dialogue compilations, no pre-release or leaked footage, no in-game TV shows on their own. Rockstar's policy covers NON-commercial use - monetized videos are at Take-Two's discretion, so keep your own voice and edit in every upload."],
    "gta5": ["Rockstar video policy (checked 2026-11-16): cutscenes only inside a play-through or narrated Let's Play; 'isolated videos of the game's cinematics will be removed'; no endings, no edited cutscene/dialogue compilations, no pre-release or leaked footage, no in-game TV shows on their own. Rockstar's policy covers NON-commercial use - monetized videos are at Take-Two's discretion, so keep your own voice and edit in every upload."],
    "cs2": ["Valve allows monetized CS2 gameplay videos, but never show case openings, skin trading or skin-gambling "
            "sites: YouTube age-restricts/strikes gambling content (rules in force since Nov 2025)."],
    "bgmi": ["No public Krafton rule on monetizing BGMI videos was found - Krafton runs the Krafton Creator Network "
             "(creators.krafton.com) if you want official creator status."],
    "phantom_blade_zero": ["No S-Game creator/video policy was found - keep story cutscenes short in uploads "
                           "(spoilers + 'extended gameplay' risk), and never upload whole cutscenes as their own videos."],
}


def checklist(cfg, short=False):
    key = (cfg.get("_game") or {}).get("name") or ""
    g = _game(cfg)
    lines = [
        "Upload as UNLISTED first; go Public / schedule only after the HD version and the copyright check finish.",
        f"Category: Gaming -> type the game title '{g['alt'] or g['name']}' (links the video to the game's page).",
        "Audience: No, it's not made for kids.   Altered or synthetic content: No (real gameplay, no AI people).",
    ]
    if short:
        lines.append("Shorts: pick the cover frame in the YouTube app; post 1-2 Shorts a day, spaced apart, not all at once.")
    else:
        lines += [
            "Thumbnail: use Test & Compare with Thumbnail_1/2/3.jpg (it ranks by watch-time share; small "
            "differences are noise - don't re-test or edit in the first 48 h).",
            "End screen: ONE big 'video' element for the last 5-10 s (one choice converts better than four).",
            "Chapters are already in the description - keep them; they name fights, they don't spoil them.",
        ]
    lines += POLICY.get(key, [])
    lines.append("Monetization: keep your voice commentary when the recording has it - YouTube says videos that "
                 "'simply show game play for extended periods' may not be monetizable; commentary tied to the action "
                 "and real editing are what make gameplay eligible.")
    return "UPLOAD CHECKLIST:\n" + "\n".join(f"  - {x}" for x in lines) + "\n"


def _write(path, titles, desc, tags, extra=""):
    with open(path, "w", encoding="utf-8") as f:
        f.write("TITLE (pick one - or put 3 into Test & Compare):\n"
                + "\n".join(f"  {i}. {t}" for i, t in enumerate(titles, 1)) + "\n\n")
        f.write("DESCRIPTION:\n" + desc + "\n\n")
        f.write("TAGS (paste into the Tags box - tags matter little, don't spend time here):\n" + tags + "\n\n")
        f.write(extra)


def thumb_words(groups, cfg):
    """<= 3 words per thumbnail (playbook: one subject, 3 words max, readable on a phone). Facts only."""
    g = _game(cfg)
    if _verified(cfg):
        fights = [x for x in groups if not x.get("cold")]
        best = next((x for x in groups if x.get("cold")), None)
        total = sum(_kills_of(x) for x in fights) + (_kills_of(best) if best else 0)
        bk = _kills_of(best) if best else 0
        return [f"{total} KILLS", f"{bk} KILL FIGHT" if bk >= 3 else "ZERO MERCY", f"{g['name'].upper()} MONTAGE"]
    boss = any(x.get("cat") == "boss" for x in groups)
    n = len([x for x in groups if not x.get("cold")])
    return ["BOSS FIGHTS" if boss else "INSANE FIGHTS", f"{n} FIGHTS", "NO MERCY"]


def write_all(long_out, groups, short_outs, shorts, cfg):
    try:
        if long_out and groups:
            d = os.path.dirname(long_out)
            _write(long_out[:-4] + "_UPLOAD.txt", *long_text(groups, cfg, os.path.join(d, "YouTube_Chapters.txt")),
                   extra=checklist(cfg))
        names = {n: sp for n, sp in shorts}
        for i, p in enumerate(short_outs):
            n = os.path.basename(p)[:-4]
            if n in names:
                _write(p[:-4] + "_UPLOAD.txt", *short_text(n, names[n], cfg, seed=hash(n) + i),
                       extra=_status_note(names[n]) + checklist(cfg, short=True))
    except Exception as e:                     # upload text must never cost the rendered videos their QC
        print(f"  WARNING: upload text not written: {e!r}")


# ------------------------------------------------------------------------------------------- cinematic mode
def _status_note(sp):
    st = sp.get("status")
    if st in ("REVIEW", "DO NOT POST"):
        return (f"POLICY STATUS: {st} - {sp.get('reason', '')}\n"
                "(Check this before posting. 3 copyright strikes delete a channel.)\n\n")
    return ""


def episode_thumb_words(groups, i, cfg):
    """<= 3 words: the strongest story moment of the part (true), the part number, the game."""
    g = _game(cfg)
    cuts = [x["cut"] for x in groups if x.get("cut")]
    best = max(cuts, key=lambda c: c.get("viral", 0)) if cuts else None
    w1 = " ".join(_clean(best.get("m_title", "")).upper().split()[:3]) if best and best.get("m_title") else "STORY MODE"
    return [w1, f"PART {i}", g["name"].upper()]


def write_episode(path, groups, i, cfg):
    """Upload text for one playthrough part. Titles use the strongest moment of the part WITHOUT spoiling it."""
    try:
        g = _game(cfg); ch = _channel(cfg)
        cuts = [x["cut"] for x in groups if x.get("cut")]
        safe = [c for c in cuts if c.get("spoiler") in ("none", "minor")]
        best = max(safe or cuts, key=lambda c: c.get("viral", 0)) if cuts else None
        hook = _clean(best["m_title"]) if best and best.get("m_title") and best in safe else "Story Mode"
        for nm in sorted({g["name"], g.get("alt") or "", "GTA VI", "GTA V", "GTA 6", "GTA 5"}, key=len, reverse=True):   # no "GTA 5 ... GTA 5"
            if nm:
                hook = re.sub(r"(?i)\b" + re.escape(nm) + r"\b[:\- ]*", "", hook).strip(" -:|")
        hook = hook or "Story Mode"
        titles = [
            f"{g['name']} Gameplay Walkthrough Part {i} - {hook}",       # exact search phrase first (vidIQ)
            f"{g['name']} Walkthrough Part {i}: {hook} 😳",
            f"{g['name']} Story Part {i} | {hook} | {g['alt'] or 'Gameplay'}",
        ]
        chap = path[:-4] + "_Chapters.txt"
        chap = open(chap, encoding="utf-8").read().strip() if os.path.exists(chap) else ""
        if chap.startswith("("):
            chap = ""
        spoil = any(c.get("spoiler") in ("major", "ending") for c in cuts)
        desc = (f"{g['name']} gameplay walkthrough part {i}: {hook}{' - ' + ch if ch else ''}.\n"
                f"The story with cutscenes kept in the playthrough{' - SPOILERS ahead' if spoil else ''}.\n\n"
                + (f"Chapters:\n{chap}\n\n" if chap else "")
                + f"Subscribe for the next part.\n\n" + " ".join(g["hashtags"][:3]))
        tags = _tags(g, [f"{g['name']} gameplay walkthrough part {i}", f"{g['name']} walkthrough part {i}",
                         f"{g['name']} story part {i}"])
        extra = ""
        if spoil:
            extra += ("SPOILER NOTE: this part contains a major story moment / the ending. Don't put it in the title or "
                      "thumbnail, and keep the cutscenes inside the playthrough (never cut them out on their own).\n\n")
        _write(path[:-4] + "_UPLOAD.txt", [_fit(t) for t in titles], desc, tags, extra=extra + checklist(cfg))
    except Exception as e:
        print(f"  WARNING: episode upload text not written: {e!r}")

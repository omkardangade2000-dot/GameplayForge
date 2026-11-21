"""Better shot search for breakdowns: visual queries written as what the camera SEES (not edit notes), plus a
caption index of the whole source (Qwen-VL describes one frame every `step` s) searched by keywords."""
import json, os, re

CAP_PROMPT = ("Describe this GTA 6 frame in under 20 words: who / what is visible, what is happening, any HUD or "
              "on-screen element (wanted stars, phone text, weapon wheel, prompts, minimap). Plain words only.")

QUERIES = {
    "S00": (["wanted level stars on the HUD during a police chase", "a car on fire next to a van"], ["wanted", "stars", "fire", "burning"]),
    "P01": (["a woman leaning out of a car window shooting at police", "a man driving a car during a police chase"], ["shoot", "car", "chase", "window"]),
    "P02": (["a man in the back seat of a car shooting backwards", "inside a car, a woman driving, a man shooting"], ["back seat", "rear", "inside car", "shooting"]),
    "P03": (["two people in a gunfight taking cover together", "shootout inside a building"], ["gunfight", "cover", "shootout"]),
    "P04": (["a phone text message on screen"], ["text", "message", "phone"]),
    "P05": (["a man clearing a building floor with a rifle", "a woman struggling with an attacker"], ["clearing", "floor", "struggle", "elevator"]),
    "P06": (["a man taking cover behind a crate during a shootout"], ["cover", "crate", "pillar"]),
    "P07": (["glass windows shattering from gunfire in an office"], ["glass", "shatter", "office", "monitor"]),
    "P08": (["slow motion aiming with red and yellow target markers on an enemy"], ["slow", "target", "red", "yellow", "marker"]),
    "P09": (["a button prompt on screen while a woman is grabbed by an attacker"], ["prompt", "grabbed", "struggle", "button"]),
    "P10": (["a circular weapon wheel menu on screen"], ["weapon wheel", "wheel", "menu", "items"]),
    "P11": (["six wanted stars in the corner of the screen"], ["six stars", "wanted", "stars"]),
    "P12": (["wanted level gauge with small icons of people and a car under it"], ["icons", "wanted", "gauge"]),
    "P13": (["a car burning after a molotov is thrown", "two people driving away in a plain white van"], ["molotov", "fire", "van", "burning"]),
    "P14": (["a police car flipping over in a crash"], ["police car", "flip", "crash"]),
    "P15": (["minimap flashing red and blue in the corner during a police chase"], ["minimap", "map"]),
    "P16": (["a man petting a dog in a hallway"], ["dog", "pet"]),
    "P17": (["a scuba diver underwater with fish and a manta ray"], ["underwater", "scuba", "diver", "fish", "manta"]),
    "P18": (["an airboat moving through a swamp"], ["airboat", "swamp", "boat"]),
    "P19": (["race cars on a race track", "dirt bikes racing", "street race at night with neon"], ["race", "track", "dirt bike"]),
    "P20": (["a man shooting a basketball at a hoop", "a backyard wrestling ring", "a television in an apartment"], ["basketball", "hoop", "wrestling", "television", "tv"]),
    "P21": (["a TV screen showing a commercial"], ["commercial", "tv", "television", "ad"]),
}


def caption_index(video, workdir, judge, step=3.0):
    from .judge import grab
    p = os.path.join(workdir, f"captions_{step}.json")
    caps = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    import subprocess
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                               capture_output=True, text=True).stdout)
    ts = [round(i * step, 1) for i in range(int(dur / step))]
    todo = [t for t in ts if str(t) not in caps]
    for i in range(0, len(todo), 8):
        ch = todo[i:i + 8]
        res = judge.ask([(grab(video, [t], 640, 360, True), CAP_PROMPT) for t in ch], 48)
        for t, r in zip(ch, res):
            caps[str(t)] = r.strip()[:200]
        if i % 80 == 0:
            json.dump(caps, open(p, "w", encoding="utf-8"), indent=0, ensure_ascii=False)
    json.dump(caps, open(p, "w", encoding="utf-8"), indent=0, ensure_ascii=False)
    return {float(k): v for k, v in caps.items()}


def candidates(sid, t_arr, e, text_emb_fn, caps, k=6, gap=6.0):
    import numpy as np
    qs, kws = QUERIES.get(sid, ([], []))
    cands = []
    def add(t):
        if all(abs(t - c) >= gap for c in cands):
            cands.append(float(t))
    if qs:
        for qe in text_emb_fn(qs):
            sim = e @ qe
            n = 0
            for i in np.argsort(-sim):
                before = len(cands); add(t_arr[i])
                if len(cands) > before:
                    n += 1
                if n >= k:
                    break
    scored = []
    for t, c in caps.items():
        cl = c.lower()
        s = sum(1 for w in kws if w in cl)
        if s:
            scored.append((s, t))
    for s, t in sorted(scored, reverse=True)[:k]:
        add(t)
    return cands[:18]

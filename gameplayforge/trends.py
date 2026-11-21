"""TREND RADAR - which moments of a game are going viral on YouTube right now. Reads public search results only
(titles, view counts, upload dates, channel sizes via yt-dlp metadata). It NEVER downloads anyone's video.

Built from the GTA 5 study (2026-11-16, 483 videos for 16 searches):
  * formats by median views: endings / funny-meme scenes 2.3M, reactions 836K, walkthrough parts 790K,
    'all cutscenes' movies 212K, pure cinematic edits 38K -> the player's VOICE over the story wins, raw cinematics lose
  * characters drive titles (Franklin, Trevor, Lamar; 'Lamar roasts Franklin' = 20 of the top 150 titles)
For a new game (GTA 6 at launch) old view totals mean nothing: the radar ranks by VELOCITY (views per day since
upload) and by OUTLIER ratio (views / channel subscribers), then turns the winning titles into MOMENT KEYS
(character names, quoted lines, mission words) that cinematic.py searches for in the player's own footage.

Usage:  python -m gameplayforge trends --game gta6         (writes <output_dir>/_Trends/<game>/<date>/)
"""
import collections, csv, datetime as dt, json, math, os, re, time

STOP = set("""the a an and or of to in on at for with from is are was be it this that you your my i me we our
they he she his her them its as by not no yes how why what when who all best top new vs full part episode ep
game gameplay video shorts short official trailer gta grand theft auto 4k hd pc ps5 xbox walkthrough mission
movie scene scenes cutscene cutscenes funny moments first time reaction react ending story explained real life
v vi 5 6 one two three after before ever just like get got make made can will do does did
look looks extended watch finally know everything need here play played playing plays things games launch
features feature reveal revealed missed crazy funniest funny downloading download rockstar netflix concept details
detail reacts reacting memes meme fails fail insane epic first trailer trailers every never really actually
blew mind minds secret secrets realistic leak leaked leaks news update days left hour hours minutes look
new old official exclusive impressions things thing want wanted went going gonna better best worst most more
compilation moments moment highlights clips clip shorts reels free time times day year years
about gets ready entire complete found being it's again live reveals iler lanzamiento tráiler guide ultra journey
hindi tamil telugu urdu bangla marathi spanish español español português deutsch français""".split())
# YouTube hype vocabulary above is dropped: it matches any dialogue and says nothing about WHICH scene people want.
# What is kept: character names (weighted x2), quoted lines (x2) and specific words (places, missions, objects).


def _fmt(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def bucket(title, dur):
    t = title.lower()
    if dur and dur <= 61: return "short"
    if any(k in t for k in ("all cutscene", "game movie", "full movie", "full game")): return "all_cutscenes_movie"
    if any(k in t for k in ("walkthrough", "let's play", "lets play", "playthrough", "part ")): return "lets_play"
    if "react" in t or "first time" in t: return "reaction"
    if any(k in t for k in ("explained", "lore", "theory", "story of", "hidden", "secret", "details")): return "analysis"
    if any(k in t for k in ("funny", "meme", "roast", "fails", "thug life")): return "funny_meme"
    if "ending" in t or "final mission" in t: return "ending"
    if "cinematic" in t or "editor" in t: return "cinematic_edit"
    return "scene_other"


def search(queries, per_query=40, detail_top=60, pause=1.5):
    import yt_dlp
    vids = {}
    with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": "in_playlist", "skip_download": True,
                           "no_warnings": True}) as y:
        for q in queries:
            try:
                r = y.extract_info(f"ytsearch{per_query}:{q}", download=False)
            except Exception as e:
                print(f"  search failed for '{q}': {e}"); continue
            for e in r.get("entries") or []:
                if not e.get("id") or not e.get("view_count"):
                    continue
                d = vids.setdefault(e["id"], dict(id=e["id"], title=e.get("title") or "", views=int(e["view_count"]),
                                                   duration=e.get("duration") or 0,
                                                   channel=e.get("channel") or e.get("uploader") or "", queries=[]))
                d["queries"].append(q)
            time.sleep(pause)
    # second pass (metadata only) for the biggest ones: upload date + channel size -> velocity and outlier ratio
    top = sorted(vids.values(), key=lambda v: -v["views"])[:detail_top]
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "no_warnings": True}) as y:
        for v in top:
            try:
                d = y.extract_info(f"https://www.youtube.com/watch?v={v['id']}", download=False, process=False)
                v["upload_date"] = d.get("upload_date")
                v["subs"] = d.get("channel_follower_count")
                v["likes"] = d.get("like_count")
                v["tags"] = (d.get("tags") or [])[:25]
            except Exception as e:
                v["error"] = str(e)[:80]
            time.sleep(pause)
    return list(vids.values())


def moment_keys(videos, gp, top=150):
    """Character names / quoted lines / recurring words from the best titles, weighted by views."""
    chars = [c.lower() for c in gp.get("characters", [])]
    keys = collections.Counter(); ev = collections.defaultdict(list)
    for v in sorted(videos, key=lambda v: -v["views"])[:top]:
        t = v["title"]; tl = t.lower(); w = math.log10(max(10, v["views"]))
        strong = set(c for c in chars if re.search(rf"\b{re.escape(c)}\b", tl))
        strong |= {q.strip().lower() for q in re.findall(r"[\"“]([^\"”]{4,60})[\"”]", t)}      # quoted lines
        words = {x for x in re.findall(r"[a-z][a-z'\-]{3,}", tl) if x not in STOP} - strong
        for k in strong:
            keys[k] += 2 * w; ev[k].append(v)
        for k in words:
            keys[k] += w; ev[k].append(v)
    out = []
    for k, s in keys.most_common(80):
        vs = ev[k]
        if len(vs) < 2 and k not in chars:
            continue
        out.append(dict(key=k, score=round(s, 1), videos=len(vs), best_views=max(x["views"] for x in vs),
                        example=max(vs, key=lambda x: x["views"])["title"][:90],
                        character=k in chars))
    return out


def run(game, cfg, out_root=None):
    gp = (cfg.get("games") or {}).get(game) or {}
    tq = gp.get("trend") or {}
    queries = tq.get("queries") or [f"{game} cutscene", f"{game} funny", f"{game} reaction", f"{game} ending"]
    print(f"Trend radar: {game} - {len(queries)} searches (titles and view counts only, nothing is downloaded)")
    vids = search(queries, int(tq.get("per_query", 40)), int(tq.get("detail_top", 60)))
    today = dt.date.today()
    for v in vids:
        v["format"] = bucket(v["title"], v["duration"])
        if v.get("upload_date"):
            age = max(1, (today - dt.datetime.strptime(v["upload_date"], "%Y%m%d").date()).days)
            v["age_days"] = age; v["per_day"] = int(v["views"] / age)
        if v.get("subs"):
            v["outlier"] = round(v["views"] / max(1000, v["subs"]), 2)
    keys = moment_keys(vids, gp)
    root = out_root or os.path.join(cfg["paths"]["output_dir"], "_Trends", game, today.isoformat())
    os.makedirs(root, exist_ok=True)
    cols = ["views", "per_day", "outlier", "age_days", "format", "duration", "channel", "subs", "title", "id"]
    with open(os.path.join(root, "Trends.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols + ["url"])
        for v in sorted(vids, key=lambda v: -(v.get("per_day") or 0) * 1000 - v["views"] / 1e6):
            w.writerow([v.get(c, "") for c in cols] + [f"https://youtu.be/{v['id']}"])
    import statistics as st
    fm = collections.defaultdict(list)
    for v in vids:
        fm[v["format"]].append(v["views"])
    with open(os.path.join(root, "Formats.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["format", "videos", "median_views", "max_views"])
        for k, a in sorted(fm.items(), key=lambda kv: -st.median(kv[1])):
            w.writerow([k, len(a), int(st.median(a)), max(a)])
    json.dump(dict(game=game, date=today.isoformat(), keys=keys), open(os.path.join(root, "moment_keys.json"), "w",
                                                                        encoding="utf-8"), ensure_ascii=False, indent=1)
    fast = sorted([v for v in vids if v.get("per_day")], key=lambda v: -v["per_day"])[:15]
    with open(os.path.join(root, "Trend_Report.txt"), "w", encoding="utf-8") as f:
        f.write(f"TREND RADAR - {game} - {today}  ({len(vids)} videos from {len(queries)} searches)\n\n")
        f.write("FORMATS (median views - what kind of video wins):\n")
        for k, a in sorted(fm.items(), key=lambda kv: -st.median(kv[1])):
            f.write(f"  {k:<22} {len(a):>4} videos   median {_fmt(st.median(a)):>7}   best {_fmt(max(a))}\n")
        f.write("\nFASTEST RIGHT NOW (views per day since upload):\n")
        for v in fast:
            f.write(f"  {_fmt(v['per_day']):>7}/day  {_fmt(v['views']):>7}  {v['format']:<20} {v['title'][:80]}\n")
        f.write("\nMOMENT KEYS (what people click - searched for in your footage):\n")
        for k in keys[:30]:
            f.write(f"  {k['key']:<28} score {k['score']:>6}  in {k['videos']:>3} titles  e.g. {k['example']}\n")
        f.write("\nRULES THAT APPLY: never re-upload these videos or their clips - use them only to decide WHICH moments "
                "of YOUR OWN playthrough to feature and how to title them.\n")
    # newest radar = latest/ copy used by cinematic mode
    latest = os.path.join(cfg["paths"]["output_dir"], "_Trends", game, "latest_moment_keys.json")
    json.dump(dict(game=game, date=today.isoformat(), keys=keys), open(latest, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  {len(vids)} videos, {len(keys)} moment keys -> {root}")
    return root


def load_keys(game, cfg):
    p = os.path.join(cfg["paths"]["output_dir"], "_Trends", game, "latest_moment_keys.json")
    if not os.path.exists(p):
        return []
    return json.load(open(p, encoding="utf-8")).get("keys", [])

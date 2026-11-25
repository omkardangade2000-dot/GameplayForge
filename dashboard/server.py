"""GameplayForge local dashboard.

Starts runs of `python -m gameplayforge run ...` (same environment as run.bat), shows live progress
from the run's log, and lets you preview the finished videos, thumbnails and reports.

Binds to 127.0.0.1 only. No uploads, no network calls. Standard library + psutil/yaml from the venv.
Start it with dashboard.bat, then open http://127.0.0.1:8770/
"""
import json, mimetypes, os, re, subprocess, sys, threading, time, uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlsplit, parse_qs

import psutil, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                         # D:\GameplayForge
JOBS_DIR = os.path.join(HERE, "jobs")
JOBS_FILE = os.path.join(JOBS_DIR, "jobs.json")
PYTHON = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
PORT = int(os.environ.get("GF_DASHBOARD_PORT", "8770"))
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts"}
CATEGORIES = ["Long_Videos", "YouTube_Shorts", "Cutscenes", "Reports"]
STAGE_RE = re.compile(r"^\[(\d)/6\]\s*(.+?)\s*\.*$")
os.makedirs(JOBS_DIR, exist_ok=True)

LOCK = threading.Lock()
PROCS = {}                                            # job id -> Popen (only for jobs started by this server)


def config():
    with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def paths():
    p = config()["paths"]
    return os.path.normpath(p["input_dir"]), os.path.normpath(p["output_dir"])


def load_jobs():
    try:
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def save_jobs(jobs):
    tmp = JOBS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(jobs[-200:], f, indent=1)
    os.replace(tmp, JOBS_FILE)


def log_path(jid):
    return os.path.join(JOBS_DIR, jid + ".log")


def read_tail(path, limit=400_000):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - limit))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def outcome_from_log(text):
    if "\nDone -> " in text or text.startswith("Done -> "):
        return "finished"
    if "Traceback (most recent call last)" in text or "Video not found" in text:
        return "failed"
    return None


def pid_alive(job):
    pid = job.get("pid")
    if not pid:
        return False
    try:
        p = psutil.Process(pid)
        # guard against PID reuse: must be our python started at about the recorded time
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE and abs(p.create_time() - job.get("pid_created", 0)) < 2
    except psutil.Error:
        return False


def refresh(jobs):
    """Settle the state of running jobs (works even for jobs started before a dashboard restart)."""
    changed = False
    for j in jobs:
        if j["state"] not in ("running", "stopping"):
            continue
        proc = PROCS.get(j["id"])
        code = proc.poll() if proc else None
        alive = (code is None) if proc else pid_alive(j)
        if alive:
            continue
        text = read_tail(log_path(j["id"]), 60_000)
        if j["state"] == "stopping":
            j["state"] = "stopped"
        elif proc is not None:
            j["state"] = "finished" if code == 0 else "failed"
            j["exit_code"] = code
        else:
            j["state"] = outcome_from_log(text) or "failed"
        if j["kind"] == "doctor" and j["state"] == "failed" and proc is None:
            j["state"] = "finished"          # doctor prints no "Done ->" line; its own [FAIL] lines are the result
        j["ended"] = time.time()
        PROCS.pop(j["id"], None)
        changed = True
    if changed:
        save_jobs(jobs)
    return jobs


def progress_of(text):
    """Stage + percent from the log. tqdm '\r' progress bars are reduced to their last frame."""
    stage, n = None, 0
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.split("\r")[-1].strip()
        m = STAGE_RE.match(line)
        if m:
            n, stage = int(m.group(1)), m.group(2)
    last = ""
    for line in reversed(text.replace("\r\n", "\n").split("\n")):
        seg = line.split("\r")[-1].strip()
        if seg:
            last = seg[:240]
            break
    return {"step": n, "stage": stage, "last_line": last}


SET_RE = re.compile(r'^\s*set\s+"?([A-Za-z_][A-Za-z0-9_]*)=(.*?)"?\s*$', re.I)


def run_env():
    """Same environment as run.bat: every `set NAME=value` line in run.bat is applied (GPU choice,
    FFmpeg folder, ...), so a run started here behaves exactly like `run.bat "file.mp4"` on this PC."""
    env = dict(os.environ)
    try:
        with open(os.path.join(ROOT, "run.bat"), "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = SET_RE.match(line)
                if not m:
                    continue
                name, value = m.group(1), m.group(2)
                value = value.replace("%~dp0", ROOT + os.sep)
                value = re.sub(r"%([A-Za-z_][A-Za-z0-9_]*)%", lambda v: env.get(v.group(1), ""), value)
                env[name] = value
    except OSError:
        pass
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    return env


def start_job(opts):
    kind = opts.get("kind", "run")
    if kind not in ("run", "doctor", "thumbs"):
        raise ValueError("Unknown job kind")
    inp, _ = paths()
    args = [PYTHON, "-m", "gameplayforge", kind]
    video = None
    if kind in ("run", "thumbs"):
        video = os.path.basename(str(opts.get("video", "")))
        full = os.path.join(inp, video)
        if not video or not os.path.isfile(full) or os.path.splitext(video)[1].lower() not in VIDEO_EXT:
            raise ValueError("Pick a recording from the list")
        args.append(video)
    games = list((config().get("games") or {}).keys())
    game = str(opts.get("game") or "")
    if game:
        if game not in games:
            raise ValueError("Unknown game profile")
        args += ["--game", game]
    if kind == "run":
        if opts.get("mode") == "cinematic":
            args += ["--mode", "cinematic"]
        for flag in ("commentary", "no_shorts", "no_long", "force", "no_judge"):
            if opts.get(flag):
                args.append("--" + flag.replace("_", "-"))
        lt = opts.get("long_target")
        if lt not in (None, ""):
            lt = float(lt)
            if not 30 <= lt <= 4 * 3600:
                raise ValueError("Long target must be 30 s to 4 h")
            args += ["--long-target", str(lt)]
    if kind == "thumbs":
        if opts.get("title"):
            args += ["--title", str(opts["title"])[:120]]
        if opts.get("part"):
            args += ["--part", str(opts["part"])[:40]]
    with LOCK:
        jobs = refresh(load_jobs())
        if any(j["state"] in ("running", "stopping") for j in jobs):
            raise ValueError("A job is already running. Only one runs at a time (one GPU).")
        jid = uuid.uuid4().hex[:12]
        logf = open(log_path(jid), "wb")
        logf.write(("$ " + " ".join(a if " " not in a else f'"{a}"' for a in args[1:]) + "\n").encode("utf-8"))
        logf.flush()
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        proc = subprocess.Popen(args, cwd=ROOT, env=run_env(), stdout=logf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, creationflags=flags)
        logf.close()
        try:
            created = psutil.Process(proc.pid).create_time()
        except psutil.Error:
            created = 0
        PROCS[jid] = proc
        job = {"id": jid, "kind": kind, "video": video, "options": {k: v for k, v in opts.items() if k != "kind"},
               "state": "running", "pid": proc.pid, "pid_created": created, "started": time.time(), "ended": None}
        jobs.append(job)
        save_jobs(jobs)
    return job


def stop_job(jid):
    with LOCK:
        jobs = refresh(load_jobs())
        job = next((j for j in jobs if j["id"] == jid), None)
        if not job or job["state"] != "running":
            raise ValueError("That job is not running")
        if pid_alive(job) or PROCS.get(jid):
            try:
                parent = psutil.Process(job["pid"])
                kids = parent.children(recursive=True)
                for p in kids + [parent]:
                    try:
                        p.kill()
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
        job["state"] = "stopping"
        save_jobs(jobs)
    return {"stopped": jid}


def recordings():
    inp, out = paths()
    rows = []
    try:
        for name in os.listdir(inp):
            full = os.path.join(inp, name)
            if os.path.isfile(full) and os.path.splitext(name)[1].lower() in VIDEO_EXT:
                st = os.stat(full)
                stem = os.path.splitext(name)[0]
                rows.append({"name": name, "size": st.st_size, "mtime": st.st_mtime,
                             "has_output": os.path.isdir(os.path.join(out, stem))})
    except OSError:
        pass
    return sorted(rows, key=lambda r: -r["mtime"])


def projects():
    _, out = paths()
    rows = []
    try:
        names = os.listdir(out)
    except OSError:
        names = []
    for name in names:
        d = os.path.join(out, name)
        if not os.path.isdir(d):
            continue
        counts, newest, total = {}, os.path.getmtime(d), 0
        for cat in CATEGORIES:
            files = []
            try:
                files = [f for f in os.listdir(os.path.join(d, cat)) if os.path.isfile(os.path.join(d, cat, f))]
            except OSError:
                pass
            counts[cat] = sum(1 for f in files if f.lower().endswith(".mp4"))
            for f in files:
                st = os.stat(os.path.join(d, cat, f))
                newest = max(newest, st.st_mtime)
                total += st.st_size
        rows.append({"name": name, "videos": counts, "mtime": newest, "size": total})
    return sorted(rows, key=lambda r: -r["mtime"])


def project_detail(name):
    _, out = paths()
    d = safe_join(out, name)
    if not os.path.isdir(d):
        raise ValueError("Project not found")
    files = {}
    for cat in CATEGORIES:
        items = []
        try:
            for f in sorted(os.listdir(os.path.join(d, cat))):
                full = os.path.join(d, cat, f)
                if os.path.isfile(full):
                    items.append({"name": f, "path": f"{name}/{cat}/{f}", "size": os.path.getsize(full),
                                  "mtime": os.path.getmtime(full)})
        except OSError:
            pass
        files[cat] = items
    report = ""
    rp = os.path.join(d, "Reports", "Editing_Report.txt")
    if os.path.isfile(rp):
        report = read_tail(rp, 60_000)
    return {"name": name, "files": files, "report": report}


def safe_join(base, rel):
    full = os.path.normpath(os.path.join(base, rel))
    if os.path.commonpath([os.path.normcase(full), os.path.normcase(base)]) != os.path.normcase(base):
        raise ValueError("Path outside the output folder")
    return full


_GPU = {"t": 0, "v": []}


def gpus():
    if time.time() - _GPU["t"] < 2:
        return _GPU["v"]
    rows = []
    visible = run_env().get("CUDA_VISIBLE_DEVICES")          # set in run.bat = the GPU GameplayForge uses
    visible = {v.strip() for v in visible.split(",")} if visible else set()
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        for line in res.stdout.strip().splitlines():
            i, name, util, used, total, temp = [x.strip() for x in line.split(",")]
            rows.append({"index": int(i), "name": name.replace("NVIDIA GeForce ", ""), "util": int(util),
                         "mem_used": int(used), "mem_total": int(total), "temp": int(temp), "used_by_forge": i in visible})
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    _GPU.update(t=time.time(), v=rows)
    return rows


def state():
    inp, out = paths()
    with LOCK:
        jobs = refresh(load_jobs())
    active = next((j for j in reversed(jobs) if j["state"] in ("running", "stopping")), None)
    act = None
    if active:
        act = dict(active, **progress_of(read_tail(log_path(active["id"]), 400_000)))
    try:
        free = psutil.disk_usage(out if os.path.isdir(out) else os.path.splitdrive(out)[0] + "\\").free
    except OSError:
        free = None
    return {
        "input_dir": inp, "output_dir": out,
        "games": list((config().get("games") or {}).keys()),
        "active": act,
        "history": [dict(j) for j in reversed(jobs[-30:])],
        "system": {"cpu": psutil.cpu_percent(None), "ram": psutil.virtual_memory().percent,
                   "disk_free": free, "gpus": gpus()},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "GameplayForgeDashboard/1.0"

    def log_message(self, *a):
        pass

    def host_ok(self):
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        return host in ("127.0.0.1", "localhost")      # blocks DNS-rebinding pages

    def send_json(self, code, value):
        body = json.dumps(value).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.host_ok():
            return self.send_json(403, {"error": "Open the dashboard at http://127.0.0.1:%d/" % PORT})
        u = urlsplit(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self.static("index.html", "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self.send_json(200, state())
            if u.path == "/api/recordings":
                return self.send_json(200, recordings())
            if u.path == "/api/projects":
                return self.send_json(200, projects())
            if u.path == "/api/project":
                return self.send_json(200, project_detail(q.get("name", "")))
            if u.path == "/api/log":
                jid = re.sub(r"[^a-f0-9]", "", q.get("job", ""))
                p = log_path(jid)
                if not jid or not os.path.isfile(p):
                    raise ValueError("No log for that job")
                size = os.path.getsize(p)
                off = int(q.get("offset", "0") or 0)
                if off <= 0 or off > size:
                    off = max(0, size - 200_000)
                with open(p, "rb") as f:
                    f.seek(off)
                    data = f.read(2_000_000)
                return self.send_json(200, {"offset": off + len(data), "text": data.decode("utf-8", "replace")})
            if u.path == "/api/text":
                _, out = paths()
                full = safe_join(out, q.get("path", ""))
                if not full.lower().endswith((".txt", ".csv")) or not os.path.isfile(full):
                    raise ValueError("Not a text file")
                return self.send_json(200, {"text": read_tail(full, 300_000)})
            if u.path == "/file":
                _, out = paths()
                return self.media(safe_join(out, q.get("path", "")))
            return self.send_json(404, {"error": "Not found"})
        except (ValueError, OSError) as e:
            return self.send_json(400, {"error": str(e)[:300]})

    def do_POST(self):
        if not self.host_ok() or self.headers.get("X-Forge") != "1":
            return self.send_json(403, {"error": "Forbidden"})   # custom header = no cross-site form posts
        try:
            n = min(int(self.headers.get("Content-Length") or 0), 100_000)
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("JSON object required")
            u = urlsplit(self.path).path
            if u == "/api/run":
                return self.send_json(201, start_job(body))
            if u == "/api/stop":
                return self.send_json(200, stop_job(str(body.get("job", ""))))
            if u == "/api/open":
                _, out = paths()
                full = safe_join(out, str(body.get("path", "")))
                if not os.path.exists(full):
                    raise ValueError("Not found")
                if os.path.isdir(full):
                    os.startfile(full)
                else:
                    subprocess.Popen(["explorer", "/select,", full])
                return self.send_json(200, {"opened": True})
            return self.send_json(404, {"error": "Not found"})
        except (ValueError, OSError, TypeError) as e:
            return self.send_json(400, {"error": str(e)[:300]})

    def static(self, name, ctype):
        with open(os.path.join(HERE, name), "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def media(self, full):
        if not os.path.isfile(full):
            raise ValueError("File not found")
        size = os.path.getsize(full)
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
        if m and size:
            if m.group(1):
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
            elif m.group(2):
                start = max(0, size - int(m.group(2)))
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        length = end - start + 1 if size else 0
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(full, "rb") as f:
            f.seek(start)
            left = length
            try:
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass


def main():
    mimetypes.add_type("video/mp4", ".mp4")
    psutil.cpu_percent(None)
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        print(f"Port {PORT} is busy - the dashboard is probably already running: http://127.0.0.1:{PORT}/")
        return 1
    srv.daemon_threads = True
    print(f"GameplayForge dashboard: http://127.0.0.1:{PORT}/   (Ctrl+C to stop; running jobs keep going)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

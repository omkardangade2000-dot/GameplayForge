"""Checks everything GameplayForge needs on THIS machine and says exactly what is missing."""
import importlib, os, shutil, subprocess, sys
from .util import run, pick_encoder, hw_decode_ok


def main(cfg):
    ok = True
    def line(good, msg, fix=""):
        nonlocal ok
        print(("  [OK]   " if good else "  [FAIL] ") + msg + ("" if good or not fix else "\n         -> " + fix))
        ok &= bool(good)
    print("GameplayForge doctor"); print("Python", sys.version.split()[0])
    line(sys.version_info >= (3, 10), "Python >= 3.10", "install Python 3.11 or 3.12")
    for exe in ("ffmpeg", "ffprobe"):
        line(shutil.which(exe), f"{exe} on PATH", "winget install Gyan.FFmpeg   (then reopen the terminal)")
    for m in ("numpy", "scipy", "cv2", "yaml"):
        try: importlib.import_module(m); line(True, f"python module {m}")
        except Exception: line(False, f"python module {m}", "pip install -r requirements.txt")
    try:
        enc, _ = pick_encoder(cfg); line(True, f"video encoder: {enc}" + ("" if enc.endswith("nvenc") else "  (CPU - slower; update NVIDIA driver for NVENC)"))
    except Exception as e: line(False, f"no encoder: {e}")
    line(True, "NVDEC hardware decode available" if hw_decode_ok() else "NVDEC not available (CPU decode will be used - slower)")
    if cfg["analysis"]["vision_backend"] == "siglip2":
        try:
            import torch, transformers
            line(True, f"torch {torch.__version__}, transformers {transformers.__version__}")
            cuda = torch.cuda.is_available()
            line(cuda, "CUDA GPU visible to PyTorch", "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
            if cuda:
                cap = torch.cuda.get_device_capability(0)
                print(f"         GPU: {torch.cuda.get_device_name(0)}  compute capability {cap[0]}.{cap[1]}  VRAM {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB")
                arch = torch.cuda.get_arch_list()
                line(f"sm_{cap[0]}{cap[1]}" in arch or f"compute_{cap[0]}{cap[1]}" in arch, f"PyTorch build includes this GPU architecture {arch[-3:]}", "use the CUDA 12.8 (cu128) wheels from pytorch.org - RTX 50 series needs them")
                a = torch.randn(512, 512, device="cuda", dtype=torch.float16); torch.cuda.synchronize(); line((a @ a).isfinite().all().item(), "GPU matrix-multiply self-test")
            from .vision import SigLIP2Backend
            b = SigLIP2Backend(cfg["analysis"]["model"]); import numpy as np
            e = b.encode_images(np.random.randint(0, 255, (4, 216, 384, 3), dtype=np.uint8)); t = b.encode_texts(["a test"])
            line(e.shape[1] == t.shape[1], f"vision model '{cfg['analysis']['model']}' loads and runs (embedding dim {e.shape[1]})")
            b.close()
        except Exception as ex:
            line(False, f"vision model stack: {type(ex).__name__}: {str(ex)[:150]}", "pip install -r requirements.txt  (first run downloads ~3.5 GB of model weights)")
    if cfg.get("judge", {}).get("enabled"):
        from .judge import model_cached, pick_model
        name = pick_model(cfg)
        if model_cached(name):
            try:
                from .judge import Judge, parse_json
                from PIL import Image
                import numpy as np
                j = Judge(name)
                r = j.ask([([Image.fromarray(np.zeros((360, 640, 3), np.uint8))], 'Reply with JSON only: {"color": the main color of this image}')], 20)[0]
                j.close()
                line(parse_json(r) is not None, f"judge model '{name}' loads and answers ({r.strip()[:40]})")
            except Exception as ex:
                line(False, f"judge model: {type(ex).__name__}: {str(ex)[:150]}", "pip install -r requirements.txt")
        else:
            line(False, f"judge model '{name}' not downloaded yet",
                 "run:  .venv\\Scripts\\python -c \"from huggingface_hub import snapshot_download as s; s('" + name + "')\"")
    d = cfg["paths"]["input_dir"]; line(os.path.isdir(d), f"input folder exists: {d}", "create it or edit paths.input_dir in config.yaml")
    free = shutil.disk_usage(os.path.splitdrive(cfg['paths']['output_dir'])[0] + os.sep if os.name == 'nt' else '/').free / 2**30
    line(free > 60, f"free disk space {free:.0f} GB (>= 60 GB recommended)")
    print("\nREADY" if ok else "\nFix the [FAIL] items above, then run doctor again."); return 0 if ok else 1

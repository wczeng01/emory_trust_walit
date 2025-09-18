import functools
import logging
import os
import re
import time

# OLD
# import utils.file as uf

# NEW (works whether run as a package or via editable install)
try:
    # preferred: relative import within the trust package
    from . import file as uf
except Exception:
    # fallback in case code is executed from a flat layout
    from trust.utils import file as uf  # type: ignore

# import pprint as pp

logger = logging.getLogger(__name__)


def timer(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info(f"Function '{func.__name__}' executed in {format_time(elapsed)}.")
        print(f"Function '{func.__name__}' executed in {format_time(elapsed)}.")
        return result

    return clocked


def format_time(elapsed: float) -> str:
    """Takes a time in seconds and returns a string hh:mm:ss"""
    if elapsed < 60:
        return f"{elapsed:.4f} seconds"
    elif elapsed < 3600:
        return time.strftime("%Mm:%Ss", time.gmtime(elapsed))
    else:
        return time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed))


# def get_key(filename: str, keyname: str | None = None):
#     # Determine the user's home directory
#     home_dir = os.path.expanduser("~")
#     files = uf.grab_files(os.path.join(home_dir, ".pw"), filename=filename)
#     if not files:
#         raise FileNotFoundError(f"File {filename} not found in {home_dir}/.pw")
#     elif len(files) > 1:
#         raise ValueError(f"Multiple files found with name {filename}: {files}")
#     else:
#         keys = uf.File(files[0]).load()
#         return keys[keyname] if keyname else keys

from pathlib import Path

def get_key(filename: str, keyname: str) -> str:
    """
    Return a credential value by looking for ~/.pw/<filename>
    and (optionally) OneDrive/.pw/<filename>. Accepts .txt too.
    """
    home = Path.home()
    candidates = [
        home / ".pw" / filename,
        home / ".pw" / f"{filename}.txt",
    ]

    # Also try OneDrive redirect if present
    onedrive = os.environ.get("OneDrive")
    if onedrive:
        od = Path(onedrive)
        candidates.append(od / ".pw" / filename)
        candidates.append(od / ".pw" / f"{filename}.txt")

    searched = []
    for path in candidates:
        searched.append(str(path))
        if path.exists():
            kv = {}
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    kv[k.strip().lower()] = v.strip()
            if keyname.lower() in kv:
                return kv[keyname.lower()]

    raise FileNotFoundError(
        f"Key '{keyname}' not found in '{filename}'. Tried: {searched}"
    )

def get_timestamp(fmt: str = "yy-mm-dd"):
    fmt_map = {
        "yy": "%y",
        "yyyy": "%Y",
        "mm": "%m",
        "mmm": "%b",
        "dd": "%d",
        "H": "%H",
        "M": "%M",
        "S": "%S",
    }
    pattern = re.compile(r"[y|m|d|H|M|S]+")
    matches = pattern.findall(fmt)
    if not matches:
        raise ValueError(f"Invalid format string: {fmt}")
    for match in matches:
        if match not in fmt_map:
            raise ValueError(f"Invalid format specifier: {match}")
        fmt = fmt.replace(match, fmt_map[match])
    return time.strftime(fmt, time.localtime())


if __name__ == "__main__":
    get_timestamp()

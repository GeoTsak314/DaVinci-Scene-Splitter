# Z-3 Scene Splitter v1.2 (by George Tsakalos)


import os
import sys
import subprocess
import json
import csv
from pathlib import Path

# ========== CONFIG YOU CAN EDIT ==========
# Your Resolve executable path (given by you):
RESOLVE_EXE = r"C:\DaVinci\Resolve.exe"

# Additional Resolve scripting search paths:
# If the import fails, we’ll try these + some common defaults.
RESOLVE_SCRIPT_HINTS = [
    r"C:\DaVinci\Developer\Scripting\\",
    r"C:\DaVinci\Developer\Scripting\Examples\\",
    r"C:\DaVinci\fuscript\\",
    r"C:\Program Files\Blackmagic Design\DaVinci Resolve\Developer\Scripting\\",
    r"C:\Program Files\Blackmagic Design\DaVinci Resolve\Developer\Scripting\Examples\\",
    r"C:\Program Files\Blackmagic Design\DaVinci Resolve\fuscript\\",
]
# ========================================

def has_cmd(cmd):
    try:
        subprocess.run([cmd, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except FileNotFoundError:
        return False

def ffprobe_fps(path):
    cmd = [
        "ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=r_frame_rate,avg_frame_rate",
        "-show_entries","format=duration",
        "-of","json", path
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    dur = float(data.get("format",{}).get("duration", "0") or 0.0)
    stream = data["streams"][0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    num, den = [int(x) for x in rate.split("/")]
    fps = num/den if den else float(num)
    return num, den, fps, dur

def timecode_from_seconds(sec, fps):
    frame = round(sec * fps)
    h = frame // int(fps*3600)
    frame -= int(fps*3600)*h
    m = frame // int(fps*60)
    frame -= int(fps*60)*m
    s = frame // int(fps)
    f = frame - int(fps)*s
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}", frame

def export_edl(cut_secs, fps, out_edl_path, reel="AX", clip_name="SOURCE"):
    if not cut_secs or cut_secs[0] != 0.0:
        cut_secs = [0.0] + cut_secs
    segments = [(cut_secs[i], cut_secs[i+1]) for i in range(len(cut_secs)-1)]
    with open(out_edl_path, "w", newline="", encoding="utf-8") as f:
        f.write("TITLE: SceneDetect\n")
        f.write("FCM: NON-DROP FRAME\n")
        for idx, (ss, es) in enumerate(segments, start=1):
            in_tc, _ = timecode_from_seconds(ss, fps)
            out_tc, _ = timecode_from_seconds(es, fps)
            rec_in = in_tc
            rec_out = out_tc
            f.write(f"{idx:003d}  {reel}      V     C        {in_tc} {out_tc} {rec_in} {rec_out}\n")
            f.write(f"* FROM CLIP NAME: {clip_name}\n")
    return out_edl_path

def export_csv(cut_secs, fps, out_csv_path):
    if cut_secs and cut_secs[0] != 0.0:
        cut_secs = [0.0] + cut_secs
    rows = []
    for i, sec in enumerate(cut_secs):
        tc, frame = timecode_from_seconds(sec, fps)
        rows.append({"index": i, "seconds": round(sec,6), "timecode": tc, "frame": frame})
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["index","seconds","timecode","frame"])
        w.writeheader()
        w.writerows(rows)
    return out_csv_path

def export_frames_txt(cut_secs, fps, out_txt_path):
    if cut_secs and cut_secs[0] != 0.0:
        cut_secs = [0.0] + cut_secs
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for sec in cut_secs:
            _, frame = timecode_from_seconds(sec, fps)
            f.write(str(frame) + "\n")
    return out_txt_path

def detect_with_pyscenedetect(video_path, threshold_value):
    try:
        from scenedetect import VideoManager, SceneManager
        from scenedetect.detectors import ContentDetector
    except ImportError:
        return None
    vm = VideoManager([video_path])
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold_value))
    vm.start()
    sm.detect_scenes(frame_source=vm)
    scene_list = sm.get_scene_list()
    vm.release()
    cut_secs = []
    for (start, end) in scene_list:
        sec = start.get_seconds()
        if sec > 0.0:
            cut_secs.append(sec)
    return cut_secs

def detect_with_ffmpeg(video_path, scene_th):
    cmd = [
        "ffmpeg","-hide_banner","-nostats","-i", video_path,
        "-filter:v", f"select='gt(scene,{scene_th})',showinfo",
        "-f","null","-"
    ]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, universal_newlines=True)
    cut_secs = []
    for line in proc.stderr:
        if "showinfo" in line and "pts_time:" in line:
            try:
                part = line.split("pts_time:")[1]
                sec_str = part.split(" ")[0].strip()
                sec = float(sec_str)
                cut_secs.append(sec)
            except Exception:
                pass
    proc.wait()
    cut_secs = sorted(set([round(s, 6) for s in cut_secs if s > 0.0]))
    return cut_secs

def option1_detect(video_path, pyscene_threshold, ffmpeg_threshold):
    if not os.path.isfile(video_path):
        print("Video file not found.")
        return
    if not has_cmd("ffprobe") or not has_cmd("ffmpeg"):
        print("FFmpeg/ffprobe not found in PATH.")
        return

    _, _, fps, dur = ffprobe_fps(video_path)
    print(f"Detected FPS: {fps:.6f}, Duration: {dur:.2f}s")

    cut_secs = None
    try:
        cut_secs = detect_with_pyscenedetect(video_path, pyscene_threshold)
        if cut_secs is not None:
            print(f"PySceneDetect found {len(cut_secs)} cuts (threshold {pyscene_threshold}).")
    except Exception as e:
        print("PySceneDetect failed:", e)

    if not cut_secs:
        print(f"Falling back to FFmpeg scene detection (threshold {ffmpeg_threshold})...")
        cut_secs = detect_with_ffmpeg(video_path, ffmpeg_threshold)
        print(f"FFmpeg found {len(cut_secs)} cuts.")

    snapped_frames = set()
    snapped_secs = []
    for s in cut_secs:
        _, fr = timecode_from_seconds(s, fps)
        if fr > 0 and fr not in snapped_frames:
            snapped_frames.add(fr)
            snapped_secs.append(s)

    base = Path(video_path).with_suffix("")
    out_csv = str(base) + "_scenes.csv"
    out_txt = str(base) + "_cuts.txt"
    out_edl = str(base) + "_scenes.edl"

    export_csv(snapped_secs, fps, out_csv)
    export_frames_txt(snapped_secs, fps, out_txt)
    export_edl(snapped_secs, fps, out_edl, reel="AX", clip_name=Path(video_path).name)

    print("Wrote:")
    print(" ", out_csv)
    print(" ", out_txt)
    print(" ", out_edl)
    print("Done.")

def bootstrap_resolve_paths():
    # Hints from config:
    for p in RESOLVE_SCRIPT_HINTS:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)

    # Try env override if the user sets RESOLVE_SCRIPT_LIB
    env_hint = os.environ.get("RESOLVE_SCRIPT_LIB", "").strip()
    if env_hint and os.path.isdir(env_hint) and env_hint not in sys.path:
        sys.path.append(env_hint)

    # Also set RESOLVE_SCRIPT_API to a likely fuscript dir if present
    for p in RESOLVE_SCRIPT_HINTS:
        if p.lower().endswith("fuscript\\") and os.path.isdir(p):
            os.environ.setdefault("RESOLVE_SCRIPT_API", p)
            break

def ensure_resolve_running():
    # Optional: try to launch Resolve if not already open
    # You can comment this out if you always run Resolve yourself first.
    try:
        subprocess.Popen([RESOLVE_EXE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass  # ignore launch errors; user may already have it open

def get_resolve_app():
    bootstrap_resolve_paths()
    try:
        import DaVinciResolveScript as bmd
    except Exception:
        try:
            import fusionscript as bmd
        except Exception as e:
            print("Could not import Resolve scripting module. You may set RESOLVE_SCRIPT_LIB to the scripting folder.")
            raise e
    return bmd.scriptapp("Resolve")

def option2_apply_to_resolve(video_path, cuts_txt_or_csv):
    if not os.path.isfile(video_path):
        print("Video file not found.")
        return
    if not os.path.isfile(cuts_txt_or_csv):
        print("Cuts file not found.")
        return

    # Read cut frames from TXT or CSV
    cut_frames = []
    suff = Path(cuts_txt_or_csv).suffix.lower()
    if suff == ".txt":
        with open(cuts_txt_or_csv, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        cut_frames.append(int(line))
                    except:
                        pass
    elif suff == ".csv":
        with open(cuts_txt_or_csv, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                try:
                    cut_frames.append(int(row["frame"]))
                except:
                    pass
    else:
        print("Unsupported cuts file. Use the CSV or TXT produced by option 1.")
        return

    if not has_cmd("ffprobe"):
        print("ffprobe required for fps query.")
        return
    _, _, fps, duration = ffprobe_fps(video_path)
    total_frames = int(round(duration * fps))

    cut_frames = sorted(set([f for f in cut_frames if 0 < f < total_frames]))
    segments = []
    prev = 0
    for cf in cut_frames:
        segments.append((prev, cf))
        prev = cf
    segments.append((prev, total_frames))

    ensure_resolve_running()
    resolve = get_resolve_app()
    pm = resolve.GetProjectManager()
    project = pm.GetCurrentProject()
    if not project:
        project = pm.CreateProject("SceneSplit")
        if not project:
            print("Failed to create/open project.")
            return
    mp = project.GetMediaPool()

    media_items = mp.ImportMedia([video_path])  # list of MediaPoolItem
    if not media_items or media_items[0] is None:
        # Try find by exact file path
        root = mp.GetRootFolder()
        existing = None
        for item in (root.GetClipList() or []):
            try:
                if os.path.normcase(item.GetClipProperty("File Path")) == os.path.normcase(video_path):
                    existing = item
                    break
            except:
                pass
        if not existing:
            print("Could not import or find clip in Media Pool.")
            return
        media_item = existing
    else:
        media_item = media_items[0]

    timeline_name = f"SceneSplit - {Path(video_path).name}"
    timeline = mp.CreateEmptyTimeline(timeline_name)
    if not timeline:
        print("Failed to create timeline.")
        return

    clip_entries = []
    for (start_f, end_f) in segments:
        if end_f <= start_f:
            continue
        clip_entries.append({
            "mediaPoolItem": media_item,
            "startFrame": int(start_f),
            "endFrame": int(end_f)
        })

    res = mp.AppendToTimeline(clip_entries)
    if not res:
        print("AppendToTimeline failed in this Resolve version.")
        print("Workaround: import the generated EDL (File > Import > Timeline > Pre-conformed EDL) and link it to the same source clip.")
        return

    print(f"Created timeline '{timeline_name}' with {len(clip_entries)} scene segments.")
    print("Reorder freely. Cuts are frame-accurate on the timeline.")

def prompt_float(prompt_text, default_val, min_val=None, max_val=None):
    raw = input(f"{prompt_text} [{default_val}]: ").strip()
    if raw == "":
        return float(default_val)
    try:
        val = float(raw)
        if min_val is not None and val < min_val:
            print(f"Value too small; using {default_val}")
            return float(default_val)
        if max_val is not None and val > max_val:
            print(f"Value too large; using {default_val}")
            return float(default_val)
        return val
    except:
        print(f"Invalid number; using {default_val}")
        return float(default_val)

def main():
    print("=== Scene Splitter ===")
    print("1) Detect scenes -> CSV + EDL + frames list")
    print("2) Apply cuts in DaVinci Resolve timeline (frame-accurate)")
    choice = input("Choose 1 or 2: ").strip()

    if choice == "1":
        vpath = input("Path to video: ").strip('"').strip()
        print("\nThresholds (tune if needed):")
        print(" - PySceneDetect sensitivity: typical 25–35 (lower = MORE cuts).")
        pys_th = prompt_float("Enter PySceneDetect threshold", 27.0)
        print(" - FFmpeg scene threshold: 0.20–0.50 (lower = MORE cuts).")
        ff_th = prompt_float("Enter FFmpeg scene threshold", 0.30, 0.0, 1.0)
        option1_detect(vpath, pys_th, ff_th)

    elif choice == "2":
        vpath = input("Path to the original video: ").strip('"').strip()
        cpath = input("Path to cuts file (CSV or TXT from option 1): ").strip('"').strip()
        print("Tip: make sure Resolve is open (or I’ll try to launch it).")
        option2_apply_to_resolve(vpath, cpath)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()

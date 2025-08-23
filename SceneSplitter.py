# Scene Splitter v2.1.2 (by George Tsakalos)

import os, sys, subprocess, json, csv, re, time, shutil, stat, tempfile, traceback
from pathlib import Path
from datetime import datetime



# -------------------------- Utilities --------------------------

def has_cmd(cmd: str) -> bool:
    try:
        subprocess.run([cmd, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except FileNotFoundError:
        return False

def run(cmd: list):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def _run_capture(cmd: list):
    print(">>", " ".join(cmd))
    res = subprocess.run(cmd, text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or "Command failed")
    return res.stdout, res.stderr

def ffmpeg_has_libvmaf() -> bool:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-filters"], text=True, stderr=subprocess.STDOUT)
        return "libvmaf" in out.lower()
    except Exception:
        return False

def ffmpeg_has_cuda() -> bool:
    hw = ""
    enc = ""
    try:
        hw = subprocess.check_output(["ffmpeg", "-hide_banner", "-hwaccels"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        pass
    try:
        enc = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        pass
    return ("cuda" in hw.lower()) or ("h264_nvenc" in enc.lower())

def ffprobe_fps(path: str):
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

def ffprobe_stream_info(path: str):
    cmd = ["ffprobe","-v","error","-show_streams","-show_format","-of","json", path]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    v = None; a = None
    for st in data.get("streams", []):
        if st.get("codec_type") == "video" and v is None:
            v = st
        if st.get("codec_type") == "audio" and a is None:
            a = st
    fmt = data.get("format", {})
    def g(d,k,default=None):
        return d.get(k, default) if isinstance(d, dict) else default
    info = {
        "container": Path(path).suffix.lower().lstrip("."),
        "duration": float(g(fmt,"duration",0.0) or 0.0),
        "size_bytes": int(g(fmt,"size",0) or 0),
        "bitrate": int(g(fmt,"bit_rate",0) or 0),
        "v_codec": g(v,"codec_name",""),
        "v_profile": g(v,"profile",""),
        "v_pix_fmt": g(v,"pix_fmt",""),
        "width": int(g(v,"width",0) or 0),
        "height": int(g(v,"height",0) or 0),
        "v_rate": g(v,"avg_frame_rate", g(v,"r_frame_rate","0/1")),
        "a_codec": g(a,"codec_name",""),
        "a_channels": int(g(a,"channels",0) or 0),
        "a_sr": int(g(a,"sample_rate",0) or 0),
        "a_layout": g(a,"channel_layout",""),
    }
    try:
        rn, rd = [int(x) for x in (info["v_rate"] or "0/1").split("/")]
        info["fps"] = rn/rd if rd else float(rn)
    except Exception:
        info["fps"] = 0.0
    return info

def human_bitrate(bps: int):
    if not bps: return "n/a"
    kbps = bps/1000.0
    return (f"{kbps:.0f} kbps" if kbps < 1000 else f"{kbps/1000.0:.2f} Mbps")

def timecode_from_seconds(sec: float, fps: float):
    frame = round(sec * fps)
    h = frame // int(fps*3600)
    frame -= int(fps*3600)*h
    m = frame // int(fps*60)
    frame -= int(fps*60)*m
    s = frame // int(fps)
    f = frame - int(fps)*s
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}", frame

def nearest_standard_fps(fps: float):
    std = [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0]
    return min(std, key=lambda x: abs(x - fps))

def ask_gpu(prompt_hint: str) -> bool:
    print(f"\nUse GPU ({prompt_hint}) if available? (y/N)")
    ans = input("> ").strip().lower()
    return ans in ("y","yes","1","true")




# -------------------------- Option 1: Scene detection --------------------------

def detect_with_pyscenedetect(video_path: str, threshold_value: float):
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        import logging
        logging.getLogger('pyscenedetect').setLevel(logging.ERROR)
    except ImportError:
        return None
    try:
        video = open_video(video_path)
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=threshold_value))
        sm.detect_scenes(video)
        scene_list = sm.get_scene_list()
    except Exception:
        return None

    cut_secs = []
    for (start, _end) in scene_list:
        try:
            sec = start.get_seconds()
        except AttributeError:
            try:
                sec = float(start)
            except Exception:
                continue
        if sec > 0.0:
            cut_secs.append(sec)
    return cut_secs

def ffmpeg_has_nvdec():
    try:
        out = subprocess.check_output(["ffmpeg","-hide_banner","-decoders"], text=True, stderr=subprocess.STDOUT)
        return "h264_cuvid" in out.lower() or "hevc_cuvid" in out.lower()
    except Exception:
        return False

def detect_with_ffmpeg(video_path: str, scene_th: float, use_gpu: bool):
    filters = f"select='gt(scene,{scene_th})',showinfo"
    can_cuda = ffmpeg_has_cuda()
    if use_gpu and can_cuda:
        cmd = [
            "ffmpeg","-hide_banner","-nostats",
            "-hwaccel","cuda","-hwaccel_output_format","cuda",
            "-i", video_path,
            "-filter:v", f"hwdownload,format=yuv420p,{filters}",
            "-f","null","-"
        ]
        print("[Option 1] Using CUDA for decode (frames downloaded to CPU for detection).")
    else:
        if use_gpu and not can_cuda:
            print("[Option 1] GPU requested but CUDA/NVENC not found in this ffmpeg build. Falling back to CPU.")
        cmd = [
            "ffmpeg","-hide_banner","-nostats","-i", video_path,
            "-filter:v", filters, "-f","null","-"
        ]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, universal_newlines=True)
    cut_secs = []
    for line in proc.stderr:
        if "showinfo" in line and "pts_time:" in line:
            try:
                sec = float(line.split("pts_time:")[1].split(" ")[0].strip())
                cut_secs.append(sec)
            except Exception:
                pass
    proc.wait()
    return sorted(set([round(s, 6) for s in cut_secs if s > 0.0]))

def export_edl(cut_secs, fps, out_edl_path, reel="AX", clip_name="SOURCE"):
    if not cut_secs or cut_secs[0] != 0.0:
        cut_secs = [0.0] + cut_secs
    segments = [(cut_secs[i], cut_secs[i+1]) for i in range(len(cut_secs)-1)]
    with open(out_edl_path, "w", newline="", encoding="utf-8") as f:
        f.write("TITLE: SceneDetect\nFCM: NON-DROP FRAME\n")
        for idx, (ss, es) in enumerate(segments, start=1):
            in_tc, _ = timecode_from_seconds(ss, fps)
            out_tc, _ = timecode_from_seconds(es, fps)
            f.write(f"{idx:03d}  {reel}      V     C        {in_tc} {out_tc} {in_tc} {out_tc}\n")
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
        w.writeheader(); w.writerows(rows)
    return out_csv_path

def export_frames_txt(cut_secs, fps, out_txt_path):
    if cut_secs and cut_secs[0] != 0.0:
        cut_secs = [0.0] + cut_secs
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for sec in cut_secs:
            _, frame = timecode_from_seconds(sec, fps)
            f.write(str(frame) + "\n")
    return out_txt_path

def option1_detect(video_path: str, pyscene_th: float, ffmpeg_th: float, use_gpu: bool):
    if not os.path.isfile(video_path):
        print("Video file not found."); return
    if not (has_cmd("ffmpeg") and has_cmd("ffprobe")):
        print("FFmpeg/ffprobe not found in PATH."); return

    _, _, fps, dur = ffprobe_fps(video_path)
    print(f"Detected FPS: {fps:.6f} (nearest std {nearest_standard_fps(fps)}), Duration: {dur:.2f}s")

    cuts = None
    try:
        cuts = detect_with_pyscenedetect(video_path, pyscene_th)
        if cuts is not None:
            print(f"PySceneDetect cuts: {len(cuts)} (threshold {pyscene_th})")
    except Exception as e:
        print("PySceneDetect failed:", e)

    if not cuts:
        print(f"Falling back to FFmpeg scene detection (threshold {ffmpeg_th})...")
        cuts = detect_with_ffmpeg(video_path, ffmpeg_th, use_gpu=use_gpu)
        print(f"FFmpeg cuts: {len(cuts)}")

    seen = set(); snapped = []
    for s in cuts:
        _, fr = timecode_from_seconds(s, fps)
        if fr > 0 and fr not in seen:
            seen.add(fr); snapped.append(s)

    base = Path(video_path).with_suffix("")
    out_csv = str(base) + "scenes.csv"
    out_txt = str(base) + "cuts.txt"
    out_edl = str(base) + "scenes.edl"
    export_csv(snapped, fps, out_csv)
    export_frames_txt(snapped, fps, out_txt)
    export_edl(snapped, fps, out_edl, reel="AX", clip_name=Path(video_path).name)
    print("Wrote:\n ", out_csv, "\n ", out_txt, "\n ", out_edl, "\nDone.")





# -------------------------- Option 2: Build final (edited in DaVinci Resolve) file --------------------------

EDL_LINE = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+V\s+C\s+(\d{2}:\d{2}:\d{2}:\d{2})\s+(\d{2}:\d{2}:\d{2}:\d{2})\s+(\d{2}:\d{2}:\d{2}:\d{2})\s+(\d{2}:\d{2}:\d{2}:\d{2})"
)

def parse_timecode_to_seconds(tc: str, fps: float) -> float:
    h, m, s, f = [int(x) for x in tc.split(":")]
    return h*3600 + m*60 + s + f/float(fps)

def parse_edl_segments(edl_path: str, fps: float):
    segs = []
    with open(edl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = EDL_LINE.search(line)
            if not m: continue
            si = parse_timecode_to_seconds(m.group(3), fps)
            so = parse_timecode_to_seconds(m.group(4), fps)
            if so > si: segs.append((si, so))
    if not segs:
        raise ValueError("No segments parsed from EDL.")
    return segs

def ffprobe_stream_props(path: str):
    cmd = [
        "ffprobe","-v","error",
        "-select_streams","v:0","-show_entries","stream=codec_name,profile,level,pix_fmt,width,height,r_frame_rate,bit_rate",
        "-select_streams","a:0","-show_entries","stream=codec_name,channels,channel_layout,sample_rate,bit_rate",
        "-of","json", path
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    v = None; a = None
    for st in data.get("streams", []):
        if st.get("codec_type") == "video" or "width" in st:
            v = st
        elif st.get("codec_type") == "audio" or "sample_rate" in st:
            a = st
    return v, a

def build_concat_listfile(paths, list_dir: Path):
    list_dir.mkdir(parents=True, exist_ok=True)
    lf = list_dir / "concat_list.txt"
    with open(lf, "w", encoding="utf-8") as tf:
        for p in paths:
            tf.write(f"file '{p}'\n")
    return str(lf)

def suggested_crf(width, height, fps):
    area = (width or 1920) * (height or 1080)
    if area >= 3840*2160: return int(os.environ.get("SCENESPLIT_CRF", 20))
    if fps >= 60 and area >= 1920*1080: return int(os.environ.get("SCENESPLIT_CRF", 19))
    return int(os.environ.get("SCENESPLIT_CRF", 18))

def x264_params_for_fps(fps):
    gop = max(24, int(round(fps*2))); min_k = max(12, int(round(fps)))
    return f"keyint={gop}:min-keyint={min_k}:scenecut=40:force-cfr=1"

def aac_bitrate_and_channels(channels):
    if channels and channels >= 6: return int(os.environ.get("SCENESPLIT_AAC_KBPS", 384)), 6
    return int(os.environ.get("SCENESPLIT_AAC_KBPS", 192)), 2

def encode_segment_to_h264(input_video, start, dur, out_path, width, height, fps, channels, sample_rate, use_gpu: bool):
    a_kbps, ac = aac_bitrate_and_channels(channels)
    ar = 48000 if (sample_rate or 48000) >= 48000 else (sample_rate or 48000)
    if use_gpu and ffmpeg_has_cuda():
        gop = max(24, int(round(fps*2)))
        preset = os.environ.get("SCENESPLIT_NVENC_PRESET", "p6")
        cq = os.environ.get("SCENESPLIT_NVENC_CQ", "19")
        cmd = [
            "ffmpeg","-y","-hide_banner","-nostats",
            "-ss", f"{start:.6f}","-t", f"{dur:.6f}",
            "-i", input_video,
            "-c:v","h264_nvenc","-preset", preset,
            "-rc","vbr","-cq", cq,"-b:v","0","-maxrate","0",
            "-g", str(gop), "-bf","3","-profile:v","high","-pix_fmt","yuv420p",
            "-r", f"{fps:.6f}","-rc-lookahead","32","-spatial-aq","1","-temporal-aq","1",
            "-c:a","aac","-b:a", f"{a_kbps}k","-ac", str(ac),"-ar", str(ar),
            str(out_path)
        ]
        print("[Option 2] Using NVENC (GPU) for encode.")
    else:
        crf = suggested_crf(width, height, fps)
        x264p = x264_params_for_fps(fps)
        preset = os.environ.get("SCENESPLIT_PRESET", "slow")
        cmd = [
            "ffmpeg","-y","-hide_banner","-nostats",
            "-ss", f"{start:.6f}","-t", f"{dur:.6f}",
            "-i", input_video,
            "-c:v","libx264","-preset", preset,"-crf", str(crf),
            "-pix_fmt","yuv420p","-r", f"{fps:.6f}","-x264-params", x264p,
            "-c:a","aac","-b:a", f"{a_kbps}k","-ac", str(ac),"-ar", str(ar),
            str(out_path)
        ]
        if use_gpu and not ffmpeg_has_cuda():
            print("[Option 2] GPU requested but NVENC not found; falling back to CPU x264.")
    run(cmd)

def _choose_temp_root_from_user():
    default_tmp = Path(tempfile.gettempdir())
    print("\n(Optional) Temporary working folder for segments/concat.")
    print(f"Leave blank for system temp [{default_tmp}]")
    t = input("Temp folder path: ").strip('"').strip()
    if not t:
        tmp = default_tmp
    else:
        tmp = Path(os.path.expanduser(t))
        try: tmp.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Could not create '{tmp}' ({e}); falling back to system temp."); tmp = default_tmp
    return tmp

def _ask_auto_delete() -> bool:
    ans = input("Auto-delete the temp folder after success? (y/N): ").strip().lower()
    return ans in ("y","yes")

def _make_job_temp_dir(temp_root: Path, input_video: str):
    stamp = int(time.time()); safe_stem = Path(input_video).stem or "video"
    jobdir = temp_root / f"scenesplit_{safe_stem}_{stamp}"
    jobdir.mkdir(parents=True, exist_ok=True); return jobdir

def _on_rm_error(func, path, exc_info):
    try: os.chmod(path, stat.S_IWRITE); func(path)
    except Exception: pass

def remove_dir_tree(p: Path, attempts: int = 3) -> bool:
    for i in range(attempts):
        try: shutil.rmtree(p, onerror=_on_rm_error); return True
        except Exception: time.sleep(0.5*(i+1))
    return False

def option2_build_h264_normalized(input_video: str, edl_path: str, container_ext: str, temp_root: Path, auto_delete: bool, use_gpu: bool):
    if not (has_cmd("ffmpeg") and has_cmd("ffprobe")):
        print("FFmpeg/ffprobe not found in PATH."); return
    if not (os.path.isfile(input_video) and os.path.isfile(edl_path)):
        print("Missing input video or EDL file."); return

    _, _, fps, _ = ffprobe_fps(input_video)
    vprops, aprops = ffprobe_stream_props(input_video)
    width = (vprops or {}).get("width"); height = (vprops or {}).get("height")
    channels = (aprops or {}).get("channels") or 2
    sample_rate = (aprops or {}).get("sample_rate")
    try: sample_rate = int(sample_rate) if sample_rate else 48000
    except: sample_rate = 48000

    segments = parse_edl_segments(edl_path, fps)
    if not segments: print("No segments in EDL."); return

    jobdir = _make_job_temp_dir(temp_root, input_video)
    print(f"[Temp] Using working folder: {jobdir}")

    success = False
    try:
        parts = []
        for idx, (start, end) in enumerate(segments):
            dur_seg = max(0.0, end-start)
            if dur_seg <= 0.0: continue
            part = jobdir / f"part_{idx:03d}.mkv"
            encode_segment_to_h264(input_video, start, dur_seg, str(part), width, height, fps, channels, sample_rate, use_gpu=use_gpu)
            parts.append(str(part))

        listfile = build_concat_listfile(parts, jobdir)
        inter_out = jobdir / "concat.mkv"
        run(["ffmpeg","-y","-hide_banner","-nostats","-f","concat","-safe","0","-i", listfile,"-c","copy", str(inter_out)])

        out_base = str(Path(input_video).with_suffix("")) + "_EDITED"
        out_path = out_base + container_ext
        if container_ext.lower()==".mp4":
            run(["ffmpeg","-y","-hide_banner","-nostats","-i", str(inter_out),"-c","copy","-movflags","+faststart", out_path])
        else:
            run(["ffmpeg","-y","-hide_banner","-nostats","-i", str(inter_out),"-c","copy", out_path])
        print(f"Done. Output: {out_path}")
        success = True
    finally:
        if success and auto_delete:
            ok = remove_dir_tree(jobdir)
            print("[Temp] Auto-deleted working folder." if ok else f"[Temp] Could not delete working folder: {jobdir}")
        else:
            print(f"[Temp] Kept working folder: {jobdir}")

def option2_prompt():
    vpath = input("Path to the original video: ").strip('"').strip()
    edl = input("Path to EDL (CMX3600) with the kept segments: ").strip('"').strip()
    print("\nOutput container?\n  1) MKV (default)\n  2) MP4\n  3) AVI")
    fmt_choice = input("Select number [1]: ").strip() or "1"
    ext = ".mkv" if fmt_choice=="1" else ".mp4" if fmt_choice=="2" else ".avi"
    temp_root = _choose_temp_root_from_user()
    auto_del = _ask_auto_delete()
    use_gpu = ask_gpu("NVENC encode")
    option2_build_h264_normalized(vpath, edl, ext, temp_root, auto_del, use_gpu)




# -------------------------- Option 3: Quality Compare (original EDL-aligned vs edited)--------------------------

def _vmaf_try_parse_pooled(vmaf_json_path):
    try:
        with open(vmaf_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    pooled = data.get("pooled_metrics", {})
    if pooled and "vmaf" in pooled:
        vm = pooled.get("vmaf", {})
        return {"mean": vm.get("mean"), "min": vm.get("min"), "5th": vm.get("5th_percentile")}
    frames = data.get("frames", [])
    vals = []
    for fr in frames:
        fm = fr.get("metrics", {}) or fr.get("frameMetrics", {})
        if "vmaf" in fm:
            try: vals.append(float(fm["vmaf"]))
            except: pass
    if not vals: return None
    vals_sorted = sorted(vals); n = len(vals_sorted)
    def pct(p):
        if n==0: return None
        idx = max(0, min(n-1, int(round(p/100.0*(n-1)))))
        return vals_sorted[idx]
    return {"mean": sum(vals)/n, "min": min(vals), "5th": pct(5)}

def _interpret_vmaf(score: float) -> str:
    if score is None: return "n/a"
    if score >= 95: return "95–100: essentially indistinguishable"
    if score >= 85: return "85–95: good, mild differences"
    if score >= 80: return "80–85: fair; visible in places"
    return "<80: visible degradation likely"

def _interpret_ssim(ssim: float) -> str:
    if ssim is None: return "n/a"
    if ssim >= 0.98: return "≥0.98: excellent / near-identical"
    if ssim >= 0.95: return "0.95–0.98: good"
    if ssim >= 0.90: return "0.90–0.95: fair; visible differences"
    return "<0.90: poor"

def _interpret_psnr(psnr: float) -> str:
    if psnr is None: return "n/a"
    if psnr >= 40: return "≥40 dB: very good / excellent"
    if psnr >= 35: return "35–40 dB: good"
    if psnr >= 30: return "30–35 dB: fair"
    return "<30 dB: poor"

def _build_select_expr(segments):
    parts = [f"between(t,{s:.6f},{e:.6f})" for (s,e) in segments if e > s]
    return "+".join(parts) if parts else "gte(t,0)"

def run_quality_compare_edl(edited: str, original: str, edl_path: str, report_txt_path: str, use_hw: bool):
    if not (has_cmd("ffmpeg") and has_cmd("ffprobe")):
        print("FFmpeg/ffprobe not found in PATH."); return
    if not (os.path.isfile(original) and os.path.isfile(edited) and os.path.isfile(edl_path)):
        print("Missing file(s). Provide original, edited, and EDL."); return

    # Probe & parse EDL (use original's fps)
    _, _, ref_fps, _ = ffprobe_fps(original)
    ref_info = ffprobe_stream_info(original)
    dist_info = ffprobe_stream_info(edited)
    w = ref_info.get("width") or 1920; h = ref_info.get("height") or 1080
    scaled_to = f"{w}x{h}"

    segments = parse_edl_segments(edl_path, ref_fps)
    sel_expr = _build_select_expr(segments)

    tmpdir = Path(tempfile.mkdtemp(prefix="metrics_"))
    vmaf_avail = ffmpeg_has_libvmaf()
    can_cuda = ffmpeg_has_cuda()
    if use_hw and not can_cuda:
        print("[Option 3] GPU requested but CUDA/NVENC not found; falling back to CPU.")
        use_hw = False



    # --------- Build commands ---------
    def vmaf_cmd():
        vmaf_json = tmpdir / "vmaf.json"
        opts = f"log_path={vmaf_json}:log_fmt=json:n_threads=8:phone_model=0"
        opts += ":feature=name=ssim|name=ms_ssim|name=psnr"
        if use_hw:
            vf = (
              f"[1:v]scale_cuda={w}:{h}:format=yuv420p,hwdownload,format=yuv420p,"
              f"select='{sel_expr}',fps={ref_fps:.6f},setpts=N/FRAME_RATE/TB[ref];"
              f"[0:v]scale_cuda={w}:{h}:format=yuv420p,hwdownload,format=yuv420p,"
              f"fps={ref_fps:.6f},setpts=PTS-STARTPTS[dist];"
              f"[dist][ref]libvmaf={opts}"
            )
            return [
              "ffmpeg",
              "-hwaccel","cuda","-hwaccel_output_format","cuda","-i", edited,
              "-hwaccel","cuda","-hwaccel_output_format","cuda","-i", original,
              "-lavfi", vf, "-f","null","-"
            ], vmaf_json
        else:
            vf = (
              f"[1:v]select='{sel_expr}',scale={w}:{h}:flags=bicubic,format=yuv420p,"
              f"fps={ref_fps:.6f},setpts=N/FRAME_RATE/TB[ref];"
              f"[0:v]scale={w}:{h}:flags=bicubic,format=yuv420p,"
              f"fps={ref_fps:.6f},setpts=PTS-STARTPTS[dist];"
              f"[dist][ref]libvmaf={opts}"
            )
            return ["ffmpeg","-i", edited,"-i", original,"-lavfi", vf,"-f","null","-"], vmaf_json

    def metric_cmd(metric_name):
        if use_hw:
            vf = (
              f"[1:v]scale_cuda={w}:{h}:format=yuv420p,hwdownload,format=yuv420p,"
              f"select='{sel_expr}',fps={ref_fps:.6f},setpts=N/FRAME_RATE/TB[ref];"
              f"[0:v]scale_cuda={w}:{h}:format=yuv420p,hwdownload,format=yuv420p,"
              f"fps={ref_fps:.6f},setpts=PTS-STARTPTS[dist];"
              f"[dist][ref]{metric_name}=stats_file=-"
            )
            return [
              "ffmpeg",
              "-hwaccel","cuda","-hwaccel_output_format","cuda","-i", edited,
              "-hwaccel","cuda","-hwaccel_output_format","cuda","-i", original,
              "-lavfi", vf, "-f","null","-"
            ]
        else:
            vf = (
              f"[1:v]select='{sel_expr}',scale={w}:{h}:flags=bicubic,format=yuv420p,"
              f"fps={ref_fps:.6f},setpts=N/FRAME_RATE/TB[ref];"
              f"[0:v]scale={w}:{h}:flags=bicubic,format=yuv420p,"
              f"fps={ref_fps:.6f},setpts=PTS-STARTPTS[dist];"
              f"[dist][ref]{metric_name}=stats_file=-"
            )
            return ["ffmpeg","-i", edited,"-i", original,"-lavfi", vf,"-f","null","-"]

    # --------- Run VMAF / SSIM / PSNR ---------
    vmaf_mean = vmaf_min = vmaf_5p = None
    if vmaf_avail:
        try:
            cmd, vmaf_json = vmaf_cmd()
            _out, _err = _run_capture(cmd)
            pooled = _vmaf_try_parse_pooled(str(vmaf_json))
            if pooled:
                vmaf_mean, vmaf_min, vmaf_5p = pooled.get("mean"), pooled.get("min"), pooled.get("5th")
        except Exception as e:
            print("VMAF run failed; continuing without VMAF. Error:", e)
            vmaf_avail = False
    else:
        print("Note: libvmaf not found in your ffmpeg build. Skipping VMAF.")

    ssim_all = None
    try:
        _o, ssim_stderr = _run_capture(metric_cmd("ssim"))
        m = re.search(r"SSIM.*All:([0-9.]+)", ssim_stderr)
        if m: ssim_all = float(m.group(1))
    except Exception as e:
        print("SSIM run failed:", e)

    psnr_avg = None
    try:
        _o, psnr_stderr = _run_capture(metric_cmd("psnr"))
        m2 = re.search(r"average:([0-9.]+)", psnr_stderr)
        if m2: psnr_avg = float(m2.group(1))
    except Exception as e:
        print("PSNR run failed:", e)




    # --------- Report ---------
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("Scene Splitter — Quality Comparison Report (EDL-aligned)")
    lines.append(f"Generated: {dt}")
    lines.append("")
    lines.append("Inputs:")
    lines.append(f"  Reference (original): {original}")
    lines.append(f"  Distorted  (edited) : {edited}")
    lines.append(f"  EDL (kept segments): {edl_path}")
    lines.append("")
    lines.append("Reference properties:")
    lines.append(f"  Container: .{ref_info['container']}  Duration: {ref_info['duration']:.2f}s  Size: {ref_info['size_bytes']/1_000_000:.2f} MB  Bitrate: {human_bitrate(ref_info['bitrate'])}")
    lines.append(f"  Video: {ref_info['v_codec']} {ref_info['width']}x{ref_info['height']} @ {ref_info['fps']:.3f} fps  pix_fmt={ref_info['v_pix_fmt']}  profile={ref_info['v_profile']}")
    lines.append(f"  Audio: {ref_info['a_codec']} {ref_info['a_channels']}ch @ {ref_info['a_sr']} Hz ({ref_info['a_layout']})")
    lines.append("")
    lines.append("Edited properties:")
    lines.append(f"  Container: .{dist_info['container']}  Duration: {dist_info['duration']:.2f}s  Size: {dist_info['size_bytes']/1_000_000:.2f} MB  Bitrate: {human_bitrate(dist_info['bitrate'])}")
    lines.append(f"  Video: {dist_info['v_codec']} {dist_info['width']}x{dist_info['height']} @ {dist_info['fps']:.3f} fps  pix_fmt={dist_info['v_pix_fmt']}  profile={dist_info['v_profile']}")
    lines.append(f"  Audio: {dist_info['a_codec']} {dist_info['a_channels']}ch @ {dist_info['a_sr']} Hz ({dist_info['a_layout']})")
    lines.append("")
    lines.append(f"Alignment: Original filtered by EDL (frame ranges concatenated), both streams scaled to {scaled_to}, forced to {ref_fps:.3f} fps CFR, starts aligned to t=0.")
    lines.append("")
    lines.append("Results:")
    if vmaf_avail and vmaf_mean is not None:
        lines.append(f"  VMAF: mean={vmaf_mean:.2f}   min={vmaf_min:.2f}   5th percentile={vmaf_5p:.2f}   → {_interpret_vmaf(vmaf_mean)}")
    elif vmaf_avail:
        lines.append("  VMAF: available but parsing failed")
    else:
        lines.append("  VMAF: not available in this ffmpeg build")
    lines.append(f"  SSIM (All): {ssim_all:.4f}   → {_interpret_ssim(ssim_all)}" if ssim_all is not None else "  SSIM (All): n/a")
    lines.append(f"  PSNR average: {psnr_avg:.2f} dB   → {_interpret_psnr(psnr_avg)}" if psnr_avg is not None else "  PSNR average: n/a")
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("  VMAF: 95–100 ≈ indistinguishable; 85–95 good; 80–85 fair; <80 visible degradation likely.")
    lines.append("  SSIM: ≥0.98 excellent; 0.95–0.98 good; 0.90–0.95 fair; <0.90 poor.")
    lines.append("  PSNR: ≥40 dB excellent; 35–40 dB good; 30–35 dB fair; <30 dB poor.")
    lines.append("")
    lines.append("Notes:")
    lines.append("  • This comparison uses your EDL to cut the ORIGINAL on-the-fly, so timeline/content matches the EDITED file.")
    lines.append("  • GPU mode here accelerates decode/scale only; metrics run on CPU.")
    lines.append("  • libvmaf must be present in ffmpeg to compute VMAF (otherwise only SSIM/PSNR are reported).")

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved quality report: {report_txt_path}")

    try: shutil.rmtree(tmpdir, onerror=_on_rm_error)
    except Exception: pass

def option3_prompt():
    print("\n=== Quality Compare (VMAF/SSIM/PSNR, EDL-aligned) ===")
    ref = input("Path to ORIGINAL (reference) video: ").strip('"').strip()
    dist = input("Path to EDITED   (distorted) video: ").strip('"').strip()
    edl  = input("Path to EDL (CMX3600) that defines kept segments: ").strip('"').strip()
    if not (os.path.isfile(ref) and os.path.isfile(dist) and os.path.isfile(edl)):
        print("One or more paths are invalid."); return
    use_hw = ask_gpu("CUDA decode/scale")
    report_path = str(Path(dist).with_suffix("")) + "_quality_report.txt"
    try:
        run_quality_compare_edl(dist, ref, edl, report_path, use_hw=use_hw)
    except Exception as e:
        print("Quality comparison failed:", e)
        traceback.print_exc()





# -------------------------- Main Menu --------------------------

def prompt_float(prompt_text, default_val, min_val=None, max_val=None):
    raw = input(f"{prompt_text} [{default_val}]: ").strip()
    if raw == "": return float(default_val)
    try:
        val = float(raw)
        if min_val is not None and val < min_val: print(f"Too small; using {default_val}"); return float(default_val)
        if max_val is not None and val > max_val: print(f"Too large; using {default_val}"); return float(default_val)
        return val
    except Exception:
        print(f"Invalid; using {default_val}"); return float(default_val)

def main():
    while True:
        print("\n=== Scene Splitter (by George Tsakalos) ===")
        print("1) Detect scenes -> CSV + EDL + frames list")
        print("2) Build final file (H.264-normalized, concat-safe)  [use your EDL]")
        print("3) Compare quality (VMAF/SSIM/PSNR) — EDL-aligned")
        print("4) Exit")
        choice = input("Choose 1, 2, 3 or 4: ").strip()

        if choice == "1":
            try:
                vpath = input("Path to video (full path): ").strip('"').strip()
                print("\nThresholds:")
                print(" - PySceneDetect sensitivity: typical 25–35 (lower = MORE cuts).")
                pys_th = prompt_float("Enter PySceneDetect threshold", 27.0)
                print(" - FFmpeg scene threshold: 0.20–0.50 (lower = MORE cuts).")
                ff_th = prompt_float("Enter FFmpeg scene threshold", 0.30, 0.0, 1.0)
                use_gpu = ask_gpu("CUDA decode (ffmpeg path)")
                option1_detect(vpath, pys_th, ff_th, use_gpu)
            except Exception as e:
                print("Error in Option 1:", e); traceback.print_exc()
            input("\n[Enter] to return to main menu...")

        elif choice == "2":
            try:
                option2_prompt()
            except Exception as e:
                print("Error in Option 2:", e); traceback.print_exc()
            input("\n[Enter] to return to main menu...")

        elif choice == "3":
            try:
                option3_prompt()
            except Exception as e:
                print("Error in Option 3:", e); traceback.print_exc()
            input("\n[Enter] to return to main menu...")

        elif choice == "4":
            print("Bye!"); break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        try: input("\n[Press Enter to close]")
        except Exception: pass

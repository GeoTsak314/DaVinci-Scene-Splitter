# Scene Splitter for DaVinci Resolve v2.1.2 (by George Tsakalos)

Fast & frame-accurate editing workflows driven by a Resolve-free Python helper:

- **Option 1 – Detect scenes** (PySceneDetect or FFmpeg fallback; optional **CUDA** decode) → outputs **CSV**, **TXT** (frames), **EDL** (CMX3600).
- **Option 2 – Build final file** from an **EDL**: re-encodes selected segments to **harmonized H.264** (concat-safe) using **CPU x264** or **NVENC**, concatenates, and writes **MKV/MP4/AVI**. Lets you pick a **temp folder** and **auto-delete** it when done.
- **Option 3 – Compare quality** of your **edited file** vs the **original filtered by the same EDL** (no pre-render): computes **VMAF** (if your FFmpeg has `libvmaf`), **SSIM**, and **PSNR**. Optional **CUDA** decode/scale; metrics run on CPU. Writes a **TXT report** next to the edited file.

---

## Requirements

- **Windows 10 or later**
- **Python 3.9+**
- **FFmpeg & ffprobe** in `PATH`
  - For **GPU**: FFmpeg build with **CUDA/NVENC** (`h264_nvenc`, `-hwaccels` shows `cuda`).
  - For **VMAF** in Option 3: FFmpeg with **`libvmaf`** and a **VMAF JSON model** file (e.g. `vmaf_v0.6.1.json`).

- **Python packages** (only needed for the PySceneDetect path in Option 1):
  ```bash
  pip install -r requirements.txt
  ```

---

## Install

1. Place files in a folder, e.g. `C:\SceneSplitter\`:
   - `SceneSplitter.py`
   - `requirements.txt`
   - `README.md` (this file)

2. Install Python deps:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify FFmpeg:
   ```bash
   ffmpeg -version
   ffprobe -version
   ```

4. (Optional) Verify GPU & VMAF capability:
   ```bash
   ffmpeg -hide_banner -hwaccels
   ffmpeg -hide_banner -encoders | findstr /I nvenc
   ffmpeg -hide_banner -filters  | findstr /I vmaf
   ```

---

## Usage

Run:
```bash
python SceneSplitter.py
```

You’ll see the main menu:

### 1) Detect scenes → CSV + EDL + frames list
Prompts for:
- **Video path**
- **PySceneDetect threshold** (default **27.0**, typical **25–35**; lower = more cuts)
- **FFmpeg scene threshold** (default **0.30**, typical **0.20–0.50**; lower = more cuts)
- **Use GPU?** (*CUDA decode* for the FFmpeg fallback)

Outputs (next to your video):
- `*scenes.csv` — seconds, timecodes, frame numbers
- `*cuts.txt`   — frame numbers only
- `*scenes.edl` — **CMX3600** EDL (NON-DROP FCM)

If PySceneDetect isn’t installed, it auto-falls back to FFmpeg.

---

### 2) Build final file (H.264-normalized, concat-safe) [use your EDL]
Prompts for:
- **Original video path**
- **EDL path** (CMX3600; from Option 1 *or* exported from your NLE)
- **Output container**: MKV / MP4 / AVI
- **Temp working folder** (optional; use a fast SSD/NVMe for speed)
- **Auto-delete temp folder** (y/N)
- **Use GPU?** (**NVENC** encode)

What it does:
- Reads EDL, encodes each segment with **uniform H.264 params** (profile/pix_fmt/GOP/CFR) → **concat-safe**.
- Concatenates without re-encode and writes your chosen container.
- MP4 outputs use `-movflags +faststart`.

Quality knobs (optional env vars):
- NVENC:
  - `SCENESPLIT_NVENC_PRESET=p7` (p1 fastest … p7 best)
  - `SCENESPLIT_NVENC_CQ=18` (lower = higher quality)
- CPU x264:
  - `SCENESPLIT_PRESET=slow` (default)
  - `SCENESPLIT_CRF=18` (lower CRF = higher quality)

---

### 3) Compare quality (VMAF/SSIM/PSNR) — EDL-aligned
Prompts for:
- **Original (reference) video path**
- **Edited (distorted) video path**
- **EDL path** (same one used to create the edit)
- **Use GPU?** (CUDA decode/scale; metrics run on CPU)

How it works:
- Cuts the **original** on-the-fly using your **EDL** (no render), scales both streams to the same size & FPS, then computes:
  - **VMAF** (if FFmpeg has `libvmaf`)
  - **SSIM**
  - **PSNR**
- Saves a text report next to your **edited file**: `*quality_report.txt`, including quick interpretation thresholds.

Interpretation (rule of thumb):
- **VMAF**: 95–100 ≈ indistinguishable; 85–95 good; 80–85 fair; <80 visible degradation likely.
- **SSIM (All)**: ≥0.98 excellent; 0.95–0.98 good; 0.90–0.95 fair; <0.90 poor.
- **PSNR (avg)**: ≥40 dB excellent; 35–40 good; 30–35 fair; <30 poor.

---

## Typical Workflow

1. **Detect**: Run Option 1 to get `*scenes.edl`.
2. **Fine-tune & re-arrange timeline**: Import the EDL into your NLE (Resolve: **File → Import → Timeline → Pre-Conformed EDL**), link to the source, rearrange/trim as needed.
3. **Build final**: Use Option 2 with your **original video + final EDL** to produce the edited file (NVENC or x264).
4. **Compare** (optional): Use Option 3 to get **VMAF/SSIM/PSNR** of the result vs the original **filtered by the same EDL**.

---

## Guide for in-DaVinci-Resolve editing (key-shortcuts, menus and instructions)

### Import the pre-conformed EDL
1. **Menu:** `File → Import → Timeline → Pre-Conformed EDL`
2. Choose the EDL exported from Option 1 (CMX3600, NON-DROP).
3. When prompted, **link to the source file** (your original video).
4. Verify frame rate and timecode mode match your source.

### Keep A/V locked together when moving clips
- **Linked Selection** on → linked video+audio move as one. *(Default toggle: `Ctrl+Shift+L` on Windows; check your keymap.)*
- Use **Auto Select** toggles (the arrow icons per track) to decide which tracks **ripple** (shift) and which stay frozen during insert/delete.

### Move a clip elsewhere and close gaps (Ripple move)
1. Select the clip(s) you want to relocate.
2. **Ripple Cut:** `Ctrl+Shift+X`  → removes selection and **closes the gap**.
3. Park the playhead at the new spot.
4. **Paste Insert:** `Ctrl+Shift+V` → inserts the clip and **pushes later clips to the right** on Auto-Selected tracks.

> Tip: If you don’t want certain tracks to move (e.g., music), **turn off Auto Select** on those tracks before the ripple.

### Blade vs precision trimming (frame-level)
- **Blade tool**:  
  - `B` = Blade; `A` = Selection tool  
  - `Ctrl+B` = Blade **at playhead** (cuts across any tracks with Auto Select enabled)
- **Precision trims (keyboard-first)**:  
  1) `T` = **Trim Edit Mode**  
  2) `V` = **Select nearest edit point** (the cut)  
  3) `U` = Cycle trim type (**Trim In**, **Trim Out**, **Roll**)  
  4) `,` / `.` = Nudge selected edit **±1 frame**; `Shift+,` / `Shift+.` = **±5 frames** (default)
- **To/From playhead trims**:  
  - `Shift+[`: **Trim Start** to playhead (non-ripple)  
  - `Shift+]`: **Trim End** to playhead (non-ripple)  
  - `Ctrl+Shift+[` / `Ctrl+Shift+]`: **Ripple** Trim Start/End to playhead (closes/opens timeline)

> Roll trims keep overall timeline length; ripple trims change it.

### Avoid unintended edits
- If a cut or insert shifts “everything,” you likely had **Auto Select enabled** on more tracks than intended → toggle off on tracks you want untouched.
- If `Ctrl+B` blades too many tracks, disable Auto Select on tracks you don’t want cut.

### Optional: quick visual QA
- Stack **edited** over **original** on the timeline and set the top clip’s **Composite Mode** to **Difference** to eyeball changes; use scopes for quantifying luma/chroma differences.

---

## Notes & Gotchas

- **GPU use**
  - Option 1: only the FFmpeg fallback path uses **CUDA decode**; PySceneDetect remains CPU.
  - Option 2: **NVENC** for encode (when enabled), else **libx264** CPU.
  - Option 3: **CUDA** for decode/scale; metric math (VMAF/SSIM/PSNR) is **CPU**.
- **VMAF on Windows**
  - Use an FFmpeg build with **`libvmaf`** and ensure a **JSON model** is available.
- **EDL format**
  - Script writes **CMX3600 / NON-DROP FCM**.
- **Temp working folder** (Option 2)
  - Using a fast NVMe can greatly speed up segment encodes and concat. You can auto-delete it after success.

---

## Changelog (since v1.x)

- Removed fragile Resolve-API automation; script switched to a Resolve-free processes approach, with manual EDL import/export in Resolve.
- Added **GPU toggles per option** (CUDA/NVENC where it helps).
- Added **Option 2**: H.264-normalized, concat-safe rebuild from EDL (temp folder & auto-cleanup prompts).
- Added **Option 3**: **EDL-aligned** quality comparison (VMAF/SSIM/PSNR) with TXT report & interpretation guide.

---

## License
MIT

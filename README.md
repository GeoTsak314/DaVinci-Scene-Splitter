# Z-3 Scene Splitter for DaVinci Resolve v1.5 (by George Tsakalos)

This Python tool automates **scene detection** on large video files and
builds a **pre-sliced DaVinci Resolve timeline** with frame-accurate
cuts.

👉 No re-encoding, no quality loss: the script only creates **virtual
cuts** on the timeline. You can freely rearrange, trim, or export the
final video inside Resolve.

------------------------------------------------------------------------

## Features

-   Detects **scene changes** automatically using:
    -   [PySceneDetect](https://pyscenedetect.readthedocs.io/)
        (OpenCV-based, recommended)
    -   or fallback to **FFmpeg scene detection** (fast, lightweight)
    -   DaVinci Resolve has build-in detect mech, but available only
        in the paid version (Studio).
-   Outputs:
    -   `*_scenes.csv` -- list of cut times with seconds, timecodes, and
        frame numbers
    -   `*_cuts.txt` -- plain list of cut frame numbers
    -   `*_scenes.edl` -- CMX3600 EDL you can import into any NLE
        (Resolve, Premiere, etc.)
-   **Resolve integration**:
    -   Optionally creates a new Resolve timeline with the video
        **pre-split into clips** at each detected cut.

------------------------------------------------------------------------

## Requirements

-   Windows 10\

-   Python 3.9+\

-   [FFmpeg](https://ffmpeg.org/) & ffprobe in PATH\

-   DaVinci Resolve (Free or Studio)

    -   Resolve must already include its scripting module
        (`DaVinciResolveScript` / `fusionscript`).\

-   Python packages:

    ``` bash
    pip install -r requirements.txt
    ```

### requirements.txt

    scenedetect>=0.6
    opencv-python>=4.8

------------------------------------------------------------------------

## Installation

1.  Copy files:

    -   `_SceneSplitter.py`
    -   `requirements.txt`
    -   `README.md` (this file)

2.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

3.  Ensure `ffmpeg` and `ffprobe` work from your terminal:

    ``` bash
    ffmpeg -version
    ffprobe -version
    ```

4.  Make sure Resolve scripting is enabled:

    -   Script looks for Resolve at:

            C:\DaVinci\Resolve.exe

        (edit inside `_SceneSplitter.py` if needed)

------------------------------------------------------------------------

## Usage

Run:

``` bash
python _SceneSplitter.py
```

You'll see a menu:

### Option 1 -- Detect Scenes

-   Prompts you for:
    -   **Video path**
    -   **PySceneDetect threshold** (default: 27.0, typical range:
        25--35.\
        Lower = more cuts, higher = fewer)
    -   **FFmpeg scene threshold** (default: 0.30, typical range:
        0.20--0.50.\
        Lower = more cuts)
-   Produces three files in the same folder:
    -   `video_scenes.csv`
    -   `video_cuts.txt`
    -   `video_scenes.edl`

### Option 2 -- Apply to Resolve

-   Prompts you for:
    -   **Video path**
    -   **Cuts file** (`.csv` or `.txt` from Option 1)
-   Connects to Resolve:
    -   Ensures the video is imported into the Media Pool
    -   Creates a new timeline:\
        `SceneSplit - <videoname>`
    -   Appends the clip segments based on the cut frames
        (frame-accurate)

------------------------------------------------------------------------

## Notes

-   **Frame accuracy (hybrid method = minimal loss & best speed)**:\
    Cuts are quantized to the nearest frame (from ffprobe FPS).
    Resolve's timeline guarantees frame-accurate editing.

-   **If timeline append fails**:\
    Some Resolve builds don't support frame dicts in
    `AppendToTimeline`.\
    In that case, just import the generated **EDL**:

    -   Add the clip to Media Pool
    -   `File → Import → Timeline → Pre-conformed EDL`
    -   Select the generated `*_scenes.edl`
    -   Link to the same source clip → You'll get the cut-up timeline
        instantly.

-   **No quality loss**:\
    No intermediate encoding happens. You only export your final
    rearranged edit from Resolve.

------------------------------------------------------------------------

## Example Workflow

1.  Detect cuts:

    ``` bash
    python _SceneSplitter.py
    # Choose option 1
    # Enter video path: D:\Videos\movie.mkv
    # Thresholds: 27.0 (PySceneDetect), 0.30 (FFmpeg fallback)
    ```

2.  Apply cuts in Resolve:

    ``` bash
    python _SceneSplitter.py
    # Choose option 2
    # Enter video path: D:\Videos\movie.mkv
    # Enter cuts file: D:\Videos\movie_scenes.csv
    ```

3.  In Resolve:

    -   Timeline "SceneSplit -- movie.mkv" will contain the pre-split
        clips.\
    -   Rearrange / trim / edit freely.\
    -   Export the final video as usual.

------------------------------------------------------------------------

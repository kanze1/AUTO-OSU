# Converter v0.2

The converter uses a two-column workspace: song/folder queue on the left, difficulties and GPU setup on the right, with progress and actions fixed below. The avatar, executable icon and cover share the user's orange, graphite and cyan character reference. Chinese/English and light/dark modes remain available.

## Managed GPU runtime

The frozen application keeps its bundled CPU environment. `autoosu.runtime` creates a separate uv-managed Python 3.12 environment and sends inference requests to `autoosu.worker` using JSON lines. The installer uses `uv pip install --torch-backend auto` to select PyTorch for the installed NVIDIA driver. It tests an actual CUDA tensor operation before atomically activating the environment. A failed or cancelled installation preserves the previous active runtime.

The Windows runtime root is `%LOCALAPPDATA%\AUTO-OSU\runtime`. `AUTOOSU_RUNTIME_HOME` can override it. The installer stores uv under `tools`, managed Python under `python`, environments under `envs`, and the active environment in `active.json`. Reconfiguration stages a new environment. It does not remove previous environments automatically. Model files remain in the app's existing model locations.

The executable bundles matching inference source in `runtime/runtime-source.zip`, so the installed worker does not depend on downloading AUTO-OSU from a package index or on the user retaining a repository checkout. Source checkouts use an editable installation. Python paths inherited from another environment are removed from child processes; frozen Windows DLL search paths are reset while starting the worker.

The installer requires network access and an installed NVIDIA driver. It does not install GPU drivers. First-time dependencies occupy several GB. If uv encounters its Windows minor-version junction issue, setup uses the downloaded patch-version interpreter directly.

uv reports package download start/completion events in the piped GUI log, but terminal progress bars are disabled. `Built ...runtime-source.zip` means only the app's wheel has finished building; concurrent dependency downloads can still be pending. The current source installer adds a localized elapsed/waiting notice every 15 seconds without new logs. Python environment creation and dependency installation each have a one-hour wall-clock limit, including time spent downloading; this is not a claim that a quiet download has stalled. Cancellation and the deadline remain active after output EOF. See the [runtime setup regression evidence](runtime-setup-stall.md).

References: [uv's PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/), [uv installation](https://docs.astral.sh/uv/getting-started/installation/), [Windows Python junction issue](https://github.com/astral-sh/uv/issues/19622).

## Folder behavior

Files are processed sequentially, with rhythm and coordinate models reused within the batch. Output directory names contain the input stem and a hash of its relative path. Each report is written atomically after each file. An unreadable input fails only that item. Cancellation completes the current song and marks pending items as cancelled. The scanner excludes nested output directories and hidden directories, and does not follow directory symlinks.

## Cover generation

Created with the built-in ImageGen tool from the user's supplied character image. Output: `autoosu/assets/cover.png`. The avatar and icon retain the supplied artwork. Text is rendered by the interface, rather than baked into the cover.

Exact prompt:

> Use case: identity-preserve. Asset type: wide 3:1 illustrated cover banner for the AUTO-OSU desktop music-to-beatmap converter. Use the attached image as the exact character identity and drawing-style reference. Preserve the short dark navy bob haircut, amber eye, cyan glowing cheek mark, black outfit, left-facing profile, and recognizable face. Recompose into a refined wide banner: the character occupies the far right 35 percent of the frame, looking left into calm negative space. The left 60 percent is nearly solid deep charcoal black (#11151c), with a very subtle warm edge where it meets the saturated orange background behind the character. Keep a generous uncrowded area on the left for real UI text added later. Chest-up portrait; head fully visible, not clipped. A few extremely subtle thin circular rhythm-game motifs may sit behind the character but do not obscure the face. Mood: precise, calm, contemporary, sharp anime line work. Palette: charcoal, warm orange, one tiny cyan cheek accent. No text, no lettering, no logos, no watermarks, no UI frames, no fake controls. Wide landscape image, approximately 3:1 aspect ratio.

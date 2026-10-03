# 0.3.0rc3: GPU setup waiting and cancellation

This Windows preview includes the runtime setup fix from [PR #8](https://github.com/kanze1/AUTO-OSU/pull/8), squash-merged as `23bc3b3`. [Download 0.3.0rc3](https://github.com/kanze1/AUTO-OSU/releases/tag/v0.3.0rc3). The public stable release remains 0.2.0; experimental controls retain their existing labels and limitations. Both model weights remain v0.

## 中文

`Built runtime-source.zip` 表示源码包构建完成，其余 GPU 依赖可能还在下载或解压。本版在没有新日志时每 15 秒显示等待状态，修复输出关闭后取消失效的问题，单个安装步骤超过 1 小时会停止并显示原因。失败或取消保留原来可用的环境。

下载 `AUTO-OSU-0.3.0rc3-win64-cpu.zip`，解压后运行其中的 `AUTO-OSU/AUTO-OSU.exe`。CPU 可直接使用；需要 GPU 时点击「自动配置 GPU 加速」。GPU 环境需要匹配应用版本，旧版用户升级后需要重新配置，可复用已有下载缓存。

这是 0.3 系列预览版，包含此前候选版的来源检查、实测星级、highlight 和谱面偏好功能；高星控制、自动段落位置及谱面偏好仍为实验功能。

## Packaged application verification

Validation date: 2026-10-04. Built with Python 3.13.2, PyInstaller 6.22.3 and CPU PyTorch 2.14.0. Separate output/work directories preserved previous builds.

- **172 local tests passed** on the complete Python 3.12 project environment; the repository/documentation check and diff whitespace check passed. Required Windows/Ubuntu CI is checked on the release PR before merging.
- The actual Windows EXE launched its GUI, displayed **0.3.0rc3**, and closed normally with exit code 0. The source GUI's bilingual waiting/cancellation tests are recorded in the [fix evidence](runtime-setup-stall.md).
- All **50 bundled runtime-source files** matched the release source bytes, and both packaged v0 model files matched their declared SHA-256 values.
- The EXE's `--setup-runtime` command created an isolated fresh runtime, downloaded managed Python 3.12.14 (21 MiB), installed 39 packages, ran a CUDA tensor calculation and activated only that scratch runtime. It reused uv 0.12.17 and dependency caches; setup completed in 32.0 seconds. The user's existing active manifest remained byte-for-byte unchanged.
- Actual EXE inference succeeded on CPU and, after setup, through the managed GPU runtime on an RTX 4090 with PyTorch `2.14.1+cu132` / CUDA 13.2. Both outputs passed structural checks, contained the content watermark, and matched their own isolated local record stores.

Frozen inference smoke settings: the first existing Q1 development song (`f7471b6a48aedbc43a9b504cf783aab6cffa972b69555231a6ddaa85fe229b51`), seed `240924`, Insane, reference BPM `164`, CLI offset `733 ms`, model-star condition `6.5`, 12 rhythm steps and **10 coordinate steps**. Balanced preference and automatic highlights were used. CPU/GPU measurements were 5.20 / 5.43 stars. This checks packaged execution, not new model quality, CPU/GPU output equivalence or human gameplay quality.

The release includes the Windows ZIP, its SHA-256 file and a compact `verification.json` with exact executable/runtime hashes, source revision, required checks and artifact checks. Third-party songs, generated maps, virtual environments and test runtime installations are excluded from the application archive. No osu! client gameplay or editor-resave verification is claimed.

## English

Version 0.3.0rc3 makes quiet GPU setup waits visible every 15 seconds, keeps cancellation responsive after installer output closes, and stops a setup command after a one-hour wall-clock limit. `Built runtime-source.zip` only means the app's wheel has finished building; other dependencies can still be downloading or unpacking. Failed or cancelled setup preserves the previous working runtime.

Extract the Windows ZIP and run `AUTO-OSU/AUTO-OSU.exe`. CPU inference works directly; use **Set up GPU acceleration** for an NVIDIA GPU. Upgrades require a runtime matching the new app version; downloaded dependency caches can be reused. This is a public preview of the 0.3 series, retaining the prior candidates' features and experimental limits.

# Smart Shelf: on-device shelf monitoring that texts a manager before you run out of stock

A USB camera watches a shelf. A Qualcomm Dragonwing EVK counts snack packets on-device, on its NPU, in real time. When the count drops, an AI agent with cross-session memory decides whether this is a normal fluctuation or a recurring restocking pattern — and if it's serious enough, texts a store manager on Telegram and flags a vendor to call.

**Hardware:** Qualcomm Dragonwing IQ-9075 EVK (36 GB RAM, Hexagon NPU) + USB webcam
**Difficulty:** Intermediate
**Stack:** Python (OpenCV, `edge_impulse_linux`) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) (Nous Research) · Telegram Bot API

Object detection is genuinely on-device and NPU-accelerated. Reasoning runs through Hermes Agent against a cloud-hosted model — a deliberate tradeoff explained in [Known limitations](#known-limitations) below, after an attempt to host the reasoning LLM on-device too ran into compatibility issues.

## What you'll build

- A live object-detection pipeline on the EVK's NPU that counts snack packets (Cheetos, Lays, Popcorners) per frame, debounced so a hand passing through the frame doesn't register as a restock or removal.
- An agent (Hermes) that reasons over that history — with real persistent memory — to spot recurring restocking patterns and decide when a manager alert is actually warranted.
- Manager alerts delivered straight to Telegram, plus a scheduled job that proactively warns of a likely shortage before it happens.

## Before you start

You'll need:

- A **Dragonwing IQ-9075 EVK**, with Qualcomm's **AI Runtime SDK** installed (the Qualcomm IoT PPA — `ppa:ubuntu-qcom-iot/qcom-ppa` — needs to be registered before `libqnn1`, `libsnpe1`, and their `-dev` packages will install; see Edge Impulse's board-specific IQ-9075 setup guide).
- A **USB webcam** pointed at the shelf.
- Python 3 with the Edge Impulse Linux SDK: `pip install edge_impulse_linux --break-system-packages`, plus `opencv-python` and `requests`.
- **[Hermes Agent](https://github.com/NousResearch/hermes-agent)** installed and configured, with the `hermes` CLI on your `PATH`.
- A **Telegram bot** — message [@BotFather](https://t.me/BotFather) on Telegram to create one and get a bot token, and get your own chat ID (e.g. via [@userinfobot](https://t.me/userinfobot)).
- Your own trained Edge Impulse object-detection (FOMO) model, compiled for this board as a QNN-accelerated `.eim` binary — `smart_shelf.eim` in this repo is one such model, trained on Cheetos/Lays/Popcorners. Retrain it on your own products via Edge Impulse Studio if you're monitoring something else.

## Repo contents

| File | Role |
|---|---|
| `smart_shelf.py` | Main pipeline — camera capture, on-device NPU inference, debouncing, Hermes + Telegram integration. Run this. |
| `history.py` | Seeds three weeks of fabricated-but-consistent demo history into Hermes's memory, so pattern detection has something to reason over from the very first run. |
| `smart_shelf.eim` | Compiled, QNN-accelerated Edge Impulse model (FOMO object detector, 3 classes, 96×96 input). |

## Step 1: configure it

Open `smart_shelf.py` and fill in the placeholders near the top:

```python
TELEGRAM_BOT_TOKEN = ""  # put your Telegram bot token here (from @BotFather)
TELEGRAM_CHAT_ID = ""    # put your Telegram chat ID here
...
VENDOR_NAME = "Your Vendor Name"       # put your snack vendor's name here
VENDOR_CONTACT = "Contact Name — (555) 000-0000 — orders@example.com"  # put your vendor's contact info here
```

Also check `MODEL_PATH` (defaults to `./smart_shelf.eim`), `CAMERA_DEVICE_ID` (defaults to `0`), and `SHELF_ID`, `FLAVOR_LABELS` against your own setup.

## Step 2: seed some history (optional, but recommended for a first demo)

Pattern detection is more convincing with a few weeks of history behind it. Run once, before your first live demo:

```bash
python3 history.py
```

This feeds three weeks of fabricated-but-internally-consistent shelf events into Hermes's memory via the CLI — enough for genuine weekly patterns (a Lays low-stock pattern on Wednesday evenings, a Cheetos one on Friday evenings) to show up in reasoning, regardless of which real day you actually run the demo.

## Step 3: run it

```bash
python3 smart_shelf.py
```

Point the camera at the shelf and just remove packets by hand — detection, debouncing, and alerting are all automatic. `Ctrl+C` to quit.

## How it decides when to alert

1. The NPU model returns a bounding box per visible packet per frame. Overlapping same-label boxes (a known FOMO grid-cell artifact) are deduplicated by IoU.
2. A count is only trusted once it's held steady across `STABLE_FRAMES` (15) consecutive frames, with a minimum interval between accepted state changes per flavor — this filters out a hand or shadow passing through without meaningfully delaying a real removal.
3. Once a removal is confirmed, it's handed off to a single background worker thread, which calls `hermes chat` as a subprocess. Hermes reasons over its own persistent memory, checks `backroom_inventory.csv` if the shelf is critically low, and decides whether to alert the manager and/or recommend contacting the vendor.
4. Alerts are sent directly via the Telegram Bot API. A separate scheduled job (cron) can independently check memory for known weekly patterns and warn proactively, before a live camera event even happens.

## Performance

| Metric | Result |
|---|---|
| On-device NPU inference (dsp + classification) | ~1.0 ms average (~1000 FPS), reported directly by Edge Impulse's own per-frame timing |
| Hermes reasoning round-trip latency | 34.6s–88.6s observed (avg ~57.7s in one run) — grows as memory content grows |

## Troubleshooting / things that came up during development

**`error while loading shared libraries: libQnnTFLiteDelegate.so: cannot open shared object file`.** The Qualcomm IoT PPA isn't registered yet — add it (`ppa:ubuntu-qcom-iot/qcom-ppa`), then install `libqnn1`, `libsnpe1`, and their `-dev` packages via `apt`. This is a device-setup step, documented in Edge Impulse's own IQ-9075 guide.

**Not sure the model is actually running on the NPU, not silently falling back to CPU?** Check the compiled model's own `--print-info` metadata — it reports `"engine_type": 4, "properties": ["qnn_delegates"]` directly when genuinely NPU-accelerated. This isn't something the surrounding Python code can misreport.

**No detections at all, even though the camera and model load fine.** Check `info['model_parameters'].get('model_type')` — a FOMO object-detection model returns a `bounding_boxes` list, not a flat classification score. Code that assumes classification output will silently find nothing.

**One physical packet gets counted as two.** FOMO's grid-cell-based detection can fire on adjacent cells for the same object. Deduplicate overlapping same-label boxes by IoU, keeping the higher-confidence box.

**Counts oscillate rapidly with no real shelf change.** Add a temporal-stability requirement — only trust a count once it's held for several consecutive frames — plus a minimum interval between accepted changes per flavor.

**Live camera preview looks low-resolution/blocky.** The SDK's own `classifier()` generator hands back the model's internal input frame (96×96), not a full-resolution camera frame. Take ownership of the camera yourself with a single `cv2.VideoCapture`, and call the SDK's lower-level `get_features_from_image_auto_studio_settings()` + `classify()` methods manually per frame instead — this gives full-resolution frames for display while still running inference through the same model. (A USB webcam only supports one open connection at a time, so opening a second capture handle purely for display won't work.)

**A Telegram webhook signature check fails with `401 Invalid signature` despite a correct shared secret.** Hermes's generic/legacy webhook route expects a bare hex digest in `X-Webhook-Signature`, with no `sha256=` prefix — different from GitHub's convention. Drop the prefix.

**Webhook-triggered agent runs report no memory tool or Telegram-send capability, even though the CLI works fine.** This is a documented security default — Hermes restricts webhook-originated sessions to a safe tool subset, since webhook payloads may come from untrusted third parties. If you need full tool access per event, consider calling `hermes chat` directly as a subprocess for that event instead of going through the webhook platform (this is what `smart_shelf.py` does).

**A memory entry gets garbled after rapid testing.** Likely a race condition — if your monitoring script spawns a new thread per detected event and each one calls `hermes chat` concurrently, you can get overlapping non-atomic read-modify-write calls against the same memory file. Serialize all memory-writing calls through a single persistent background worker thread instead (this is what `smart_shelf.py` does).

**Conversational (Telegram Q&A) questions miss data the automated pipeline clearly has.** Likely a retrieval-quality issue, not a memory bug — vague natural-language questions may not match well against a dense memory file. Ask more specifically (name the flavor and shelf explicitly) and it should resolve.

## Known limitations

- **Reasoning runs in the cloud, not on the NPU.** An attempt to host the reasoning LLM on-device via Qualcomm's GenieX runtime hit real compatibility issues with Hermes Agent (streaming response schema mismatches, then a colon-handling bug in Hermes's model-ID resolution). Given time constraints, this was abandoned in favor of a reliable cloud-hosted model via Nous Portal. Only the vision stage is genuinely fully on-device end-to-end.
- **Reasoning latency scales with memory size** — there's no summarization or pruning yet, so latency grows as history accumulates.
- **Conversational memory retrieval is less reliable than the automated pipeline's.** Specific, well-scoped questions work; vague ones can miss relevant data.
- **Not designed for heavily occluded or deeply stacked items** — a structural limitation of the lightweight FOMO architecture (grid-cell-based detection), not a labeling or training-data problem.
- **Vendor-escalation logic is simplified** — it always recommends contacting the vendor when conditions are met, rather than tracking contact state across episodes, to keep behavior predictable for demos.

## Future improvements

- Load-cell (weight sensor) fusion for genuine stacked-item counting, with vision retained for SKU/flavor ID.
- Context summarization / rolling-window pruning for Hermes's memory reasoning, to keep latency flat over time.
- Real vendor integration — an actual outbound email or ordering-API call, rather than a message recommending a phone call.
- Bake a condensed pattern summary into the agent's persona file, so conversational queries are as reliable as the automated pipeline.
- Revisit on-device LLM hosting once Hermes's streaming/model-ID handling issues are resolved upstream.
- Multi-shelf, multi-store rollout with a consolidated manager dashboard.

## Sources consulted

- Edge Impulse's official Qualcomm deployment documentation, including the IQ-9075-specific setup guide.
- Qualcomm's AI Runtime SDK (Community Edition) installation instructions, including the IoT PPA setup for the IQ-9075.
- The `edge_impulse_linux` Python SDK's own installed package, introspected directly (`dir()`, `inspect.signature()`) to confirm exact method names and parameters.
- Hermes Agent's official documentation (config reference, webhook/gateway docs, memory/compression subsystem docs) and its public GitHub repository.
- A publicly filed GitHub issue against Hermes Agent describing the same streaming-compatibility failure encountered with custom OpenAI-compatible endpoints.

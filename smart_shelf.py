#!/usr/bin/env python3
"""
shelf_inventory_monitor.py — FULLY AUTOMATIC shelf inventory tracking.

Your model is object detection (FOMO), which returns a separate bounding
box for EVERY packet visible in frame, per flavor. This script counts
boxes per flavor every frame, and when a flavor's stable count drops
(debounced across several frames so a hand passing through doesn't
false-trigger), it treats that as a real removal — completely hands-off,
no keypress required.

Usage:
    python3 shelf_inventory_monitor.py
    (just point the camera at the shelf and physically remove packets —
    everything else is automatic. Ctrl+C to quit.)

Requires:
    pip install edge_impulse_linux --break-system-packages
"""
import json
import os
import queue
import subprocess
import threading
import time
from collections import Counter, deque

import cv2
from edge_impulse_linux.image import ImageImpulseRunner
import requests

# ---- Configuration ------------------------------------------------------
MODEL_PATH = "./smart_shelf.eim"
CAMERA_DEVICE_ID = 0

SHELF_ID = "A3"
STATE_FILE = "inventory_state.json"
FLAVOR_LABELS = ("cheetos", "lays", "popcorners")
FLAVOR_NAMES = {"cheetos": "Cheetos", "lays": "Lays", "popcorners": "Popcorners"}

CONFIDENCE_THRESHOLD = 0.65    # per-box detection confidence to count it at all (raised from 0.5)
STABLE_FRAMES = 15             # require this many consecutive frames agreeing
                                # on a count before trusting it (raised from 8 —
                                # 1-packet setups need more certainty to avoid flicker)
MIN_EVENT_INTERVAL = 4.0       # seconds — ignore new events for a label faster than this,
                                # prevents rapid-fire oscillation from being reported repeatedly

TELEGRAM_BOT_TOKEN = ""  # put your Telegram bot token here (from @BotFather)
TELEGRAM_CHAT_ID = ""    # put your Telegram chat ID here

BACKROOM_CSV = "backroom_inventory.csv"
REORDER_THRESHOLD = 10  # if backroom stock is below this, escalate to vendor

VENDOR_NAME = "Your Vendor Name"       # put your snack vendor's name here
VENDOR_CONTACT = "Contact Name — (555) 000-0000 — orders@example.com"  # put your vendor's contact info here

MAX_TELEGRAM_CHARS = 320  # hard safety cap — even with prompt instructions,
                           # models don't always self-limit length reliably
# ---------------------------------------------------------------------------


def load_counts():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}  # empty = "not yet established", first stable reading sets baseline


def save_counts(counts):
    with open(STATE_FILE, "w") as f:
        json.dump(counts, f, indent=2)


def load_backroom_stock(flavor_key: str) -> int:
    """Reads the backroom CSV fresh each time (so manual edits between
    demo runs — e.g. simulating a restock — are picked up automatically)."""
    if not os.path.exists(BACKROOM_CSV):
        return -1  # sentinel: file missing, don't claim to know backroom stock
    with open(BACKROOM_CSV) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[0].strip().lower() == flavor_key.lower():
                try:
                    return int(parts[1])
                except ValueError:
                    return -1
    return -1


def send_telegram(text: str):
    if len(text) > MAX_TELEGRAM_CHARS:
        text = text[:MAX_TELEGRAM_CHARS - 3].rstrip() + "..."
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=15)
        print(f"    -> Telegram send status: {resp.status_code}"
              + ("" if resp.status_code == 200 else f" — {resp.text[:200]}"))
    except requests.RequestException as e:
        print(f"    -> Telegram send failed: {e}")


def ask_hermes(flavor_name: str, remaining: int, day_time: str):
    """Returns (response_text, elapsed_seconds) — timing is measured around
    the actual subprocess call, i.e. real end-to-end latency including
    network + model reasoning time, not an estimate."""
    backroom_count = load_backroom_stock(flavor_name.lower())
    backroom_line = (
        f"Backroom stock for this flavor: {backroom_count} units."
        if backroom_count >= 0 else
        "Backroom stock: unknown (inventory file unavailable)."
    )

    prompt = (
        f"Shelf inventory event. Shelf: {SHELF_ID}. Flavor removed: {flavor_name}. "
        f"Remaining on shelf: {remaining}. Day/time: {day_time}. {backroom_line}\n\n"
        f"Log this event to memory with the day and time. Check memory for whether "
        f"this flavor has hit low stock (1 or fewer remaining) at a similar day/time "
        f"before, and note briefly if a pattern is emerging.\n\n"
        f"If shelf remaining is <= 1:\n"
        f"- If backroom stock is below {REORDER_THRESHOLD} and above 0: recommend "
        f"contacting the vendor and include this contact info exactly once: "
        f"{VENDOR_NAME}, {VENDOR_CONTACT}.\n"
        f"- If backroom stock is 0: recommend putting up a temporary out-of-stock "
        f"sign for this flavor instead of contacting the vendor.\n"
        f"- If backroom stock is at or above {REORDER_THRESHOLD}: no vendor contact "
        f"needed, backroom can cover a restock.\n\n"
        f"CRITICAL FORMAT REQUIREMENT: your reply is sent directly as a Telegram "
        f"message to a busy store manager. Keep it to 2-3 short sentences MAXIMUM, "
        f"under 300 characters total. No headers, no bullet lists, no markdown "
        f"formatting, no restating the raw data back. Just the essential status and "
        f"the one action needed, in plain conversational text."
    )
    start = time.time()
    result = subprocess.run(
        ["hermes", "chat", "--quiet", "-q", prompt],
        capture_output=True, text=True, timeout=300,
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"    -> hermes chat error: {result.stderr[:500]}")
        return "", elapsed
    return result.stdout.strip(), elapsed


def dedupe_boxes(boxes, iou_threshold=0.3):
    """FOMO-style detectors can fire on adjacent grid cells for the same
    physical object, producing two overlapping boxes with the same label.
    This collapses overlapping same-label boxes into one before counting."""
    def box_iou(a, b):
        ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["width"], a["y"] + a["height"]
        bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area_a = a["width"] * a["height"]
        area_b = b["width"] * b["height"]
        return intersection / float(area_a + area_b - intersection)

    boxes_sorted = sorted(boxes, key=lambda b: -b["value"])
    kept = []
    for box in boxes_sorted:
        if any(box["label"] == k["label"] and box_iou(box, k) > iou_threshold for k in kept):
            continue  # overlaps a higher-confidence box of the same label — same object
        kept.append(box)
    return kept


def counts_from_boxes(boxes):
    """Count confident detections per label in a single frame, after
    deduplicating overlapping boxes on the same physical object."""
    boxes = dedupe_boxes(boxes)
    counts = Counter()
    for box in boxes:
        if box["value"] >= CONFIDENCE_THRESHOLD:
            counts[box["label"]] += 1
    return counts


def draw_overlay_fullres(display_frame, boxes, confirmed_counts, model_w, model_h, inference_stats):
    """Draws bounding boxes + a running count readout onto a FULL-RESOLUTION
    frame we captured ourselves, scaling box coordinates (given in the
    model's small input space) up to the real display resolution. Note:
    Edge Impulse's auto-studio preprocessing may center-crop before
    resizing, so this scaling is an approximation, not pixel-perfect —
    fine for a demo overlay, not for precision measurement."""
    frame = display_frame.copy()
    box_color = {
        "cheetos": (0, 165, 255),
        "lays": (0, 200, 0),
        "popcorners": (255, 150, 0),
    }
    boxes = dedupe_boxes(boxes)

    scale_x = frame.shape[1] / model_w
    scale_y = frame.shape[0] / model_h

    for box in boxes:
        if box["value"] < CONFIDENCE_THRESHOLD:
            continue
        x = int(box["x"] * scale_x)
        y = int(box["y"] * scale_y)
        w = int(box["width"] * scale_x)
        h = int(box["height"] * scale_y)
        color = box_color.get(box["label"], (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        label_text = f"{FLAVOR_NAMES.get(box['label'], box['label'])} {box['value']:.2f}"
        cv2.putText(frame, label_text, (x, max(y - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    y0 = 30
    for label in FLAVOR_LABELS:
        count = confirmed_counts.get(label, "?")
        text = f"{FLAVOR_NAMES[label]}: {count}"
        cv2.putText(frame, text, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    box_color[label], 2, cv2.LINE_AA)
        y0 += 32

    # Engineering stats readout — real on-device inference timing reported
    # by Edge Impulse itself (dsp + classification time), not estimated.
    if inference_stats["avg_ms"] is not None:
        stats_text = f"NPU inference: {inference_stats['avg_ms']:.1f}ms avg  |  {inference_stats['fps']:.1f} FPS"
        h = frame.shape[0]
        cv2.rectangle(frame, (0, h - 34), (frame.shape[1], h), (0, 0, 0), -1)
        cv2.putText(frame, stats_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)

    return frame


def draw_overlay(img, boxes, confirmed_counts, inference_stats):
    """Draws bounding boxes + a running count readout onto the frame, returns
    a BGR image ready for cv2.imshow (the SDK gives RGB, OpenCV wants BGR)."""
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    box_color = {
        "cheetos": (0, 165, 255),    # BGR: orange
        "lays": (0, 200, 0),          # green
        "popcorners": (255, 150, 0),  # blue
    }
    boxes = dedupe_boxes(boxes)  # match what's actually being counted

    for box in boxes:
        if box["value"] < CONFIDENCE_THRESHOLD:
            continue
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        color = box_color.get(box["label"], (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label_text = f"{FLAVOR_NAMES.get(box['label'], box['label'])} {box['value']:.2f}"
        cv2.putText(frame, label_text, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    y0 = 20
    for label in FLAVOR_LABELS:
        count = confirmed_counts.get(label, "?")
        text = f"{FLAVOR_NAMES[label]}: {count}"
        cv2.putText(frame, text, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    box_color[label], 1, cv2.LINE_AA)
        y0 += 20

    upscaled = cv2.resize(frame, (frame.shape[1] * 5, frame.shape[0] * 5), interpolation=cv2.INTER_CUBIC)

    # Engineering stats readout, bottom-left — real on-device inference timing
    # from Edge Impulse's own reported dsp+classification time, not estimated.
    if inference_stats["avg_ms"] is not None:
        stats_text = f"NPU inference: {inference_stats['avg_ms']:.1f}ms avg  |  {inference_stats['fps']:.1f} FPS"
        h = upscaled.shape[0]
        cv2.rectangle(upscaled, (0, h - 28), (upscaled.shape[1], h), (0, 0, 0), -1)
        cv2.putText(upscaled, stats_text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)

    return upscaled


hermes_latency_log = []  # (timestamp, elapsed_seconds) — for the end-of-run summary
ei_timing_log = deque(maxlen=60)  # rolling window of Edge Impulse inference timing

# Removal events go through a QUEUE processed by a single worker thread, not
# one thread per event. Concurrent hermes chat calls each independently
# read-modify-write MEMORY.md, and overlapping writes can corrupt/merge
# entries (this actually happened — a Popcorners entry got garbled after
# rapid-fire testing). Serializing writes eliminates that race condition
# while still keeping the camera loop non-blocking (it just enqueues).
removal_queue = queue.Queue()


def hermes_worker():
    """Runs in its own background thread for the whole program's lifetime,
    processing one removal event at a time — never concurrently."""
    while True:
        flavor_name, majority_count, day_time, shelf_id = removal_queue.get()
        print(f"    -> asking Hermes to log + check pattern (queued, {removal_queue.qsize()} waiting)...")
        response, elapsed = ask_hermes(flavor_name, majority_count, day_time)
        hermes_latency_log.append(elapsed)
        print(f"    -> Hermes says ({elapsed:.1f}s): {response[:300]}")
        if majority_count <= 1:
            message = f"⚠️ {shelf_id} · {flavor_name} ({majority_count} left)\n{response}"
            send_telegram(message)
        removal_queue.task_done()


def handle_removal_async(flavor_name, majority_count, day_time, shelf_id):
    """Enqueues a removal event for the single serialized Hermes worker
    thread — replaces the old one-thread-per-event approach."""
    removal_queue.put((flavor_name, majority_count, day_time, shelf_id))


def main():
    confirmed_counts = load_counts()  # last known stable state, e.g. {"nc": 3, "sn": 3}
    last_event_time = {label: 0.0 for label in FLAVOR_LABELS}
    print(f"Starting from known state: {confirmed_counts if confirmed_counts else '(none yet — first stable reading sets baseline)'}")
    print(f"Loading model: {MODEL_PATH}")

    # Single serialized worker for all Hermes calls — see removal_queue comment above.
    threading.Thread(target=hermes_worker, daemon=True).start()

    recent_frames = {label: deque(maxlen=STABLE_FRAMES) for label in FLAVOR_LABELS}

    with ImageImpulseRunner(MODEL_PATH) as runner:
        model_info = runner.init()
        labels = model_info['model_parameters']['labels']
        model_w = model_info['model_parameters']['image_input_width']
        model_h = model_info['model_parameters']['image_input_height']
        print(f"Model loaded. Labels: {labels}")
        print("Fully automatic — point the camera at the shelf. Ctrl+C to quit.\n")

        # Own the camera ourselves at full resolution (same pattern as any
        # normal OpenCV app) — this is what actually gives a sharp display,
        # since letting the SDK manage the camera only ever hands back its
        # own tiny, already-shrunk model-input frame.
        cap = cv2.VideoCapture(CAMERA_DEVICE_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {CAMERA_DEVICE_ID}")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features, _cropped = runner.get_features_from_image_auto_studio_settings(rgb)
            res = runner.classify(features)

            # Edge Impulse reports real on-device timing per inference —
            # dsp (preprocessing) + classification (actual NPU inference).
            timing = res.get("timing", {})
            frame_ms = timing.get("dsp", 0) + timing.get("classification", 0)
            if frame_ms > 0:
                ei_timing_log.append(frame_ms)
            avg_ms = (sum(ei_timing_log) / len(ei_timing_log)) if ei_timing_log else None
            inference_stats = {
                "avg_ms": avg_ms,
                "fps": (1000.0 / avg_ms) if avg_ms else 0.0,
            }

            boxes = res["result"].get("bounding_boxes", [])
            frame_counts = counts_from_boxes(boxes)

            display_frame = draw_overlay_fullres(frame, boxes, confirmed_counts, model_w, model_h, inference_stats)
            cv2.imshow("Shelf Sentinel — live camera", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit requested from camera window.")
                break

            for label in FLAVOR_LABELS:
                recent_frames[label].append(frame_counts.get(label, 0))

                if len(recent_frames[label]) < STABLE_FRAMES:
                    continue  # not enough history yet

                # Require the count to be the same across the whole window
                # before trusting it as "stable" (not mid-grab, not a flicker)
                majority_count, agreement = Counter(recent_frames[label]).most_common(1)[0]
                if agreement < STABLE_FRAMES:
                    continue  # still fluctuating, don't act yet

                now = time.time()
                if now - last_event_time[label] < MIN_EVENT_INTERVAL:
                    continue  # too soon after the last change for this label, likely flicker

                previous = confirmed_counts.get(label)

                if previous is None:
                    # First-ever stable reading — this is the baseline, not an event
                    confirmed_counts[label] = majority_count
                    save_counts(confirmed_counts)
                    print(f"Baseline established: {FLAVOR_NAMES[label]} = {majority_count}")
                    last_event_time[label] = now
                    recent_frames[label].clear()

                elif majority_count < previous:
                    # Real removal(s) detected automatically
                    removed = previous - majority_count
                    confirmed_counts[label] = majority_count
                    save_counts(confirmed_counts)
                    flavor_name = FLAVOR_NAMES[label]
                    day_time = time.strftime("%A %H:%M")
                    print(f"\n>>> AUTO-DETECTED: {removed} {flavor_name} removed -> {majority_count} remaining ({day_time})")

                    handle_removal_async(flavor_name, majority_count, day_time, SHELF_ID)

                    last_event_time[label] = now
                    recent_frames[label].clear()

                elif majority_count > previous:
                    # Restocked — update baseline silently, no alert needed
                    confirmed_counts[label] = majority_count
                    save_counts(confirmed_counts)
                    print(f"Restock detected: {FLAVOR_NAMES[label]} now at {majority_count}")
                    last_event_time[label] = now
                    recent_frames[label].clear()

    cap.release()
    cv2.destroyAllWindows()
    print_stats_summary()


def print_stats_summary():
    print("\n" + "=" * 50)
    print("ENGINEERING STATS SUMMARY")
    print("=" * 50)
    if ei_timing_log:
        avg = sum(ei_timing_log) / len(ei_timing_log)
        print(f"Edge Impulse NPU inference: {avg:.1f}ms avg  ({1000.0/avg:.1f} FPS)  "
              f"over last {len(ei_timing_log)} frames")
    else:
        print("Edge Impulse NPU inference: no timing data captured")
    if hermes_latency_log:
        avg_h = sum(hermes_latency_log) / len(hermes_latency_log)
        print(f"Hermes round-trip latency: {avg_h:.1f}s avg over {len(hermes_latency_log)} call(s)  "
              f"(min {min(hermes_latency_log):.1f}s / max {max(hermes_latency_log):.1f}s)")
    else:
        print("Hermes round-trip latency: no calls made this run")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
        try:
            import cv2 as _cv2
            _cv2.destroyAllWindows()
        except Exception:
            pass
        print_stats_summary()
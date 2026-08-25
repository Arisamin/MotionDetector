"""Telegram notification module for real-time motion alerts and snapshot delivery."""
import os
import sys
import queue
import threading
import requests
from typing import Optional, Dict, Any, List


class TelegramNotifier:
    """Sends motion alert messages and snapshot photos to a Telegram group/chat asynchronously."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True,
    ):
        if bot_token is not None:
            self.bot_token = bot_token.strip()
        else:
            self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

        if chat_id is not None:
            self.chat_id = str(chat_id).strip()
        else:
            self.chat_id = str(os.environ.get("TELEGRAM_CHAT_ID", "")).strip()

        self.enabled = bool(enabled and self.bot_token and self.chat_id)

        # Worker thread and queue for non-blocking network delivery
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        if self.enabled:
            self._start_worker()

    def _start_worker(self):
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def _process_queue(self):
        while self._running:
            try:
                task = self._queue.get(timeout=1.0)
                if task is None:
                    break
                self._send_payload(task)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[WARN] Telegram worker error: {e}", file=sys.stderr)

    def _send_payload(self, task: Dict[str, Any]):
        caption = task.get("caption", "")
        photo_path = task.get("photo_path")

        if photo_path and os.path.exists(photo_path):
            # Send photo with caption
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            try:
                with open(photo_path, "rb") as f:
                    files = {"photo": f}
                    data = {"chat_id": self.chat_id, "caption": caption, "parse_mode": "HTML"}
                    resp = requests.post(url, data=data, files=files, timeout=10)
                    if not resp.ok:
                        print(f"[WARN] Telegram photo send failed: {resp.status_code} - {resp.text}", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Failed to send Telegram photo: {e}", file=sys.stderr)
        else:
            # Send text message
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            try:
                data = {"chat_id": self.chat_id, "text": caption, "parse_mode": "HTML"}
                resp = requests.post(url, json=data, timeout=10)
                if not resp.ok:
                    print(f"[WARN] Telegram message send failed: {resp.status_code} - {resp.text}", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Failed to send Telegram message: {e}", file=sys.stderr)

    def send_alert(self, event_data: Dict[str, Any]):
        """
        Enqueue a motion detection alert to be sent to Telegram.

        :param event_data: Event summary dict produced by MotionLogger.
        """
        if not self.enabled or not event_data:
            return

        timestamp = event_data.get("timestamp", "")
        categories = event_data.get("categories", ["motion"])
        counts = event_data.get("counts", {})
        snapshot = event_data.get("snapshot", "")
        frame_idx = event_data.get("frame_index", 0)

        # Build emoji summary
        icons = []
        if "human" in categories:
            icons.append("🚶 Human")
        if "car" in categories:
            icons.append("🚗 Vehicle")
        if "other" in categories:
            icons.append("📦 Other")

        cats_str = " | ".join(icons) if icons else "🚨 Motion"

        details = []
        if counts.get("human"):
            details.append(f"Humans: {counts['human']}")
        if counts.get("car"):
            details.append(f"Vehicles: {counts['car']}")
        if counts.get("other"):
            details.append(f"Other: {counts['other']}")
        detail_str = ", ".join(details) if details else ""

        caption = (
            f"<b>🚨 Motion Detected!</b>\n"
            f"<b>Type:</b> {cats_str}\n"
            f"<b>Counts:</b> {detail_str}\n"
            f"<b>Time:</b> <code>{timestamp}</code> (Frame #{frame_idx})"
        )

        # Enqueue task
        self._queue.put({"caption": caption, "photo_path": snapshot})

    def send_message(self, text: str):
        """Enqueue a direct text message to Telegram."""
        if self.enabled and text:
            self._queue.put({"caption": text, "photo_path": None})

    def send_photo(self, photo_path: str, caption: str = ""):
        """Enqueue a photo with optional caption to Telegram."""
        if self.enabled and photo_path:
            self._queue.put({"caption": caption, "photo_path": photo_path})

    def stop(self):
        """Stop background worker gracefully."""
        if self._running:
            self._running = False
            self._queue.put(None)
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2.0)

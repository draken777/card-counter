"""
Hi-Lo Card Counter — Screen Scanner
=====================================
Scans your screen every 2 seconds using OCR to detect card values,
updates running count and true count automatically.

Requirements:
    pip install pillow pytesseract opencv-python mss
    Also install Tesseract OCR:
        Windows: https://github.com/UB-Mannheim/tesseract/wiki
        Mac:     brew install tesseract
        Linux:   sudo apt install tesseract-ocr

Run:
    python card_counter.py
"""

import tkinter as tk
from tkinter import font as tkfont
import threading
import time
import re
import mss
import numpy as np
import cv2
from PIL import Image
import pytesseract

# ── CONFIG ──────────────────────────────────────────────────────────────────
TOTAL_DECKS   = 6
PENETRATION   = 0.75          # 0.75 = 75% of shoe dealt before reshuffle
SCAN_INTERVAL = 2             # seconds between screen scans
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows only — uncomment & set path

# ── HI-LO LOGIC ─────────────────────────────────────────────────────────────
TOTAL_CARDS   = TOTAL_DECKS * 52
SHUFFLE_POINT = int(TOTAL_CARDS * PENETRATION)

LOW_CARDS  = {'2', '3', '4', '5', '6'}
HIGH_CARDS = {'10', 'J', 'Q', 'K', 'A'}
NEUTRAL    = {'7', '8', '9'}
ALL_CARDS  = LOW_CARDS | HIGH_CARDS | NEUTRAL

def hilo_tag(card: str) -> int:
    if card in LOW_CARDS:  return +1
    if card in HIGH_CARDS: return -1
    return 0

def extract_cards_from_image(img_np: np.ndarray) -> list[str]:
    """Convert screenshot → grayscale → OCR → parse card tokens."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    # Upscale for better OCR accuracy
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    # Threshold to make text pop
    _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pil_img = Image.fromarray(thresh)
    raw = pytesseract.image_to_string(
        pil_img,
        config="--psm 6 -c tessedit_char_whitelist=0123456789AJQK"
    )
    # Parse tokens: 10, A, J, Q, K, 2-9
    tokens = re.findall(r'\b(10|[2-9]|[AJQK])\b', raw.upper())
    return tokens

# ── MAIN APP ─────────────────────────────────────────────────────────────────
class CardCounterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hi-Lo Counter")
        self.root.attributes("-topmost", True)        # always on top
        self.root.attributes("-alpha", 0.92)          # slight transparency
        self.root.resizable(False, False)
        self.root.configure(bg="#0a0a0a")

        # State
        self.running_count = 0
        self.cards_seen    = 0
        self.true_count    = 0.0
        self.seen_tokens   = set()   # deduplicate within same scan
        self.scanning      = False
        self.last_cards    = []

        self._build_ui()
        self._update_display()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        pad = dict(padx=12, pady=4)

        # Title bar
        tk.Label(self.root, text="♠ HI-LO COUNTER ♠",
                 bg="#0a0a0a", fg="#f0c040",
                 font=("Courier", 13, "bold")).pack(**pad, pady=(10,2))

        tk.Frame(self.root, bg="#333", height=1).pack(fill="x", padx=10)

        # Running count — big display
        self.rc_label = tk.Label(self.root, text="Running Count",
                                 bg="#0a0a0a", fg="#888",
                                 font=("Courier", 9))
        self.rc_label.pack(**pad, pady=(8,0))

        self.rc_value = tk.Label(self.root, text="0",
                                 bg="#0a0a0a", fg="#ffffff",
                                 font=("Courier", 48, "bold"))
        self.rc_value.pack(**pad, pady=0)

        tk.Frame(self.root, bg="#222", height=1).pack(fill="x", padx=10)

        # True count
        self.tc_label = tk.Label(self.root, text="True Count",
                                 bg="#0a0a0a", fg="#888",
                                 font=("Courier", 9))
        self.tc_label.pack(**pad, pady=(6,0))

        self.tc_value = tk.Label(self.root, text="0.00",
                                 bg="#0a0a0a", fg="#00e5ff",
                                 font=("Courier", 28, "bold"))
        self.tc_value.pack(**pad, pady=0)

        tk.Frame(self.root, bg="#222", height=1).pack(fill="x", padx=10)

        # Stats row
        stats_frame = tk.Frame(self.root, bg="#0a0a0a")
        stats_frame.pack(fill="x", padx=12, pady=6)

        self.seen_lbl = tk.Label(stats_frame, text="Cards seen: 0",
                                 bg="#0a0a0a", fg="#aaa",
                                 font=("Courier", 9))
        self.seen_lbl.pack(side="left")

        self.rem_lbl = tk.Label(stats_frame, text=f"Decks left: {TOTAL_DECKS:.1f}",
                                bg="#0a0a0a", fg="#aaa",
                                font=("Courier", 9))
        self.rem_lbl.pack(side="right")

        # Advice banner
        self.advice_lbl = tk.Label(self.root, text="Bet minimum — neutral",
                                   bg="#1a1a1a", fg="#ffffff",
                                   font=("Courier", 10, "bold"),
                                   width=28, pady=5)
        self.advice_lbl.pack(fill="x", padx=10, pady=(2,6))

        # Last detected cards
        self.cards_lbl = tk.Label(self.root, text="Last scan: —",
                                  bg="#0a0a0a", fg="#555",
                                  font=("Courier", 8),
                                  wraplength=200)
        self.cards_lbl.pack(**pad, pady=(0,4))

        tk.Frame(self.root, bg="#333", height=1).pack(fill="x", padx=10)

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#0a0a0a")
        btn_frame.pack(pady=8)

        self.scan_btn = tk.Button(btn_frame, text="▶  START SCAN",
                                  bg="#1b5e20", fg="white",
                                  font=("Courier", 10, "bold"),
                                  width=14, relief="flat", cursor="hand2",
                                  command=self.toggle_scan)
        self.scan_btn.pack(side="left", padx=4)

        tk.Button(btn_frame, text="↺  RESET",
                  bg="#37474f", fg="white",
                  font=("Courier", 10, "bold"),
                  width=10, relief="flat", cursor="hand2",
                  command=self.reset).pack(side="left", padx=4)

    # ── DISPLAY ─────────────────────────────────────────────────────────────
    def _update_display(self):
        rc  = self.running_count
        tc  = self.true_count
        rem = max(0, (SHUFFLE_POINT - self.cards_seen) / 52)

        # Running count color
        rc_color = "#ffffff" if rc == 0 else ("#00e676" if rc > 0 else "#ff5252")
        self.rc_value.config(text=str(rc), fg=rc_color)

        # True count color
        tc_color = "#00e5ff" if tc == 0 else ("#00e676" if tc > 0 else "#ff5252")
        self.tc_value.config(text=f"{tc:+.2f}", fg=tc_color)

        self.seen_lbl.config(text=f"Cards seen: {self.cards_seen}")
        self.rem_lbl.config(text=f"Decks left: {rem:.1f}")

        # Advice
        if tc >= 5:
            advice, color = "🔥 MAX BET — Strong edge!", "#b71c1c"
        elif tc >= 3:
            advice, color = "⬆  BET BIG — Good edge", "#e53935"
        elif tc >= 2:
            advice, color = "↑  Increase bets", "#fb8c00"
        elif tc >= 1:
            advice, color = "→  Slight edge, hold", "#f9a825"
        elif tc > -1:
            advice, color = "→  Bet minimum — neutral", "#37474f"
        else:
            advice, color = "⬇  Bet minimum — dealer edge", "#1565c0"

        self.advice_lbl.config(text=advice, bg=color)

        if self.last_cards:
            self.cards_lbl.config(text="Last scan: " + "  ".join(self.last_cards))

    # ── SCANNING ────────────────────────────────────────────────────────────
    def toggle_scan(self):
        if self.scanning:
            self.scanning = False
            self.scan_btn.config(text="▶  START SCAN", bg="#1b5e20")
        else:
            self.scanning = True
            self.scan_btn.config(text="⏹  STOP SCAN", bg="#b71c1c")
            threading.Thread(target=self._scan_loop, daemon=True).start()

    def _scan_loop(self):
        prev_tokens = []
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # full screen
            while self.scanning:
                try:
                    shot  = sct.grab(monitor)
                    img   = np.array(shot)
                    cards = extract_cards_from_image(img)

                    # Only process NEW cards not seen in previous scan
                    new_cards = [c for c in cards if c not in prev_tokens]

                    for card in new_cards:
                        if self.cards_seen >= SHUFFLE_POINT:
                            # Reshuffle detected
                            self.running_count = 0
                            self.cards_seen    = 0
                            self.true_count    = 0.0
                            self.root.after(0, lambda: self.advice_lbl.config(
                                text="⚠  Reshuffle — count reset", bg="#4a148c"))
                            break

                        self.running_count += hilo_tag(card)
                        self.cards_seen    += 1

                    rem_decks = max(0.01, (SHUFFLE_POINT - self.cards_seen) / 52)
                    self.true_count = self.running_count / rem_decks
                    self.last_cards = cards[:10]  # show last 10 detected
                    prev_tokens     = cards

                    self.root.after(0, self._update_display)

                except Exception as e:
                    print(f"Scan error: {e}")

                time.sleep(SCAN_INTERVAL)

    # ── RESET ───────────────────────────────────────────────────────────────
    def reset(self):
        self.running_count = 0
        self.cards_seen    = 0
        self.true_count    = 0.0
        self.last_cards    = []
        self._update_display()
        self.cards_lbl.config(text="Last scan: —")

# ── ENTRY ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = CardCounterApp(root)
    root.mainloop()
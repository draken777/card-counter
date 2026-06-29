"""
Hi-Lo Card Counter — Screen Scanner (Smart Card Detection)
===========================================================
Detects actual card shapes first, then reads values from them.
Works with live dealer streams.

Requirements:
    pip install pillow pytesseract opencv-python mss
    Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
"""

import tkinter as tk
import threading
import time
import re
import mss
import numpy as np
import cv2
from PIL import Image
import pytesseract

# ── CONFIG ───────────────────────────────────────────────────────────────────
TOTAL_DECKS   = 6
PENETRATION   = 0.75
SCAN_INTERVAL = 2
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── HI-LO LOGIC ──────────────────────────────────────────────────────────────
TOTAL_CARDS   = TOTAL_DECKS * 52
SHUFFLE_POINT = int(TOTAL_CARDS * PENETRATION)

LOW_CARDS  = {'2', '3', '4', '5', '6'}
HIGH_CARDS = {'10', 'J', 'Q', 'K', 'A'}
NEUTRAL    = {'7', '8', '9'}

def hilo_tag(card: str) -> int:
    if card in LOW_CARDS:  return +1
    if card in HIGH_CARDS: return -1
    return 0

# ── CARD DETECTION ───────────────────────────────────────────────────────────
def find_card_regions(img_np: np.ndarray) -> list:
    """
    Find white/light rectangular card shapes on the blue table.
    Returns list of cropped card corner images.
    """
    if img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

    # Isolate white/light areas (cards) using HSV
    hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_crops = []
    h_screen, w_screen = img_np.shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Filter by size
        if area < 2000 or area > 80000:
            continue

        # Filter by aspect ratio — cards are portrait shaped
        aspect = h / w
        if aspect < 0.8 or aspect > 2.5:
            continue

        # Only crop top-left corner where card value is printed
        corner_h = max(20, int(h * 0.35))
        corner_w = max(20, int(w * 0.45))

        y2 = min(y + corner_h, h_screen)
        x2 = min(x + corner_w, w_screen)
        corner = img_np[y:y2, x:x2]

        if corner.size > 0:
            card_crops.append(corner)

    return card_crops


def ocr_card_corner(corner_img: np.ndarray) -> str:
    """Run OCR on a single card corner, return card value or empty string."""
    scaled = cv2.resize(corner_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pil_img = Image.fromarray(thresh)
    raw = pytesseract.image_to_string(
        pil_img,
        config="--psm 10 -c tessedit_char_whitelist=0123456789AJQK"
    )
    tokens = re.findall(r'\b(10|[2-9]|[AJQK])\b', raw.upper())
    return tokens[0] if tokens else ""


def extract_cards_from_image(img_np: np.ndarray) -> list:
    """Full pipeline: find card shapes → OCR each corner → return values."""
    card_regions = find_card_regions(img_np)
    found_cards = []
    for region in card_regions:
        value = ocr_card_corner(region)
        if value:
            found_cards.append(value)
    return found_cards


# ── MAIN APP ──────────────────────────────────────────────────────────────────
class CardCounterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hi-Lo Counter")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.92)
        self.root.resizable(False, False)
        self.root.configure(bg="#0a0a0a")

        self.running_count = 0
        self.cards_seen    = 0
        self.true_count    = 0.0
        self.scanning      = False
        self.last_cards    = []

        self._build_ui()
        self._update_display()

    def _build_ui(self):
        tk.Label(self.root, text="♠ HI-LO COUNTER ♠",
                 bg="#0a0a0a", fg="#f0c040",
                 font=("Courier", 13, "bold")).pack(padx=12, pady=(10, 2))

        tk.Frame(self.root, bg="#333", height=1).pack(fill="x", padx=10)

        tk.Label(self.root, text="Running Count",
                 bg="#0a0a0a", fg="#888",
                 font=("Courier", 9)).pack(padx=12, pady=(8, 0))

        self.rc_value = tk.Label(self.root, text="0",
                                 bg="#0a0a0a", fg="#ffffff",
                                 font=("Courier", 48, "bold"))
        self.rc_value.pack(padx=12, pady=0)

        tk.Frame(self.root, bg="#222", height=1).pack(fill="x", padx=10)

        tk.Label(self.root, text="True Count",
                 bg="#0a0a0a", fg="#888",
                 font=("Courier", 9)).pack(padx=12, pady=(6, 0))

        self.tc_value = tk.Label(self.root, text="0.00",
                                 bg="#0a0a0a", fg="#00e5ff",
                                 font=("Courier", 28, "bold"))
        self.tc_value.pack(padx=12, pady=0)

        tk.Frame(self.root, bg="#222", height=1).pack(fill="x", padx=10)

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

        self.advice_lbl = tk.Label(self.root, text="Bet minimum — neutral",
                                   bg="#1a1a1a", fg="#ffffff",
                                   font=("Courier", 10, "bold"),
                                   width=28, pady=5)
        self.advice_lbl.pack(fill="x", padx=10, pady=(2, 6))

        # Shows how many card shapes were detected each scan
        self.status_lbl = tk.Label(self.root, text="Card shapes found: —",
                                   bg="#0a0a0a", fg="#555",
                                   font=("Courier", 8))
        self.status_lbl.pack(padx=12, pady=(0, 2))

        self.cards_lbl = tk.Label(self.root, text="Last scan: —",
                                  bg="#0a0a0a", fg="#555",
                                  font=("Courier", 8),
                                  wraplength=200)
        self.cards_lbl.pack(padx=12, pady=(0, 4))

        tk.Frame(self.root, bg="#333", height=1).pack(fill="x", padx=10)

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

    def _update_display(self):
        rc  = self.running_count
        tc  = self.true_count
        rem = max(0, (SHUFFLE_POINT - self.cards_seen) / 52)

        rc_color = "#ffffff" if rc == 0 else ("#00e676" if rc > 0 else "#ff5252")
        self.rc_value.config(text=str(rc), fg=rc_color)

        tc_color = "#00e5ff" if tc == 0 else ("#00e676" if tc > 0 else "#ff5252")
        self.tc_value.config(text=f"{tc:+.2f}", fg=tc_color)

        self.seen_lbl.config(text=f"Cards seen: {self.cards_seen}")
        self.rem_lbl.config(text=f"Decks left: {rem:.1f}")

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
            monitor = sct.monitors[0]
            while self.scanning:
                try:
                    shot  = sct.grab(monitor)
                    img   = np.array(shot)
                    cards = extract_cards_from_image(img)

                    regions_found = len(find_card_regions(img))
                    self.root.after(0, lambda n=regions_found: self.status_lbl.config(
                        text=f"Card shapes found: {n}"))

                    new_cards = [c for c in cards if c not in prev_tokens]

                    for card in new_cards:
                        if self.cards_seen >= SHUFFLE_POINT:
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
                    self.last_cards = cards[:10]
                    prev_tokens     = cards

                    self.root.after(0, self._update_display)

                except Exception as e:
                    print(f"Scan error: {e}")

                time.sleep(SCAN_INTERVAL)

    def reset(self):
        self.running_count = 0
        self.cards_seen    = 0
        self.true_count    = 0.0
        self.last_cards    = []
        self._update_display()
        self.cards_lbl.config(text="Last scan: —")
        self.status_lbl.config(text="Card shapes found: —")


if __name__ == "__main__":
    root = tk.Tk()
    app  = CardCounterApp(root)
    root.mainloop()
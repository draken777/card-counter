"""
Hi-Lo Card Counter — Screen Scanner (Smart Card Detection + Debug Overlay)
===========================================================================
Detects actual card shapes first, then reads values from them.
Shows a debug window with boxes drawn around detected cards.

Requirements:
    pip install pillow pytesseract opencv-python mss
    Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import re
import mss
import numpy as np
import cv2
from PIL import Image, ImageTk
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
def find_card_regions(img_np: np.ndarray):
    """
    Find white/light rectangular card shapes on the blue table.
    Returns list of (x, y, w, h, corner_crop) tuples.
    """
    if img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

    hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    h_screen, w_screen = img_np.shape[:2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < 2000 or area > 80000:
            continue

        aspect = h / w
        if aspect < 0.8 or aspect > 2.5:
            continue

        corner_h = max(20, int(h * 0.35))
        corner_w = max(20, int(w * 0.45))
        y2 = min(y + corner_h, h_screen)
        x2 = min(x + corner_w, w_screen)
        corner = img_np[y:y2, x:x2]

        if corner.size > 0:
            results.append((x, y, w, h, corner))

    return results


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


def extract_cards_with_positions(img_np: np.ndarray):
    """
    Full pipeline: find card shapes → OCR each corner → return values + positions.
    Returns: (cards_list, debug_img)
    """
    regions = find_card_regions(img_np)
    found_cards = []

    # Make a downscaled copy for the debug overlay
    scale = 0.35
    debug_img = cv2.resize(img_np, None, fx=scale, fy=scale)
    if debug_img.shape[2] == 4:
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGRA2BGR)

    for (x, y, w, h, corner) in regions:
        value = ocr_card_corner(corner)

        # Scale coordinates for debug image
        dx, dy, dw, dh = int(x*scale), int(y*scale), int(w*scale), int(h*scale)

        if value:
            # Green box + label for successfully read cards
            cv2.rectangle(debug_img, (dx, dy), (dx+dw, dy+dh), (0, 220, 80), 2)
            cv2.circle(debug_img, (dx + 6, dy + 6), 5, (0, 220, 80), -1)
            cv2.putText(debug_img, value, (dx + 2, dy + dh - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2)
            found_cards.append(value)
        else:
            # Orange box for detected shape but couldn't read value
            cv2.rectangle(debug_img, (dx, dy), (dx+dw, dy+dh), (0, 140, 255), 2)
            cv2.circle(debug_img, (dx + 6, dy + 6), 5, (0, 140, 255), -1)
            cv2.putText(debug_img, "?", (dx + 2, dy + dh - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

    # Legend
    cv2.rectangle(debug_img, (8, 8), (180, 52), (30, 30, 30), -1)
    cv2.circle(debug_img, (20, 22), 5, (0, 220, 80), -1)
    cv2.putText(debug_img, "= card read OK", (30, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.circle(debug_img, (20, 42), 5, (0, 140, 255), -1)
    cv2.putText(debug_img, "= shape found, no read", (30, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return found_cards, debug_img


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
        self.debug_window  = None
        self.debug_label   = None

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

        # Debug button
        tk.Button(btn_frame, text="🔍 DEBUG",
                  bg="#4a148c", fg="white",
                  font=("Courier", 10, "bold"),
                  width=10, relief="flat", cursor="hand2",
                  command=self.toggle_debug).pack(side="left", padx=4)

    def toggle_debug(self):
        if self.debug_window and tk.Toplevel.winfo_exists(self.debug_window):
            self.debug_window.destroy()
            self.debug_window = None
        else:
            self.debug_window = tk.Toplevel(self.root)
            self.debug_window.title("Debug — Card Detection Overlay")
            self.debug_window.configure(bg="#0a0a0a")
            self.debug_window.attributes("-topmost", True)

            tk.Label(self.debug_window,
                     text="🟢 = card read   🟠 = shape found, value unclear",
                     bg="#0a0a0a", fg="#aaa",
                     font=("Courier", 8)).pack(pady=(6, 2))

            self.debug_label = tk.Label(self.debug_window, bg="#0a0a0a")
            self.debug_label.pack(padx=8, pady=(0, 8))

    def _update_debug(self, debug_img_bgr):
        if not self.debug_window or not tk.Toplevel.winfo_exists(self.debug_window):
            return
        rgb = cv2.cvtColor(debug_img_bgr, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil)
        self.debug_label.config(image=photo)
        self.debug_label.image = photo  # keep reference

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
                    cards, debug_img = extract_cards_with_positions(img)

                    # Update debug overlay
                    self.root.after(0, lambda d=debug_img: self._update_debug(d))

                    # Update status
                    self.root.after(0, lambda n=len(cards): self.status_lbl.config(
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
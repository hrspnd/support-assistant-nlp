import tkinter as tk
from tkinter import font as tkfont
import threading
from pipeline import ConversationalPipeline

# ── Palette ────────────────────────────────────────────────────────────────────
BG_DARK      = "#0D1117"
BG_PANEL     = "#161B22"
BG_BUBBLE_U  = "#1F6FEB"
BG_BUBBLE_B  = "#21262D"
BG_INPUT     = "#0D1117"
ACCENT       = "#1F6FEB"
ACCENT_GLOW  = "#388BFD"
TEXT_PRIMARY = "#E6EDF3"
TEXT_MUTED   = "#8B949E"
TEXT_WHITE   = "#FFFFFF"
BORDER       = "#30363D"
TAG_BG       = "#162032"
TAG_FG       = "#58A6FF"
SUCCESS_FG   = "#56D364"
WARN_FG      = "#F0883E"
ERROR_FG     = "#FF7B72"

TYPING_DOTS  = ("●○○", "○●○", "○○●")


class ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DeliverySupport AI")
        self.geometry("820x700")
        self.minsize(580, 500)
        self.configure(bg=BG_DARK)

        self._dot_frame  = 0
        self._typing_id  = None
        self.pipeline    = ConversationalPipeline()

        self._load_fonts()
        self._build_ui()
        self._welcome()

    # ── Fonts ──────────────────────────────────────────────────────────────────
    def _load_fonts(self):
        self.font_title = tkfont.Font(family="Helvetica Neue", size=13, weight="bold")
        self.font_body  = tkfont.Font(family="Helvetica Neue", size=11)
        self.font_small = tkfont.Font(family="Helvetica Neue", size=9)
        self.font_input = tkfont.Font(family="Helvetica Neue", size=11)

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=BG_PANEL, height=64)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        av = tk.Canvas(header, width=38, height=38, bg=BG_PANEL, highlightthickness=0)
        av.pack(side="left", padx=(18, 10), pady=13)
        av.create_oval(2, 2, 36, 36, fill=ACCENT, outline=ACCENT_GLOW, width=2)
        av.create_text(19, 19, text=":]", font=("Helvetica Neue", 16))

        info = tk.Frame(header, bg=BG_PANEL)
        info.pack(side="left", fill="y", pady=12)
        tk.Label(info, text="DeliverySupport AI", bg=BG_PANEL,
                 fg=TEXT_WHITE, font=self.font_title).pack(anchor="w")
        self.status_lbl = tk.Label(info, text="● Online", bg=BG_PANEL,
                                   fg=SUCCESS_FG, font=self.font_small)
        self.status_lbl.pack(anchor="w")

        # New conversation button
        new_btn = tk.Button(
            header, text="＋ New conversation",
            command=self._new_conversation,
            bg=BG_PANEL, fg=TEXT_MUTED,
            activebackground=BG_DARK, activeforeground=TEXT_PRIMARY,
            relief="flat", font=self.font_small, cursor="hand2", bd=0,
        )
        new_btn.pack(side="right", padx=18)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # Chat canvas
        chat_frame = tk.Frame(self, bg=BG_DARK)
        chat_frame.pack(fill="both", expand=True)

        self.chat_canvas = tk.Canvas(chat_frame, bg=BG_DARK, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(chat_frame, orient="vertical",
                                 command=self.chat_canvas.yview,
                                 bg=BG_PANEL, troughcolor=BG_DARK,
                                 relief="flat", bd=0)
        self.chat_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.chat_canvas.pack(side="left", fill="both", expand=True)

        self.bubble_frame = tk.Frame(self.chat_canvas, bg=BG_DARK)
        self.canvas_win = self.chat_canvas.create_window(
            (0, 0), window=self.bubble_frame, anchor="nw"
        )

        self.bubble_frame.bind("<Configure>", self._on_frame_cfg)
        self.chat_canvas.bind("<Configure>",  self._on_canvas_cfg)
        self.chat_canvas.bind_all("<MouseWheel>", self._on_scroll)

        # Typing indicator (lives inside bubble_frame, packed/unpacked dynamically)
        self.typing_frame = tk.Frame(self.bubble_frame, bg=BG_DARK)
        self.typing_lbl   = tk.Label(
            self.typing_frame, text="", bg=BG_BUBBLE_B, fg=TEXT_MUTED,
            font=self.font_body, padx=14, pady=8,
        )

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # Input bar
        bottom = tk.Frame(self, bg=BG_PANEL, height=72)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)

        input_wrap = tk.Frame(bottom, bg=BORDER, bd=1, relief="flat")
        input_wrap.pack(fill="x", padx=16, pady=16)

        self.input_var = tk.StringVar()
        self.entry = tk.Entry(
            input_wrap, textvariable=self.input_var,
            bg=BG_INPUT, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
            relief="flat", font=self.font_input, bd=0,
        )
        self.entry.pack(side="left", fill="both", expand=True, padx=12, pady=10)
        self.entry.bind("<Return>", self._on_send)

        self.send_btn = tk.Button(
            input_wrap, text="Send →", command=self._on_send,
            bg=ACCENT, fg=TEXT_WHITE,
            activebackground=ACCENT_GLOW, activeforeground=TEXT_WHITE,
            relief="flat", font=self.font_body, cursor="hand2",
            padx=18, pady=0, bd=0,
        )
        self.send_btn.pack(side="right", padx=(0, 6), pady=6, ipady=4)

        self.entry.focus()

    # ── Canvas helpers ─────────────────────────────────────────────────────────
    def _on_frame_cfg(self, _e):
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))

    def _on_canvas_cfg(self, e):
        self.chat_canvas.itemconfig(self.canvas_win, width=e.width)

    def _on_scroll(self, e):
        self.chat_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    def _scroll_bottom(self):
        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    # ── Bubble rendering ───────────────────────────────────────────────────────
    def _add_bubble(self, text: str, side: str = "bot", intent: str = None,
                    note: bool = False, warn: bool = False):
        if note:
            row = tk.Frame(self.bubble_frame, bg=BG_DARK)
            row.pack(fill="x", padx=24, pady=4)
            color = WARN_FG if warn else TEXT_MUTED
            tk.Label(row, text=text, bg=BG_DARK, fg=color,
                     font=self.font_small, wraplength=640, justify="left").pack(anchor="w")
            self._scroll_bottom()
            return

        is_user   = side == "user"
        anchor    = "e" if is_user else "w"
        bubble_bg = BG_BUBBLE_U if is_user else BG_BUBBLE_B
        fg        = TEXT_WHITE  if is_user else TEXT_PRIMARY

        outer = tk.Frame(self.bubble_frame, bg=BG_DARK)
        outer.pack(fill="x", padx=16, pady=(3, 3))
        inner = tk.Frame(outer, bg=BG_DARK)
        inner.pack(anchor=anchor)

        tk.Label(
            inner, text=text, bg=bubble_bg, fg=fg,
            font=self.font_body, wraplength=520,
            justify="left", anchor="w",
            padx=14, pady=10,
        ).pack(anchor=anchor)

        if not is_user and intent:
            tag_row = tk.Frame(inner, bg=BG_DARK)
            tag_row.pack(anchor="w", pady=(2, 0))
            tk.Label(
                tag_row,
                text=f"  {intent.replace('_', ' ').title()}  ",
                bg=TAG_BG, fg=TAG_FG, font=self.font_small,
                padx=4, pady=2,
            ).pack(side="left")

        self._scroll_bottom()

    # ── Typing indicator ───────────────────────────────────────────────────────
    def _show_typing(self):
        self.typing_frame.pack(fill="x", padx=16, pady=4)
        self.typing_lbl.pack(anchor="w")
        self._animate_typing()
        self._scroll_bottom()

    def _animate_typing(self):
        dot = TYPING_DOTS[self._dot_frame % 3]
        self.typing_lbl.configure(text=f"  {dot}  ")
        self._dot_frame += 1
        self._typing_id = self.after(380, self._animate_typing)

    def _hide_typing(self):
        if self._typing_id:
            self.after_cancel(self._typing_id)
            self._typing_id = None
        self.typing_lbl.configure(text="")
        self.typing_frame.pack_forget()

    # ── Welcome ────────────────────────────────────────────────────────────────
    def _welcome(self):
        self._add_bubble(
            "👋  Hello! I'm your DeliverySupport assistant.\n"
            "I can help with tracking orders, delivery issues, missing items, "
            "shipping costs, and more.\n\nHow can I help you today?",
            side="bot",
        )

    # ── New conversation ───────────────────────────────────────────────────────
    def _new_conversation(self):
        for widget in self.bubble_frame.winfo_children():
            if widget is not self.typing_frame:
                widget.destroy()
        self.pipeline.reset()
        self._welcome()

    # ── Send ───────────────────────────────────────────────────────────────────
    def _on_send(self, _e=None):
        text = self.input_var.get().strip()
        if not text:
            return
        self.input_var.set("")
        self._add_bubble(text, side="user")
        self.entry.config(state="disabled")
        self.send_btn.config(state="disabled")
        threading.Thread(target=self._infer, args=(text,), daemon=True).start()

    def _infer(self, text: str):
        self.after(0, self._show_typing)
        try:
            reply, intent = self.pipeline.respond(text)
        except Exception as exc:
            reply  = f"⚠ Error: {exc}"
            intent = None

        self.after(0, self._hide_typing)
        self.after(0, lambda: self._add_bubble(reply, side="bot", intent=intent))
        self.after(0, lambda: self.entry.config(state="normal"))
        self.after(0, lambda: self.send_btn.config(state="normal"))
        self.after(0, self.entry.focus)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()

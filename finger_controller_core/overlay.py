import queue
import sys
import threading
import time
import tkinter as tk


class ScreenOverlay:
    """Transparent overlay to show action notifications.

    Important: Tkinter can be fragile when created at import time and/or off the
    main thread. To reduce crash risk, the overlay starts lazily on first use
    and becomes a no-op if initialization fails.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[tuple[str, float]]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._failed: bool = False

    def _ensure_started(self) -> None:
        if self._failed:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def show(self, text: str, duration: float = 1.0) -> None:
        """Request a message to be displayed.

        If the overlay backend can't start (e.g., Tk init error), this degrades
        gracefully to a console print.
        """
        if not sys.platform.startswith("win"):
            return

        self._ensure_started()
        if self._failed:
            # Keep notifications visible in logs rather than crashing.
            print(f"[Overlay] {text}")
            return

        self._queue.put((text, duration))

    def _run_loop(self) -> None:
        try:
            self.root = tk.Tk()
            self.root.overrideredirect(True)          # Remove window borders
            self.root.attributes("-topmost", True)    # Keep on top of all other windows

            # Transparent background setup for Windows
            self._transparent_color = "magenta"
            self.root.config(bg=self._transparent_color)
            self.root.attributes("-transparentcolor", self._transparent_color)

            # Make the window click-through using ctypes (Windows only)
            try:
                import ctypes

                self.root.update_idletasks()
                hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
                # WS_EX_LAYERED = 0x00080000, WS_EX_TRANSPARENT = 0x00000020
                style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
                ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x00080000 | 0x00000020)
            except Exception:
                pass  # Fallback if ctypes fails

            # Dimensions and positioning (bottom center, like Windows 11 volume)
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            window_width = 400
            window_height = 80
            x = (screen_width // 2) - (window_width // 2)
            y = screen_height - 150

            self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

            # Start hidden (magenta background is transparent)
            self.label = tk.Label(
                self.root,
                text="",
                font=("Segoe UI", 16, "bold"),
                fg="#FFFFFF",
                bg=self._transparent_color,
                padx=20,
                pady=15,
            )
            self.label.pack(expand=True)

            self._hide_time = 0.0
            self._update()
            self.root.mainloop()
        except Exception as exc:
            self._failed = True
            print(f"[Overlay] Disabled (init failed: {exc})")

    def _update(self) -> None:
        # Check if there's a new message
        try:
            while True:
                text, duration = self._queue.get_nowait()
                # Become a dark, visible pill shape when showing text
                self.label.config(text=text, bg="#1E1E1E")
                self._hide_time = time.time() + duration
        except queue.Empty:
            pass

        # Clear text and hide background if duration has expired
        if time.time() > self._hide_time and self.label.cget("text") != "":
            self.label.config(text="", bg=self._transparent_color)

        # Schedule the next check
        self.root.after(50, self._update)

# Global singleton
overlay = ScreenOverlay()

import threading
import queue
import time
import tkinter as tk

class ScreenOverlay:
    """Creates a transparent, click-through overlay on the screen to show action notifications."""
    
    def __init__(self):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def show(self, text: str, duration: float = 1.0):
        """Sends a message to the overlay thread to display text."""
        self._queue.put((text, duration))

    def _run_loop(self):
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
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            # WS_EX_LAYERED = 0x00080000, WS_EX_TRANSPARENT = 0x00000020
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x00080000 | 0x00000020)
        except Exception:
            pass # Fallback if ctypes fails, it'll just be a standard topmost window
        
        # Dimensions and positioning (bottom center, like Windows 11 volume)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 400
        window_height = 80
        x = (screen_width // 2) - (window_width // 2)
        y = screen_height - 150
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # We start with the background as magenta so it is invisible
        self.label = tk.Label(
            self.root, 
            text="", 
            font=("Segoe UI", 16, "bold"), 
            fg="#FFFFFF",  # crisp white text
            bg=self._transparent_color,
            padx=20,
            pady=15
        )
        self.label.pack(expand=True)
        
        self._hide_time = 0
        self._update()
        self.root.mainloop()

    def _update(self):
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

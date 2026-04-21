import ctypes
from ctypes import (c_long, c_ulong, c_uint32, c_int32, c_uint64,
                    c_void_p, Structure, Union, POINTER)
import time

# Load user32 with use_last_error=True so ctypes.get_last_error() is reliable
_user32 = ctypes.WinDLL("user32", use_last_error=True)

# ═══════════════════════════════════════════════════════════════════════
#  Win32 structures – Touch Injection API  (Windows 8+)
# ═══════════════════════════════════════════════════════════════════════

class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

class RECT(Structure):
    _fields_ = [
        ("left", c_long), ("top", c_long),
        ("right", c_long), ("bottom", c_long),
    ]

class POINTER_INFO(Structure):
    _fields_ = [
        ("pointerType",            c_uint32),
        ("pointerId",              c_uint32),
        ("frameId",                c_uint32),
        ("pointerFlags",           c_uint32),
        ("sourceDevice",           c_void_p),
        ("hwndTarget",             c_void_p),
        ("ptPixelLocation",        POINT),
        ("ptHimetricLocation",     POINT),
        ("ptPixelLocationRaw",     POINT),
        ("ptHimetricLocationRaw",  POINT),
        ("dwTime",                 c_uint32),
        ("historyCount",           c_uint32),
        ("InputData",              c_int32),
        ("dwKeyStates",            c_uint32),
        ("PerformanceCount",       c_uint64),
        ("ButtonChangeType",       c_int32),
    ]

class POINTER_TOUCH_INFO(Structure):
    _fields_ = [
        ("pointerInfo",  POINTER_INFO),
        ("touchFlags",   c_uint32),
        ("touchMask",    c_uint32),
        ("rcContact",    RECT),
        ("rcContactRaw", RECT),
        ("orientation",  c_uint32),
        ("pressure",     c_uint32),
    ]

# ─── Pointer / touch constants (matching MSDN exactly) ──────────────
PT_TOUCH                = 2

POINTER_FLAG_INRANGE    = 0x00000002
POINTER_FLAG_INCONTACT  = 0x00000004
POINTER_FLAG_DOWN       = 0x00010000
POINTER_FLAG_UPDATE     = 0x00020000
POINTER_FLAG_UP         = 0x00040000

TOUCH_FLAG_NONE         = 0x00000000
TOUCH_MASK_CONTACTAREA  = 0x00000001
TOUCH_MASK_ORIENTATION  = 0x00000002
TOUCH_MASK_PRESSURE     = 0x00000004

TOUCH_FEEDBACK_DEFAULT  = 0x1

# ═══════════════════════════════════════════════════════════════════════
#  Win32 structures – Ctrl+Scroll fallback
# ═══════════════════════════════════════════════════════════════════════

class MOUSEINPUT(Structure):
    _fields_ = [
        ("dx", c_long), ("dy", c_long),
        ("mouseData", c_ulong), ("dwFlags", c_ulong),
        ("time", c_ulong), ("dwExtraInfo", POINTER(c_ulong)),
    ]

class INPUT(Structure):
    class _INPUT(Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("_input",)
    _fields_ = [("type", c_ulong), ("_input", _INPUT)]

INPUT_MOUSE        = 0
MOUSEEVENTF_WHEEL  = 0x0800
KEYEVENTF_KEYUP    = 0x0002
VK_CONTROL         = 0x11

# ═══════════════════════════════════════════════════════════════════════
#  Smoothing parameters
# ═══════════════════════════════════════════════════════════════════════
EMA_ALPHA       = 0.4
EMIT_THRESHOLD  = 0.5
BASE_MULTIPLIER = 1.5
MAX_MULTIPLIER  = 4.0
VELOCITY_SCALE  = 0.05


# ═══════════════════════════════════════════════════════════════════════
#  Injector
# ═══════════════════════════════════════════════════════════════════════

class WindowsTouchInjector:
    """Injects pinch-to-zoom using the Windows Touch Injection API
       (trackpad-style, works in browsers).  Falls back to Ctrl+Scroll
       if touch injection is not available."""

    def __init__(self, max_points=2):
        self._max_points = max_points
        self._touch_ready = False          # True after lazy init succeeds
        self._touch_available = True       # False if init permanently fails
        self._contacts_down = False        # True while touch contacts are active
        self.initialized = True

        # Runtime state
        self._smoothed_dist  = 0.0
        self._committed_dist = 0.0
        self._accum          = 0.0
        self._last_time      = 0.0

    # ── lazy initialization ──────────────────────────────────────

    def _ensure_touch_init(self) -> bool:
        """Lazily initialize touch injection on first use.
        This avoids conflicts with camera/MediaPipe/TF init."""
        if self._touch_ready:
            return True
        if not self._touch_available:
            return False

        try:
            ctypes.set_last_error(0)
            ok = _user32.InitializeTouchInjection(self._max_points, TOUCH_FEEDBACK_DEFAULT)
            err = ctypes.get_last_error()
            if ok:
                self._touch_ready = True
                print("[TouchInjector] Touch Injection initialized (lazy)")
                return True
            else:
                print(f"[TouchInjector] InitializeTouchInjection failed (err={err})")
                self._touch_available = False
                return False
        except Exception as e:
            print(f"[TouchInjector] Touch Injection unavailable ({e})")
            self._touch_available = False
            return False

    @property
    def use_touch(self) -> bool:
        return self._touch_ready

    # ── helpers ──────────────────────────────────────────────────

    def _make_contact(self, pointer_id, x, y, flags, pressure=32000):
        """Build a POINTER_TOUCH_INFO matching the MS docs example."""
        c = POINTER_TOUCH_INFO()
        c.pointerInfo.pointerType   = PT_TOUCH
        c.pointerInfo.pointerId     = pointer_id
        c.pointerInfo.pointerFlags  = flags

        # Clamp to valid screen bounds — negative or out-of-range
        # coordinates cause InjectTouchInput to fail with err=87
        sw = ctypes.windll.user32.GetSystemMetrics(0)  # SM_CXSCREEN
        sh = ctypes.windll.user32.GetSystemMetrics(1)  # SM_CYSCREEN
        ix = max(0, min(int(x), sw - 1))
        iy = max(0, min(int(y), sh - 1))
        c.pointerInfo.ptPixelLocation.x    = ix
        c.pointerInfo.ptPixelLocation.y    = iy

        c.touchFlags  = TOUCH_FLAG_NONE
        c.touchMask   = (TOUCH_MASK_CONTACTAREA |
                         TOUCH_MASK_ORIENTATION |
                         TOUCH_MASK_PRESSURE)
        c.orientation = 90
        c.pressure    = pressure

        c.rcContact.left   = ix - 2
        c.rcContact.top    = iy - 2
        c.rcContact.right  = ix + 2
        c.rcContact.bottom = iy + 2
        return c

    def _inject(self, contacts, label=""):
        ctypes.set_last_error(0)
        arr = (POINTER_TOUCH_INFO * len(contacts))(*contacts)

        # Retry logic for err=1460 (queue full / timeout)
        for attempt in range(3):
            ok = _user32.InjectTouchInput(len(contacts), ctypes.byref(arr))
            if ok:
                return True
            err = ctypes.get_last_error()
            if err == 1460:
                time.sleep(0.02)
                continue
            break

        c0 = contacts[0]
        print(f"[TouchInjector] {label} FAILED err={err}  "
              f"pos=({c0.pointerInfo.ptPixelLocation.x},{c0.pointerInfo.ptPixelLocation.y})  "
              f"flags=0x{c0.pointerInfo.pointerFlags:08X}  "
              f"contacts_down={self._contacts_down}")
        return False

    # ── Touch Injection path ────────────────────────────────────

    def _touch_start(self, cx, cy, dist):
        half = dist / 2.0
        fl = POINTER_FLAG_DOWN | POINTER_FLAG_INRANGE | POINTER_FLAG_INCONTACT
        c0 = self._make_contact(0, cx - half, cy, fl)
        c1 = self._make_contact(1, cx + half, cy, fl)
        print(f"[TouchInjector] DOWN  center=({int(cx)},{int(cy)}) dist={dist:.1f}")
        if self._inject([c0, c1], "DOWN"):
            self._contacts_down = True
            self._fail_count = 0
            # Windows needs time to process DOWN before accepting UPDATEs
            time.sleep(0.05)

    def _touch_update(self, cx, cy, dist):
        if not self._contacts_down:
            return
        half = dist / 2.0
        fl = POINTER_FLAG_UPDATE | POINTER_FLAG_INRANGE | POINTER_FLAG_INCONTACT
        c0 = self._make_contact(0, cx - half, cy, fl)
        c1 = self._make_contact(1, cx + half, cy, fl)
        if not self._inject([c0, c1], "UPDATE"):
            self._fail_count = getattr(self, '_fail_count', 0) + 1
            if self._fail_count >= 3:
                # Recovery: lift stale contacts, re-init, re-start
                print("[TouchInjector] Recovering from failed UPDATEs...")
                self._contacts_down = False
                self._touch_ready = False
                if self._ensure_touch_init():
                    self._touch_start(cx, cy, dist)
        else:
            self._fail_count = 0

    def _touch_end(self, cx, cy):
        if not self._contacts_down:
            return
        fl = POINTER_FLAG_UP
        c0 = self._make_contact(0, cx, cy, fl, pressure=0)
        c1 = self._make_contact(1, cx, cy, fl, pressure=0)
        print(f"[TouchInjector] UP    center=({int(cx)},{int(cy)})")
        self._inject([c0, c1], "UP")
        self._contacts_down = False

    # ── Ctrl+Scroll fallback path ───────────────────────────────

    def _scroll_start(self, cx, cy):
        ctypes.windll.user32.SetCursorPos(int(cx), int(cy))
        ctypes.windll.user32.keybd_event(VK_CONTROL, 0, 0, 0)

    def _scroll_update(self, cx, cy, wheel_amount):
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.mi.mouseData  = wheel_amount & 0xFFFFFFFF
        inp.mi.dwFlags    = MOUSEEVENTF_WHEEL
        inp.mi.dwExtraInfo = None
        ctypes.windll.user32.SetCursorPos(int(cx), int(cy))
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def _scroll_end(self):
        ctypes.windll.user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)

    # ═══════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════

    def start_pinch(self, center_x, center_y, initial_dist=100):
        self._smoothed_dist  = initial_dist
        self._committed_dist = initial_dist
        self._accum          = 0.0
        self._last_time      = time.perf_counter()

        if self._ensure_touch_init():
            # Clean up any stale contacts from a previous cycle
            if self._contacts_down:
                self._touch_end(center_x, center_y)
                time.sleep(0.02)
            self._touch_start(center_x, center_y, initial_dist)
        else:
            self._scroll_start(center_x, center_y)

    def update_pinch(self, center_x, center_y, new_dist):
        now = time.perf_counter()
        dt  = max(now - self._last_time, 1e-6)
        self._last_time = now

        # EMA smooth the incoming distance
        self._smoothed_dist += EMA_ALPHA * (new_dist - self._smoothed_dist)

        if self._touch_ready:
            self._touch_update(center_x, center_y, self._smoothed_dist)
        else:
            # Ctrl+Scroll fallback with accumulator + velocity scaling
            delta = self._smoothed_dist - self._committed_dist
            self._accum += delta
            self._committed_dist = self._smoothed_dist

            if abs(self._accum) >= EMIT_THRESHOLD:
                velocity   = abs(delta) / dt
                multiplier = min(BASE_MULTIPLIER + velocity * VELOCITY_SCALE,
                                 MAX_MULTIPLIER)
                wheel_amt  = int(self._accum * multiplier)
                if wheel_amt != 0:
                    self._scroll_update(center_x, center_y, wheel_amt)
                    self._accum -= wheel_amt / multiplier

    def end_pinch(self, center_x, center_y):
        if self._touch_ready:
            self._touch_end(center_x, center_y)
        else:
            self._scroll_end()
        self._accum = 0.0


touch_injector = WindowsTouchInjector()

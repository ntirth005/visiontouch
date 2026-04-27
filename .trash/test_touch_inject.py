"""Reproduce the finger_controller import environment and test touch injection.
This tests whether pyautogui/opencv/mediapipe imports break touch injection."""
import ctypes
from ctypes import c_long, c_uint32, c_int32, c_uint64, c_void_p, Structure
import time, sys

user32 = ctypes.windll.user32

class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]
class RECT(Structure):
    _fields_ = [("left", c_long), ("top", c_long), ("right", c_long), ("bottom", c_long)]
class POINTER_INFO(Structure):
    _fields_ = [
        ("pointerType", c_uint32), ("pointerId", c_uint32),
        ("frameId", c_uint32), ("pointerFlags", c_uint32),
        ("sourceDevice", c_void_p), ("hwndTarget", c_void_p),
        ("ptPixelLocation", POINT), ("ptHimetricLocation", POINT),
        ("ptPixelLocationRaw", POINT), ("ptHimetricLocationRaw", POINT),
        ("dwTime", c_uint32), ("historyCount", c_uint32),
        ("InputData", c_int32), ("dwKeyStates", c_uint32),
        ("PerformanceCount", c_uint64), ("ButtonChangeType", c_int32),
    ]
class POINTER_TOUCH_INFO(Structure):
    _fields_ = [
        ("pointerInfo", POINTER_INFO), ("touchFlags", c_uint32),
        ("touchMask", c_uint32), ("rcContact", RECT),
        ("rcContactRaw", RECT), ("orientation", c_uint32),
        ("pressure", c_uint32),
    ]

POINTER_FLAG_INRANGE   = 0x00000002
POINTER_FLAG_INCONTACT = 0x00000004
POINTER_FLAG_DOWN      = 0x00010000
POINTER_FLAG_UPDATE    = 0x00020000
POINTER_FLAG_UP        = 0x00040000
TOUCH_MASK_CONTACTAREA = 0x00000001
TOUCH_MASK_ORIENTATION = 0x00000002
TOUCH_MASK_PRESSURE    = 0x00000004

def make(pid, x, y, flags, pressure=32000):
    c = POINTER_TOUCH_INFO()
    c.pointerInfo.pointerType = 2
    c.pointerInfo.pointerId = pid
    c.pointerInfo.pointerFlags = flags
    c.pointerInfo.ptPixelLocation.x = int(x)
    c.pointerInfo.ptPixelLocation.y = int(y)
    c.touchMask = TOUCH_MASK_CONTACTAREA | TOUCH_MASK_ORIENTATION | TOUCH_MASK_PRESSURE
    c.orientation = 90
    c.pressure = pressure
    c.rcContact.left = int(x)-2; c.rcContact.top = int(y)-2
    c.rcContact.right = int(x)+2; c.rcContact.bottom = int(y)+2
    return c

def test_inject(label):
    cx, cy = 960, 540
    fl = POINTER_FLAG_DOWN | POINTER_FLAG_INRANGE | POINTER_FLAG_INCONTACT
    c0 = make(0, cx-50, cy, fl)
    c1 = make(1, cx+50, cy, fl)
    arr = (POINTER_TOUCH_INFO * 2)(c0, c1)
    ok = user32.InjectTouchInput(2, ctypes.byref(arr))
    err = ctypes.GetLastError()
    print(f"  [{label}] DOWN: {'OK' if ok else 'FAILED'}  err={err}")
    if ok:
        time.sleep(0.05)
        fl_up = POINTER_FLAG_UP
        c0 = make(0, cx, cy, fl_up, 0)
        c1 = make(1, cx, cy, fl_up, 0)
        arr = (POINTER_TOUCH_INFO * 2)(c0, c1)
        user32.InjectTouchInput(2, ctypes.byref(arr))
    return ok

print("=== Environment Interference Test ===\n")

# Test 1: Before any imports
print("1. InitializeTouchInjection BEFORE other imports...")
ok = user32.InitializeTouchInjection(2, 1)
print(f"   Init: {'OK' if ok else 'FAILED'}  err={ctypes.GetLastError()}")
if ok:
    test_inject("before imports")

# Test 2: Import pyautogui
print("\n2. Importing pyautogui...")
import pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
print(f"   DPI awareness: {ctypes.windll.user32.GetAwarenessFromDpiAwarenessContext(ctypes.windll.user32.GetThreadDpiAwarenessContext())}")
# Re-init and test
ok = user32.InitializeTouchInjection(2, 1)
print(f"   Re-init: {'OK' if ok else 'FAILED'}  err={ctypes.GetLastError()}")
if ok:
    test_inject("after pyautogui")

# Test 3: Import cv2
print("\n3. Importing cv2...")
import cv2
ok = user32.InitializeTouchInjection(2, 1)
print(f"   Re-init: {'OK' if ok else 'FAILED'}  err={ctypes.GetLastError()}")
if ok:
    test_inject("after cv2")

# Test 4: Do a pyautogui.moveTo then test
print("\n4. Calling pyautogui.moveTo then testing...")
pyautogui.moveTo(960, 540, duration=0)
ok = user32.InitializeTouchInjection(2, 1)
print(f"   Re-init: {'OK' if ok else 'FAILED'}  err={ctypes.GetLastError()}")
if ok:
    test_inject("after moveTo")

# Test 5: Rapid DOWN → immediate UPDATE (no sleep) → UP
print("\n5. Testing rapid DOWN → UPDATE (no sleep)...")
ok = user32.InitializeTouchInjection(2, 1)
cx, cy = 960, 540
fl_dn = POINTER_FLAG_DOWN | POINTER_FLAG_INRANGE | POINTER_FLAG_INCONTACT
c0 = make(0, cx-50, cy, fl_dn)
c1 = make(1, cx+50, cy, fl_dn)
arr = (POINTER_TOUCH_INFO * 2)(c0, c1)
ok = user32.InjectTouchInput(2, ctypes.byref(arr))
print(f"   DOWN: {'OK' if ok else 'FAILED'}  err={ctypes.GetLastError()}")

# Immediate update - NO SLEEP
fl_up = POINTER_FLAG_UPDATE | POINTER_FLAG_INRANGE | POINTER_FLAG_INCONTACT
c0 = make(0, cx-55, cy, fl_up)
c1 = make(1, cx+55, cy, fl_up)
arr = (POINTER_TOUCH_INFO * 2)(c0, c1)
ok = user32.InjectTouchInput(2, ctypes.byref(arr))
print(f"   UPDATE (immediate): {'OK' if ok else 'FAILED'}  err={ctypes.GetLastError()}")

fl_up = POINTER_FLAG_UP
c0 = make(0, cx, cy, fl_up, 0)
c1 = make(1, cx, cy, fl_up, 0)
arr = (POINTER_TOUCH_INFO * 2)(c0, c1)
user32.InjectTouchInput(2, ctypes.byref(arr))

# Test 6: Check if DPI awareness changed
print(f"\n6. Final DPI check:")
try:
    dpi = ctypes.windll.user32.GetAwarenessFromDpiAwarenessContext(
        ctypes.windll.user32.GetThreadDpiAwarenessContext())
    print(f"   DPI awareness level: {dpi}")
    print(f"   (0=unaware, 1=system, 2=per-monitor)")
except:
    print("   Could not query DPI awareness")

print("\n=== Done ===")

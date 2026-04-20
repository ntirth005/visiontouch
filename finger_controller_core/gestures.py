import math
from typing import Optional

from .types import ActionEvent, FingerState


class GestureEngine:
    """Finger state -> ActionEvent state machine.

    Gesture summary
    ───────────────
    Cursor movement
        • Exactly one finger visible    → follow that finger (index or middle)
        • Any other combo               → follow index (fallback)

    Pinch (index up, thumb close to index)
        • Close → release (fast)        → CLICK  (or DOUBLE_CLICK if within window)

    Zoom (Single Hand: index + middle both up) 
    Zoom (Two Hands: both index fingers up)
        • Continuously tracks distance between index and middle (or left index and right index).
        • Spreading fingers further than anchor → ZOOM_IN
        • Bringing fingers closer than anchor  → ZOOM_OUT

    Swipe (index, middle, ring up OR index, middle, ring, pinky up)
        • 3-finger horizontal movement  → SWIPE_3F_LEFT / SWIPE_3F_RIGHT
        • 4-finger horizontal movement  → SWIPE_4F_LEFT / SWIPE_4F_RIGHT

    Freeze (all 5 fingers up)
        • Freezes cursor position for 2 seconds when activated.
    """

    # ── Pinch thresholds ──────────────────────────────────────────────────────
    PINCH_CLOSE_PX: int       = 55      # thumb–index gap → "pinched"
    PINCH_OPEN_PX: int        = 80      # thumb–index gap → "released"

    # ── Zoom (continuous proportional model) ──────────────────────────────────
    # How it works:
    #   • When index+middle enter zoom mode, the current finger distance is
    #     captured as the "anchor" distance.
    #   • Every frame, delta = current_dist - anchor_dist is computed.
    #   • If |delta| exceeds ZOOM_DEAD_ZONE_PX the engine emits ZOOM_IN or
    #     ZOOM_OUT with a normalised `scale` (delta / ZOOM_SCALE_DIVISOR).
    #   • The anchor is NOT updated each frame — it resets only when fingers
    #     leave zoom mode, giving stable proportional control.
    #
    # Tune these two values:
    ZOOM_DEAD_ZONE_PX: int    = 12      # px ignored around anchor (kills idle jitter)
    ZOOM_SCALE_DIVISOR: float = 80.0    # px of movement that equals scale 1.0
    ZOOM_GRACE_SEC: float     = 0.15    # grace period for dropped frames in zoom mode
    SWIPE_THRESHOLD_PX: int   = 80    # px of horizontal movement to trigger swipe

    # ── Internal states ───────────────────────────────────────────────────────
    _IDLE          = "IDLE"
    _MOVING        = "MOVING"
    _PINCHING      = "PINCHING"
    _RIGHT_PINCHING = "RIGHT_PINCHING"
    _ZOOM          = "ZOOM"

    def __init__(self) -> None:
        self._state: str = self._IDLE
        self._prev: Optional[FingerState] = None

        # Pinch-related
        self._pinch_start_ts: Optional[float]    = None
        self._pinch_start_pos: Optional[tuple[int, int]] = None

        # Zoom-related
        self._zoom_anchor_dist: Optional[float] = None  # set on zoom mode entry
        self._zoom_lost_ts: Optional[float] = None      # tracking lost timestamp

        # Swipe & Freeze
        self._swipe_start_x: Optional[int] = None
        self._swipe_fired: bool = False
        self._cursor_frozen_until: float = 0.0
        self._frozen_pos: tuple[int, int] = (0, 0)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, fs_list: list[FingerState]) -> Optional[ActionEvent]:
        """Consume FingerState frames for all hands and return the resulting ActionEvent (or None)."""
        if not fs_list:
            return None

        # Two-hand processing check (Two-handed zoom takes absolute priority)
        if len(fs_list) >= 2:
            return self._classify_two_hands(fs_list[0], fs_list[1])
            
        # Single hand processing
        fs = fs_list[0]
        fingers_up_count = sum([fs.index_up, fs.middle_up, fs.ring_up, fs.pinky_up, fs.thumb_up])
        
        # 5-Finger Freeze Activation
        if fingers_up_count == 5:
            if getattr(self, '_cursor_frozen_until', 0.0) < fs.ts:
                self._frozen_pos = self._cursor_pos(fs)
            self._cursor_frozen_until = fs.ts + 2.0

        ev = self._classify(fs)
        
        # Enforce Freeze logic over the event
        if fs.ts < getattr(self, '_cursor_frozen_until', 0.0):
            if ev is not None and ev.name == "MOVE":
                ev = None
            elif ev is not None:
                ev.screen_pos = self._frozen_pos

        self._prev = fs
        return ev

    # ──────────────────────────────────────────────────────────────────────────
    # Two-Handed Classifier
    # ──────────────────────────────────────────────────────────────────────────
    
    def _classify_two_hands(self, fs1: FingerState, fs2: FingerState) -> Optional[ActionEvent]:
        """Classify 2-handed gestures (specifically two-handed Zoom)."""
        
        # Only trigger two-hand zoom if BOTH hands only have their index finger up
        fs1_zoom_pose = fs1.index_up and not fs1.middle_up and not fs1.ring_up and not fs1.pinky_up
        fs2_zoom_pose = fs2.index_up and not fs2.middle_up and not fs2.ring_up and not fs2.pinky_up
        
        is_zoom_pose = fs1_zoom_pose and fs2_zoom_pose
        
        # We need a new property to store calculating physical distance between two separate hands
        dx = fs1.index_tip_px[0] - fs2.index_tip_px[0]
        dy = fs1.index_tip_px[1] - fs2.index_tip_px[1]
        dist = math.hypot(dx, dy)
        pos = self._cursor_pos(fs1)  # use the first hand for placing the cursor/event
        
        if self._state == self._ZOOM and not is_zoom_pose:
            # Handle dropped frames/graceful exit
            if self._zoom_lost_ts is None:
                self._zoom_lost_ts = fs1.ts
                
            if fs1.ts - self._zoom_lost_ts > self.ZOOM_GRACE_SEC:
                self._exit_zoom()
                self._zoom_lost_ts = None
                self._state = self._IDLE
            else:
                return ActionEvent("MOVE", pos)
                
        elif is_zoom_pose:
            self._zoom_lost_ts = None
            
            # End any in-progress drag cleanly before entering zoom.
            if self._state in (self._PINCHING, self._RIGHT_PINCHING):
                self._state = self._ZOOM
                self._pinch_start_ts = None
                self._zoom_anchor_dist = dist

            # Capture anchor on first zoom frame.
            if self._state != self._ZOOM:
                self._state = self._ZOOM
                self._zoom_anchor_dist = dist

            # Custom _update_zoom logic using the hands' distance
            if self._zoom_anchor_dist is None:
                self._zoom_anchor_dist = dist
                return ActionEvent("MOVE", pos)

            delta = dist - self._zoom_anchor_dist

            if abs(delta) < self.ZOOM_DEAD_ZONE_PX:
                return ActionEvent("MOVE", pos)

            scale = abs(delta) / self.ZOOM_SCALE_DIVISOR
            if delta > 0:
                return ActionEvent("ZOOM_IN", pos, scale=scale)
            else:
                return ActionEvent("ZOOM_OUT", pos, scale=scale)
                
        self._state = self._MOVING
        return ActionEvent("MOVE", pos)


    # ──────────────────────────────────────────────────────────────────────────
    # Cursor position helper
    # ──────────────────────────────────────────────────────────────────────────

    def _cursor_pos(self, fs: FingerState) -> tuple[int, int]:
        """Decide which fingertip to use as the cursor.

        Rules:
          1. If middle finger is up -> follow middle.
          2. Otherwise -> follow index.
        """
        if fs.middle_up:
            return (fs.middle_screen_x, fs.middle_screen_y)
        return (fs.screen_x, fs.screen_y)

    # ──────────────────────────────────────────────────────────────────────────
    # Pinch-release handler (immediate click)
    # ──────────────────────────────────────────────────────────────────────────

    def _on_pinch_release(self, pos: tuple[int, int]) -> ActionEvent:
        """Called when a PINCH (not PINCH_DRAG) is released. Emits a CLICK instantly."""
        return ActionEvent("CLICK", pos)

    def _on_right_pinch_release(self, pos: tuple[int, int]) -> ActionEvent:
        """Called when a RIGHT PINCH is released. Emits a RIGHT_CLICK instantly."""
        return ActionEvent("RIGHT_CLICK", pos)

    # ──────────────────────────────────────────────────────────────────────────
    # Zoom state machine (continuous proportional)
    # ──────────────────────────────────────────────────────────────────────────

    def _enter_zoom(self, fs: FingerState) -> None:
        """Capture the anchor distance when zoom mode is first entered."""
        self._zoom_anchor_dist = fs.two_finger_dist

    def _exit_zoom(self) -> None:
        """Clear zoom anchor when leaving zoom mode."""
        self._zoom_anchor_dist = None

    def _update_zoom(self, fs: FingerState, pos: tuple[int, int]) -> ActionEvent:
        """Continuous proportional zoom.

        Computes how far fingers have moved relative to the anchor captured
        on mode entry.  Inside the dead-zone → plain MOVE.  Outside → ZOOM_IN
        or ZOOM_OUT with a normalised scale the actions layer can use to decide
        how many Ctrl+scroll ticks (or how large a pyautogui.hotkey step) to fire.

        scale == 1.0 means fingers moved exactly ZOOM_SCALE_DIVISOR pixels from
        anchor.  Actions layer can threshold at e.g. scale > 0.15 per tick.
        """
        if self._zoom_anchor_dist is None:
            # Shouldn't happen, but be defensive.
            self._enter_zoom(fs)
            return ActionEvent("MOVE", pos)

        delta = fs.two_finger_dist - self._zoom_anchor_dist

        if abs(delta) < self.ZOOM_DEAD_ZONE_PX:
            return ActionEvent("MOVE", pos)

        scale = abs(delta) / self.ZOOM_SCALE_DIVISOR
        if delta > 0:
            return ActionEvent("ZOOM_IN", pos, scale=scale)
        else:
            return ActionEvent("ZOOM_OUT", pos, scale=scale)

    # ──────────────────────────────────────────────────────────────────────────
    # Main classifier
    # ──────────────────────────────────────────────────────────────────────────

    def _classify(self, fs: FingerState) -> Optional[ActionEvent]:  # noqa: C901
        pos = self._cursor_pos(fs)
        fingers_up_count = sum([fs.index_up, fs.middle_up, fs.ring_up, fs.pinky_up, fs.thumb_up])

        # 1.5. Swipes (strictly anatomical poses)
        is_3f_swipe = fs.index_up and fs.middle_up and fs.ring_up and not fs.pinky_up and not fs.thumb_up
        is_4f_swipe = fs.index_up and fs.middle_up and fs.ring_up and fs.pinky_up and not fs.thumb_up

        if is_3f_swipe or is_4f_swipe:
            # Gracefully clean up states
            if self._state == self._ZOOM:
                self._exit_zoom()
                self._state = self._IDLE
            elif self._state in (self._PINCHING, self._RIGHT_PINCHING):
                self._state = self._IDLE
                self._pinch_start_ts = None
                self._pinch_start_pos = None

            # Process Swipe
            if getattr(self, '_swipe_start_x', None) is None:
                self._swipe_start_x = pos[0]
                self._swipe_fired = False
                
            if not self._swipe_fired:
                delta_x = pos[0] - self._swipe_start_x
                if abs(delta_x) > self.SWIPE_THRESHOLD_PX:
                    self._swipe_fired = True
                    direction = "RIGHT" if delta_x > 0 else "LEFT"
                    prefix = "3F" if is_3f_swipe else "4F"
                    return ActionEvent(f"SWIPE_{prefix}_{direction}", pos)
                    
            return ActionEvent("MOVE", pos)
        else:
            self._swipe_start_x = None
            self._swipe_fired = False

        # 2. Zoom — highest priority when EXACTLY index + middle are visible.
        is_zoom_pose = fs.index_up and fs.middle_up and not fs.ring_up and not fs.pinky_up

        if self._state == self._ZOOM and not is_zoom_pose:
            # Handle dropped frames/graceful exit
            if self._zoom_lost_ts is None:
                self._zoom_lost_ts = fs.ts
                
            if fs.ts - self._zoom_lost_ts > self.ZOOM_GRACE_SEC:
                # Grace period expired, definitely left zoom mode
                self._exit_zoom()
                self._zoom_lost_ts = None
                self._state = self._IDLE
            else:
                # Inside grace period: hold zoom state, wait for fingers to return
                return ActionEvent("MOVE", pos)
                
        elif is_zoom_pose:
            self._zoom_lost_ts = None  # recover from any grace period
            
            # End any in-progress zoom cleanly
            if self._state in (self._PINCHING, self._RIGHT_PINCHING):
                self._state = self._ZOOM
                self._pinch_start_ts = None
                self._pinch_start_pos = None
                self._enter_zoom(fs)

            # Capture anchor on first zoom frame.
            if self._state != self._ZOOM:
                self._state = self._ZOOM
                self._enter_zoom(fs)

            return self._update_zoom(fs, pos)

        # 3. Pinch gesture: index up and thumb close to index.
        pinch_mode = fs.index_up and not fs.middle_up

        if pinch_mode:
            if fs.pinch_dist < self.PINCH_CLOSE_PX:
                # ── Fingers closing / staying closed ──────────────────────────
                if self._state != self._PINCHING:
                    self._state = self._PINCHING
                    self._pinch_start_pos = pos

                return ActionEvent("MOVE", self._pinch_start_pos)

            if fs.pinch_dist > self.PINCH_OPEN_PX:
                # ── Fingers open (release) ────────────────────────────────────
                if self._state == self._PINCHING:
                    self._state = self._IDLE
                    stored_pos = self._pinch_start_pos or pos
                    self._pinch_start_pos = None
                    return self._on_pinch_release(stored_pos)

            # ── Grey zone (CLOSE <= dist <= OPEN) ─────────────────────────
            # Hold current state, just move.
            if self._state == self._PINCHING:
                # Still deciding, keep frozen so user doesn't miss the button while separating fingers
                return ActionEvent("MOVE", self._pinch_start_pos or pos)

        # 3.5 Right click pinch: middle finger up, index down, and thumb close to middle.
        right_pinch_mode = fs.middle_up and not fs.index_up

        if right_pinch_mode:
            if fs.middle_pinch_dist < self.PINCH_CLOSE_PX:
                if self._state != self._RIGHT_PINCHING:
                    self._state = self._RIGHT_PINCHING
                    self._pinch_start_pos = pos
                return ActionEvent("MOVE", self._pinch_start_pos)
            
            if fs.middle_pinch_dist > self.PINCH_OPEN_PX:
                if self._state == self._RIGHT_PINCHING:
                    self._state = self._IDLE
                    stored_pos = self._pinch_start_pos or pos
                    self._pinch_start_pos = None
                    return self._on_right_pinch_release(stored_pos)
            
            if self._state == self._RIGHT_PINCHING:
                return ActionEvent("MOVE", self._pinch_start_pos or pos)

        # 4. Clean up stale pinch state if we left pinch mode unexpectedly.
        if self._state in (self._PINCHING, self._RIGHT_PINCHING):
            self._state = self._IDLE
            self._pinch_start_pos = None

        # 6. Regular cursor movement.
        self._state = self._MOVING
        return ActionEvent("MOVE", pos)
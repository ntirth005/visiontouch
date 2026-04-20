def normalize_rotation_deg(rotation_deg: int) -> int:
    deg = int(rotation_deg) % 360
    if deg not in (0, 90, 180, 270):
        raise ValueError("camera_rotation_deg must be one of: 0, 90, 180, 270")
    return deg

import cv2
import sys
from screen_locator import detect_template_on_screen

def test_static_images(screen_path: str, template_path: str, threshold: float = 0.6):
    print(f"Loading screen image: {screen_path}")
    screen_bgr = cv2.imread(screen_path)
    
    print(f"Loading template image: {template_path}")
    template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if screen_bgr is None:
        print(f"Error: Could not read screen image at '{screen_path}'")
        return
    if template_gray is None:
        print(f"Error: Could not read template image at '{template_path}'")
        return

    print(f"Running detection (Threshold={threshold})...")
    matches, result_img = detect_template_on_screen(screen_bgr, template_gray, match_threshold=threshold)
    
    if matches:
        print(f"Success! Found {len(matches)} match(es).")
        for i, match in enumerate(matches):
            x, y, w, h, conf = match
            print(f"  Match {i+1}: Location ({x}, {y}), Size {w}x{h}, Confidence: {conf:.3f}")
    else:
        print("No matches found above the threshold.")

    # Show the result
    cv2.namedWindow("Test Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Test Result", result_img)
    print("Press any key in the image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # You can change these paths to your test images
    SCREEN_IMAGE = "img3.png"
    TEMPLATE_IMAGE = "img3CA.png"
    
    # If passed via command line arguments
    if len(sys.argv) == 3:
        SCREEN_IMAGE = sys.argv[1]
        TEMPLATE_IMAGE = sys.argv[2]
        
    test_static_images(SCREEN_IMAGE, TEMPLATE_IMAGE, threshold=0.9)

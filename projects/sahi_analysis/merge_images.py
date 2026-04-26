#!/usr/bin/env python3
"""
Simple script to merge two images side by side
Usage: python merge_images.py image1.jpg image2.jpg output.jpg
       python merge_images.py image1.jpg image2.jpg  # auto output name
"""

import sys
import cv2
import numpy as np
from pathlib import Path


def merge_horizontal(img1_path, img2_path, output_path=None, gap=0):
    """Merge two images horizontally (side by side) with optional gap"""
    # Read images
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None:
        print(f"❌ Error: Cannot read {img1_path}")
        return False
    if img2 is None:
        print(f"❌ Error: Cannot read {img2_path}")
        return False
    
    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Make heights equal (resize to min height)
    target_height = min(h1, h2)
    if h1 != target_height:
        new_w1 = int(w1 * target_height / h1)
        img1 = cv2.resize(img1, (new_w1, target_height))
    if h2 != target_height:
        new_w2 = int(w2 * target_height / h2)
        img2 = cv2.resize(img2, (new_w2, target_height))
    
    # Merge horizontally with gap
    if gap > 0:
        # Create white gap
        h = img1.shape[0]
        gap_img = np.ones((h, gap, 3), dtype=np.uint8) * 255
        merged = cv2.hconcat([img1, gap_img, img2])
    else:
        merged = cv2.hconcat([img1, img2])
    
    # Generate output path if not provided
    if output_path is None:
        stem1 = Path(img1_path).stem
        stem2 = Path(img2_path).stem
        output_path = f"merged_{stem1}_{stem2}.jpg"
    
    # Save
    cv2.imwrite(str(output_path), merged)
    print(f"✅ Merged image saved: {output_path}")
    print(f"   Size: {merged.shape[1]}×{merged.shape[0]}")
    
    return True


def merge_vertical(img1_path, img2_path, output_path=None, gap=0):
    """Merge two images vertically (top and bottom) with optional gap"""
    # Read images
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        print(f"❌ Error: Cannot read images")
        return False
    
    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Make widths equal (resize to min width)
    target_width = min(w1, w2)
    if w1 != target_width:
        new_h1 = int(h1 * target_width / w1)
        img1 = cv2.resize(img1, (target_width, new_h1))
    if w2 != target_width:
        new_h2 = int(h2 * target_width / w2)
        img2 = cv2.resize(img2, (target_width, new_h2))
    
    # Merge vertically with gap
    if gap > 0:
        # Create white gap
        w = img1.shape[1]
        gap_img = np.ones((gap, w, 3), dtype=np.uint8) * 255
        merged = cv2.vconcat([img1, gap_img, img2])
    else:
        merged = cv2.vconcat([img1, img2])
    
    # Generate output path if not provided
    if output_path is None:
        stem1 = Path(img1_path).stem
        stem2 = Path(img2_path).stem
        output_path = f"merged_{stem1}_{stem2}_vertical.jpg"
    
    # Save
    cv2.imwrite(str(output_path), merged)
    print(f"✅ Merged image saved: {output_path}")
    print(f"   Size: {merged.shape[1]}×{merged.shape[0]}")
    
    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: python merge_images.py <image1> <image2> [output] [options]")
        print("\nOptions:")
        print("  --vertical, -v        Merge vertically (default: horizontal)")
        print("  --gap N, -g N         Add white gap of N pixels between images (default: 0)")
        print("\nExamples:")
        print("  python merge_images.py img1.jpg img2.jpg")
        print("  python merge_images.py img1.jpg img2.jpg output.jpg")
        print("  python merge_images.py img1.jpg img2.jpg --gap 20")
        print("  python merge_images.py img1.jpg img2.jpg output.jpg --gap 30")
        print("  python merge_images.py img1.jpg img2.jpg --vertical --gap 20")
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    
    # Parse arguments
    vertical = False
    output_path = None
    gap = 0
    
    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ['--vertical', '-v', '--v']:
            vertical = True
            i += 1
        elif arg in ['--gap', '-g']:
            if i + 1 < len(sys.argv):
                try:
                    gap = int(sys.argv[i + 1])
                    i += 2
                except ValueError:
                    print(f"❌ Error: Invalid gap value: {sys.argv[i + 1]}")
                    sys.exit(1)
            else:
                print(f"❌ Error: --gap requires a value")
                sys.exit(1)
        elif not arg.startswith('-'):
            output_path = arg
            i += 1
        else:
            i += 1
    
    print(f"\n{'='*60}")
    print(f"📷 Image Merger")
    print(f"{'='*60}")
    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    print(f"Mode: {'Vertical' if vertical else 'Horizontal'}")
    print(f"Gap: {gap} pixels")
    print(f"{'='*60}\n")
    
    # Merge images
    if vertical:
        success = merge_vertical(img1_path, img2_path, output_path, gap)
    else:
        success = merge_horizontal(img1_path, img2_path, output_path, gap)
    
    if success:
        print(f"\n{'='*60}")
        print("✅ Done!")
        print(f"{'='*60}\n")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()


import cv2
import numpy as np
import argparse
import os

def generate_stroboscopic_image(video_path, output_image_path, threshold=50, blend_ratio=1.0, blur_size=5, open_kernel_size=5):
    video = cv2.VideoCapture(video_path)
    
    # Read the first frame and set it as the base frame
    ret, base_frame = video.read()
    if not ret:
        print("Failed to read the first frame.")
        return
    
    prev_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    accumulator = np.zeros_like(base_frame, dtype=np.float32)
    change_count = np.zeros_like(base_frame, dtype=np.float32)
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute the absolute difference between the current frame and the previous frame
        diff = cv2.absdiff(current_gray, prev_gray)
        
        # Threshold the difference to create a binary mask
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Apply Gaussian blur to smoothen the mask edges
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        # Convert the blurred mask back to binary format
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Erode and then dilate the mask (opening operation) to remove noise
        kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Use the mask to extract the moving object from the current frame
        moving_object = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Accumulate the changes
        accumulator += moving_object.astype(np.float32)
        
        # Increase the change count wherever there's a change
        change_count += (mask[:, :, np.newaxis] > 0).astype(np.float32)
        
        prev_gray = current_gray
    
    # Average the accumulated image based on the count of changes
    with np.errstate(divide='ignore', invalid='ignore'):
        averaged_changes = np.divide(accumulator, change_count)
        averaged_changes[~np.isfinite(averaged_changes)] = base_frame[~np.isfinite(averaged_changes)]

    normalized_averaged_changes = cv2.normalize(averaged_changes, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    normalized_averaged_changes = cv2.addWeighted(normalized_averaged_changes, blend_ratio, base_frame, 1 - blend_ratio, 0)
    cv2.imwrite(output_image_path, normalized_averaged_changes)
    video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a stroboscopic image from a video.')
    parser.add_argument('input_video_path', type=str, help='Path to the input video file')
    parser.add_argument('--output', type=str, default='stroboscopic_image.jpg',
                        help='Path to save the stroboscopic image (default: stroboscopic_image.jpg in the current directory)')
    parser.add_argument('--threshold', type=int, default=50, help='Difference detection threshold')
    parser.add_argument('--blend_ratio', type=float, default=1.0, help='Blend ratio between averaged changes and base frame')
    parser.add_argument('--blur_size', type=int, default=5, help='Gaussian blur kernel size (must be odd)')
    parser.add_argument('--open_kernel_size', type=int, default=5, help='Size of the structuring element for morphological operations')
    
    args = parser.parse_args()

    if args.blur_size % 2 == 0:
        raise ValueError("Blur size must be odd.")

    generate_stroboscopic_image(args.input_video_path, args.output, args.threshold, args.blend_ratio, args.blur_size, args.open_kernel_size)

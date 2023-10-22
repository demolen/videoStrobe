import cv2
import numpy as np
import argparse
import os
from pytube import YouTube
import re
import random
import datetime

def postfix_datetime(filename):
    """
    Postfixes the filename with the current date and time.
    """
    base, ext = os.path.splitext(filename)
    datetime_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base}_{datetime_str}{ext}"
def sanitize_filename(filename):
    """
    Removes or replaces characters that aren't valid in file names.
    """
    s = re.sub(r'[^\w\s-]', '', filename)  # Remove all non-word characters (everything except numbers, letters, and -_)
    s = re.sub(r'\s+', '-', s).strip()  # Replace all runs of whitespace with a single dash
    return s

def download_youtube_video(url, download_folder='temp_video_download'):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    yt = YouTube(url)
    video_title = sanitize_filename(yt.title)
    video_path = os.path.join(download_folder, f"{video_title}.mp4")

    if os.path.exists(video_path):
        use_cached = input(f"'{video_title}.mp4' is already downloaded. Use cached version? (y/n): ").lower()
        if use_cached == 'y':
            print("Using cached video...")
            return video_path
        elif use_cached == 'n':
            print("Redownloading video...")

    ys = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
    return ys.download(download_folder, video_title)
def generate_stroboscopic_image(video_path, output_image_path, threshold=50, blend_ratio=1.0, blur_size=5, open_kernel_size=5, frame_interval=1, duration_range=None, random_range=None):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = 0
    end_frame = total_frames

    if duration_range:
        start_seconds, end_seconds = duration_range
        start_frame = int(start_seconds * fps)
        end_frame = int(end_seconds * fps)

    if random_range:
        random_duration_frames = int(random_range * fps)
        start_frame = random.randint(start_frame, end_frame - random_duration_frames)
        end_frame = start_frame + random_duration_frames

    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, base_frame = video.read()
    if not ret:
        print("Failed to read the first frame.")
        return

    prev_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    accumulator = np.zeros_like(base_frame, dtype=np.float32)
    change_count = np.zeros_like(base_frame, dtype=np.float32)

    frame_count = start_frame
    while frame_count < end_frame:
        ret, frame = video.read()
        frame_count += 1
        if not ret:
            break
        if (frame_count - start_frame) % frame_interval != 0:
            continue

        
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
    parser.add_argument('--add_datetime_postfix', action='store_true',
                        help='Option to postfix the output filename with the current date and time.')
    parser.add_argument('input', type=str, help='Path to the input video file or a YouTube URL')
    parser.add_argument('--output', type=str, default='stroboscopic_image.jpg',
                        help='Path to save the stroboscopic image (default: stroboscopic_image.jpg in the current directory)')
    parser.add_argument('--threshold', type=int, default=50, help='Difference detection threshold')
    parser.add_argument('--blend_ratio', type=float, default=1.0, help='Blend ratio between averaged changes and base frame')
    parser.add_argument('--duration_range', type=lambda s: [float(item) for item in s.split(',')], default=None, help='Specify the start and end times in seconds to process, e.g., 10,20 for the 10th to 20th second.')
    parser.add_argument('--random_range', type=float, default=None, help='Duration in seconds to randomly select within the specified or full video duration.')

    parser.add_argument('--blur_size', type=int, default=5, help='Gaussian blur kernel size (must be odd)')
    parser.add_argument('--open_kernel_size', type=int, default=5, help='Size of the structuring element for morphological operations')
    parser.add_argument('--frame_interval', type=int, default=1, help='Interval for sampling frames from the video (default: 1 to process every frame)')
    
    args = parser.parse_args()

    if args.blur_size % 2 == 0:
        raise ValueError("Blur size must be odd.")
    
    input_path = args.input

    # If the flag is provided, postfix the output filename with the current date and time
    if args.add_datetime_postfix:
        args.output = postfix_datetime(args.output)

    # Check if the input is a YouTube URL
    if "youtube.com" in input_path or "youtu.be" in input_path:
        print("Detected YouTube URL. Downloading video...")
        input_path = download_youtube_video(input_path)

    generate_stroboscopic_image(input_path, args.output, args.threshold, args.blend_ratio, args.blur_size, args.open_kernel_size, args.frame_interval, args.duration_range, args.random_range)

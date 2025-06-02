import cv2
import numpy as np
import argparse
import os
import yt_dlp
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
    s = re.sub(r'[^\w\s-]', '', filename)
    s = re.sub(r'\s+', '-', s).strip()
    return s

def download_youtube_video(url, download_folder='temp_video_download'):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    ydl_opts = {
        'format': 'bestvideo',
        'outtmpl': os.path.join(download_folder, '%(title)s.%(ext)s'),
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            video_title = sanitize_filename(info['title'])
            video_path = os.path.join(download_folder, f"{video_title}.mp4")
            
            if os.path.exists(video_path):
                use_cached = input(f"'{video_title}.mp4' is already downloaded. Use cached version? (y/n): ").lower()
                if use_cached == 'y':
                    print("Using cached video...")
                    return video_path
                elif use_cached == 'n':
                    print("Redownloading video...")
            
            ydl_opts['outtmpl'] = os.path.join(download_folder, f"{video_title}.%(ext)s")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                ydl_download.download([url])
            
            for file in os.listdir(download_folder):
                if file.startswith(video_title):
                    return os.path.join(download_folder, file)
            
            return video_path
            
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None

def create_stable_background(video, start_frame, end_frame, sample_interval=30):
    """
    Creates a stable background by taking median of sampled frames.
    This helps eliminate temporary objects and creates a cleaner base.
    """
    print("Creating stable background...")
    frames = []
    
    for frame_idx in range(start_frame, min(end_frame, start_frame + 300), sample_interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if ret:
            frames.append(frame)
    
    if not frames:
        return None
    
    # Calculate median frame for stable background
    frames_array = np.array(frames)
    background = np.median(frames_array, axis=0).astype(np.uint8)
    return background

def enhanced_motion_detection(current_gray, background_gray, method='frame_diff', background_subtractor=None):
    """
    Enhanced motion detection with multiple methods.
    """
    if method == 'background_sub' and background_subtractor is not None:
        mask = background_subtractor.apply(current_gray)
    else:  # frame_diff
        diff = cv2.absdiff(current_gray, background_gray)
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return mask

def refine_motion_mask(mask, min_area=100, blur_size=5, morph_kernel_size=5):
    """
    Refines the motion mask with better noise reduction and gap filling.
    """
    # Gaussian blur to smooth edges
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    
    # Threshold back to binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological operations - closing followed by opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    
    # Closing: fill gaps in moving objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Opening: remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Remove small components
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.fillPoly(mask, [contour], 0)
    
    return mask

def generate_enhanced_stroboscopic_image(video_path, output_image_path, threshold=50, 
                                       blend_ratio=1.0, blur_size=5, morph_kernel_size=5, 
                                       frame_interval=1, duration_range=None, random_range=None,
                                       motion_method='frame_diff', background_samples=10,
                                       min_motion_area=100, decay_factor=0.95):
    """
    Enhanced stroboscopic image generation with better background handling and motion detection.
    """
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {total_frames} frames at {fps} FPS")

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

    print(f"Processing frames {start_frame} to {end_frame}")

    # Create stable background
    stable_background = create_stable_background(video, start_frame, end_frame, 
                                               max(1, (end_frame - start_frame) // background_samples))
    
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, first_frame = video.read()
    if not ret:
        print("Failed to read the first frame.")
        return

    # Use stable background if available, otherwise use first frame
    base_frame = stable_background if stable_background is not None else first_frame
    base_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize background subtractor if requested
    background_subtractor = None
    if motion_method == 'background_sub':
        background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        # Train background subtractor with some initial frames
        for i in range(min(50, end_frame - start_frame)):
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
            ret, frame = video.read()
            if ret:
                background_subtractor.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # Initialize accumulators
    motion_accumulator = np.zeros_like(base_frame, dtype=np.float32)
    weight_accumulator = np.zeros(base_frame.shape[:2], dtype=np.float32)
    
    prev_gray = base_gray.copy()
    frame_count = start_frame
    processed_frames = 0

    print("Processing frames...")
    
    while frame_count < end_frame:
        ret, frame = video.read()
        frame_count += 1
        if not ret:
            break
        if (frame_count - start_frame) % frame_interval != 0:
            continue

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced motion detection
        if motion_method == 'background_sub':
            motion_mask = enhanced_motion_detection(current_gray, None, motion_method, background_subtractor)
        else:
            motion_mask = enhanced_motion_detection(current_gray, prev_gray, motion_method)
        
        # Refine the motion mask
        refined_mask = refine_motion_mask(motion_mask, min_motion_area, blur_size, morph_kernel_size)
        
        # Convert mask to 3-channel for color operations
        mask_3ch = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Extract moving parts with weighted accumulation
        motion_intensity = np.sum(refined_mask) / (refined_mask.shape[0] * refined_mask.shape[1])
        
        if motion_intensity > 0.001:  # Only accumulate if there's significant motion
            # Apply temporal decay to give more weight to recent frames
            weight = (decay_factor ** (processed_frames / 10)) if processed_frames > 0 else 1.0
            
            # Accumulate motion with intensity-based weighting
            motion_part = frame.astype(np.float32) * mask_3ch
            motion_accumulator += motion_part * weight
            weight_accumulator += (refined_mask / 255.0) * weight
        
        prev_gray = current_gray
        processed_frames += 1
        
        if processed_frames % 50 == 0:
            print(f"Processed {processed_frames} frames...")

    print(f"Total processed frames: {processed_frames}")

    # Create final stroboscopic image
    # Avoid division by zero
    weight_accumulator_3ch = np.stack([weight_accumulator] * 3, axis=2)
    valid_pixels = weight_accumulator_3ch > 0.1
    
    # Initialize result with base background
    result = base_frame.astype(np.float32)
    
    # Blend accumulated motion where there was significant movement
    motion_average = np.zeros_like(motion_accumulator)
    motion_average[valid_pixels] = motion_accumulator[valid_pixels] / weight_accumulator_3ch[valid_pixels]
    
    # Adaptive blending - stronger blend where more motion occurred
    adaptive_blend = np.clip(weight_accumulator / np.max(weight_accumulator) if np.max(weight_accumulator) > 0 else 0, 0, 1)
    adaptive_blend_3ch = np.stack([adaptive_blend] * 3, axis=2)
    
    # Final composition
    result = result * (1 - adaptive_blend_3ch * blend_ratio) + motion_average * adaptive_blend_3ch * blend_ratio
    
    # Normalize and save
    result = np.clip(result, 0, 255).astype(np.uint8)
    cv2.imwrite(output_image_path, result)
    
    print(f"Stroboscopic image saved to: {output_image_path}")
    video.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate an enhanced stroboscopic image from a video.')
    parser.add_argument('--add_datetime_postfix', action='store_true',
                        help='Postfix output filename with current date and time')
    parser.add_argument('input', type=str, help='Path to input video file or YouTube URL')
    parser.add_argument('--output', type=str, default='stroboscopic_image.jpg',
                        help='Output image path (default: stroboscopic_image.jpg)')
    parser.add_argument('--threshold', type=int, default=50, 
                        help='Motion detection threshold (default: 50)')
    parser.add_argument('--blend_ratio', type=float, default=0.8, 
                        help='Blend ratio for motion overlay (default: 0.8)')
    parser.add_argument('--duration_range', type=lambda s: [float(item) for item in s.split(',')], 
                        default=None, help='Time range to process: start,end in seconds')
    parser.add_argument('--random_range', type=float, default=None, 
                        help='Random duration in seconds to select from video')
    parser.add_argument('--blur_size', type=int, default=5, 
                        help='Gaussian blur kernel size (must be odd, default: 5)')
    parser.add_argument('--morph_kernel_size', type=int, default=5, 
                        help='Morphological operation kernel size (default: 5)')
    parser.add_argument('--frame_interval', type=int, default=1, 
                        help='Frame sampling interval (default: 1)')
    parser.add_argument('--motion_method', type=str, choices=['frame_diff', 'background_sub'], 
                        default='frame_diff', help='Motion detection method (default: frame_diff)')
    parser.add_argument('--background_samples', type=int, default=10, 
                        help='Number of frames to sample for background creation (default: 10)')
    parser.add_argument('--min_motion_area', type=int, default=100, 
                        help='Minimum area for motion detection (default: 100)')
    parser.add_argument('--decay_factor', type=float, default=0.95, 
                        help='Temporal decay factor for frame weighting (default: 0.95)')
    
    args = parser.parse_args()

    if args.blur_size % 2 == 0:
        raise ValueError("Blur size must be odd.")
    
    input_path = args.input

    if args.add_datetime_postfix:
        args.output = postfix_datetime(args.output)

    # Handle YouTube URLs
    if "youtube.com" in input_path or "youtu.be" in input_path:
        print("Detected YouTube URL. Downloading video...")
        input_path = download_youtube_video(input_path)
        if input_path is None:
            print("Failed to download video. Exiting.")
            exit(1)

    generate_enhanced_stroboscopic_image(
        input_path, args.output, args.threshold, args.blend_ratio, 
        args.blur_size, args.morph_kernel_size, args.frame_interval, 
        args.duration_range, args.random_range, args.motion_method,
        args.background_samples, args.min_motion_area, args.decay_factor
    )
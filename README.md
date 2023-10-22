# Stroboscopic Image Generator

Generate a stroboscopic image from a video. This tool allows you to extract the moving parts of a video into a single image. This script can also download YouTube videos for processing.

## Dependencies:

- OpenCV (`cv2`)
- `numpy`
- `argparse`
- `os`
- `pytube`
- `re`
- `random`
- `datetime`

## Usage

```bash
python script_name.py [OPTIONS] INPUT

Arguments:

    input: Path to the input video file or a YouTube URL (required).
    --output: Path to save the stroboscopic image. Default: stroboscopic_image.jpg in the current directory.
    --add_datetime_postfix: Option to postfix the output filename with the current date and time.
    --threshold: Difference detection threshold. Default: 50.
    --blend_ratio: Blend ratio between averaged changes and base frame. Default: 1.0.
    --duration_range: Specify the start and end times in seconds to process. For example, use 10,20 for the 10th to 20th second.
    --random_range: Duration in seconds to randomly select within the specified or full video duration.
    --blur_size: Gaussian blur kernel size (must be odd). Default: 5.
    --open_kernel_size: Size of the structuring element for morphological operations. Default: 5.
    --frame_interval: Interval for sampling frames from the video (default: 1 to process every frame).

Example:

To generate a stroboscopic image from a local video:

bash

python script_name.py my_video.mp4

To generate a stroboscopic image from a YouTube video:

bash

python script_name.py "https://www.youtube.com/watch?v=example"

Note:

Ensure you have the required dependencies installed. You can use pip to install them:

bash

pip install opencv-python pytube numpy

License

This script is provided as-is without any guarantees or warranty. Users are free to modify and distribute it under the terms of their choice.

vbnet


Replace `script_name.py` with the actual name of your script file. You can also add or modify the provided text to suit your needs.

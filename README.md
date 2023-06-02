# Label Studio to YOLO Video Annotations Converter

## Description

This repository features a Python script that processes video annotations created using Label Studio, converting them into a format suitable for the YOLO (You Only Look Once) object detection model. The script interpolates bounding boxes for each intermediate frame based on key-frame annotations (as needed) and exports the labels (i.e., bounding box coordinates), optionally along with the corresponding frames, into a YOLO-compatible format. As of Label Studio version 1.7.0, such functionality isn't inherently available. Please note that video annotations should be exported in the JSON-MIN format from Label Studio.

## Usage

1. Export your Label Studio video annotations in JSON-MIN format.
2. Run the provided Python script from the command line as follows:
   ```console
   python ls2yolo.py --json_path path_to_your_json_annotations --video_path path_to_your_video_file --output_base path_to_output_directory
   ```
   `video_path` is optional. If not provided, the script only generates YOLO labels without extracting frames from the video.

Please refer to the comments in the Python script for more detailed instructions.

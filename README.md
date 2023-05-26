# Label Studio Video Annotations to YOLO Format Converter

## Description

This small project features a notebook that processes video annotations created using Label Studio, converting them into a format suitable for the YOLO (You Only Look Once) object detection model. The script has the capability to interpolate bounding boxes for each individual frame based on key-frame annotations (as needed), and export these labels (i.e., bounding box coordinates), along with the corresponding frames, into a YOLO-compatible format. As it stands with Label Studio version 1.7.0, such functionality isn't inherently available. Please note that video annotations should be exported in the JSON-MIN format.

## Usage

1. Export your Label Studio video annotations in JSON-MIN format.
2. Specify the paths to annotations and video files in the Jupyter notebook.
3. Run the provided Jupyter notebook.

Please refer to the comments in the notebook for more detailed instructions.

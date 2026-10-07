#!/bin/bash

set -e
set -o pipefail

# --- Usage Function ---
print_usage() {
  echo "Usage: $0 [OPTIONS] <input_directory> [output_video_path]"
  echo ""
  echo "Creates a video from a sequence of images named 'rgb_XXXXXX.png' found in the input directory."
  echo ""
  echo "Arguments:"
  echo "  <input_directory>      Path to the directory containing the 'rgb_*.png' images."
  echo "  [output_video_path]    Optional. The full path for the output video file."
  echo "                         Defaults to '<input_directory>/output.mp4'."
  echo ""
  echo "Options:"
  echo "  -h, --help             Display this help message and exit."
}

# --- Argument Parsing ---

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_usage
      exit 0
      ;;
    -*)
      echo "Error: Unknown option: $1" >&2
      print_usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

# --- Input Validation ---

INPUT_DIR="${1}"
OUTPUT_VIDEO="${2}" # This will be empty if not provided

if [[ -z "$INPUT_DIR" ]]; then
  echo "Error: Input directory not specified." >&2
  print_usage
  exit 1
fi

if ! find "$INPUT_DIR" -maxdepth 1 -type f -name 'rgb_*.png' -print -quit | grep -q .; then
    echo "Error: No source images matching 'rgb_*.png' found in '$INPUT_DIR'." >&2
    exit 1
fi

if [[ -z "$OUTPUT_VIDEO" ]]; then
  OUTPUT_VIDEO="$INPUT_DIR/output.mp4"
fi


# --- Main Logic ---

echo "--- Video Creation Started ---"
echo "Input directory:  $INPUT_DIR"
echo "Output video:     $OUTPUT_VIDEO"
echo "--------------------------------"

IMAGE_PATTERN="$INPUT_DIR/rgb_%06d.png"

echo "Running ffmpeg..."

# The core ffmpeg command
ffmpeg -framerate 30 -i "$IMAGE_PATTERN" -c:v libx264 -crf 20 -pix_fmt yuv420p -y "$OUTPUT_VIDEO"


echo ""
echo "Video successfully created at: $OUTPUT_VIDEO"
echo ""


echo "--- Script Finished ---"
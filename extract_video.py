import cv2
import os
from pathlib import Path
import sys

def extract_frames_from_video(video_path, output_folder, frame_interval=1, image_format='jpg', quality=95):
    """
    Extract frames from a video file and save them as images.
    
    Parameters:
    -----------
    video_path : str
        Path to the input video file (.mov or other formats)
    output_folder : str
        Path to the folder where frames will be saved
    frame_interval : int
        Extract every nth frame (1 = all frames, 2 = every other frame, etc.)
    image_format : str
        Output image format ('jpg', 'png', 'bmp', etc.)
    quality : int
        JPEG quality (1-100), only used for JPEG format
    
    Returns:
    --------
    tuple : (total_frames_processed, frames_saved)
    """
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Information:")
    print(f"- FPS: {fps}")
    print(f"- Total Frames: {total_frames}")
    print(f"- Resolution: {width}x{height}")
    print(f"- Duration: {total_frames/fps:.2f} seconds")
    print(f"\nExtracting every {frame_interval} frame(s)...")
    
    frame_count = 0
    saved_count = 0
    
    # Set compression parameters
    if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif image_format.lower() == 'png':
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]  # 0-9, 9 is max compression
    else:
        encode_params = []
    
    while True:
        # Read the next frame
        success, frame = video_capture.read()
        
        if not success:
            break
        
        # Save frame if it matches the interval
        if frame_count % frame_interval == 0:
            # Generate filename with zero-padding
            filename = f"frame_{frame_count:06d}.{image_format}"
            filepath = os.path.join(output_folder, filename)
            
            # Save the frame
            cv2.imwrite(filepath, frame, encode_params)
            saved_count += 1
            
            # Show progress every 100 saved frames
            if saved_count % 100 == 0:
                print(f"Saved {saved_count} frames... ({frame_count}/{total_frames} processed)")
        
        frame_count += 1
    
    # Release the video capture object
    video_capture.release()
    
    print(f"\nExtraction complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames saved: {saved_count}")
    
    return frame_count, saved_count

def batch_extract_frames(video_paths, output_base_folder, **kwargs):
    """
    Extract frames from multiple video files.
    
    Parameters:
    -----------
    video_paths : list
        List of paths to video files
    output_base_folder : str
        Base folder where each video's frames will be saved in subfolders
    **kwargs : dict
        Additional arguments to pass to extract_frames_from_video
    """
    
    for video_path in video_paths:
        video_name = Path(video_path).stem
        output_folder = os.path.join(output_base_folder, video_name)
        
        print(f"\nProcessing: {video_path}")
        print(f"Output folder: {output_folder}")
        
        try:
            extract_frames_from_video(video_path, output_folder, **kwargs)
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Single video extraction
    video_file = "IMG_9182.mov"  # Replace with your .mov file path
    output_directory = "extracted_frames_9182"  # Output folder name
    
    # Basic extraction - save all frames
    extract_frames_from_video(video_file, output_directory)
    
    # Advanced options examples:
    
    # Extract every 5th frame as PNG with high quality
    # extract_frames_from_video(video_file, "frames_png", frame_interval=5, image_format='png')
    
    # Extract all frames as JPEG with 85% quality
    # extract_frames_from_video(video_file, "frames_jpg_85", image_format='jpg', quality=85)
    
    # Extract frames at 1 fps (approximately)
    # video_cap = cv2.VideoCapture(video_file)
    # fps = video_cap.get(cv2.CAP_PROP_FPS)
    # video_cap.release()
    # extract_frames_from_video(video_file, "frames_1fps", frame_interval=int(fps))
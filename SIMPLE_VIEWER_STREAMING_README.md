# Simple Upload Viewer - IP Camera Streaming

This document describes the IP camera streaming functionality added to the Simple Upload Viewer.

## Features

### 1. IP Camera Streaming
- Connect to IP cameras via HTTP/RTSP streams
- Real-time frame capture and processing
- Automatic frame capture at configurable intervals
- Manual frame capture capability
- **Automatic inference processing** - Each captured frame is automatically processed by the loaded model

### 2. UI Controls
- **Camera URL Input**: Enter your IP camera stream URL
- **Capture Interval**: Set how often to automatically capture frames (in seconds)
- **Start Streaming**: Begin IP camera stream
- **Stop Streaming**: End IP camera stream
- **Capture Frame**: Manually capture current frame from stream

## Setup

### 1. IP Camera Configuration
Common IP camera URL formats:
```
http://192.168.1.100:8080/video
http://admin:password@192.168.1.100:8080/video
rtsp://192.168.1.100:554/stream1
http://192.168.1.100:8080/stream
```

### 2. Testing Connection
Run the test script to verify your IP camera connection:
```bash
python test_simple_viewer_streaming.py
```

## Usage

### 1. Load Model First
1. Start the Simple Upload Viewer: `python simple_upload_viewer.py`
2. Click "Load Model" and select your trained model (.pth file)
3. Wait for the model to load successfully

### 2. Start Streaming
1. Enter your IP camera URL in the "Camera URL" field
2. Set the capture interval (default: 5 seconds)
3. Click "▶ Start Streaming"
4. The tool will automatically capture frames and run inference

### 3. View Results
- **Original Image**: Shows the captured frame
- **Prediction Results**: Shows the model's analysis with:
  - **Visualization Tab**: Bounding boxes, quality indicators, and analysis overlay
  - **Detailed Results Tab**: Text-based analysis with confidence scores

### 4. Manual Capture
- Click "📷 Capture Frame" to manually capture and process the current frame
- Useful for capturing specific moments when automatic interval isn't suitable

### 5. Stop Streaming
- Click "⏹ Stop Streaming" to end the IP camera connection
- The tool will clean up resources automatically

## Key Differences from Battery Annotation Tool

### 1. Automatic Inference Processing
- **Simple Upload Viewer**: Automatically runs inference on every captured frame
- **Battery Annotation Tool**: Captures frames for manual annotation

### 2. Real-time Analysis
- **Simple Upload Viewer**: Shows live quality assessment and object detection
- **Battery Annotation Tool**: Focuses on manual annotation tools

### 3. Workflow
- **Simple Upload Viewer**: Load model → Start streaming → View automatic results
- **Battery Annotation Tool**: Load images → Manually annotate → Save annotations

## File Management

### Streamed Frames
- Captured frames are saved as `streamed_frame_[timestamp].jpg`
- Files are saved in the current working directory
- **Note**: Unlike the annotation tool, this viewer doesn't save annotation files

### Example File Structure
```
streamed_frame_1703123456.jpg
streamed_frame_1703123461.jpg
streamed_frame_1703123466.jpg
```

## Troubleshooting

### Common Issues

1. **Model Not Loaded**
   - Ensure you've loaded a trained model before starting streaming
   - The streaming buttons are disabled until a model is loaded

2. **Connection Failed**
   - Verify the IP camera URL is correct
   - Check if the camera is accessible from your network
   - Try different URL formats (HTTP vs RTSP)

3. **No Frames Received**
   - Check camera power and network connection
   - Verify camera supports the specified URL format
   - Try the test script to isolate the issue

4. **High CPU Usage**
   - Increase the capture interval to reduce processing frequency
   - Close other applications using the camera

### Testing Your Camera
1. Run `python test_simple_viewer_streaming.py`
2. The script will try multiple common URL formats
3. Check the console output for success/failure messages

## Technical Details

### Dependencies
- OpenCV (`cv2`) for video capture
- Threading for non-blocking streaming
- PIL for image processing
- PyTorch for model inference

### Architecture
- `IPCamera` class handles the camera connection
- Streaming runs in a separate thread to avoid blocking the UI
- Frame processing includes automatic inference using the loaded model
- Results are displayed in real-time with quality assessment

### Error Handling
- Automatic retry on connection failures
- Graceful cleanup on application exit
- User-friendly error messages
- Model validation before streaming

## Configuration

### Default Settings
- Default camera URL: `http://192.168.1.100:8080/video`
- Default capture interval: 5 seconds
- Frame format: JPEG
- Image processing: BGR to RGB conversion

### Customization
You can modify the default settings in the `__init__` method of `SimpleUploadViewer`:
```python
self.camera_url = "your_camera_url_here"
self.capture_interval = 10.0  # 10 seconds
```

## Workflow Comparison

| Feature | Simple Upload Viewer | Battery Annotation Tool |
|---------|---------------------|------------------------|
| **Purpose** | Real-time quality assessment | Manual annotation |
| **Model Required** | Yes (for inference) | No |
| **Processing** | Automatic inference | Manual annotation |
| **Output** | Quality assessment | Annotation files |
| **Use Case** | Production monitoring | Training data creation |

## Security Notes

- IP camera credentials are stored in plain text in the URL
- Consider using environment variables for sensitive credentials
- Ensure your network is secure when using IP cameras
- The viewer processes frames locally - no data is sent to external servers 
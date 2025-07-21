# IP Camera Streaming Functionality

This document describes the new IP camera streaming functionality added to the Enhanced Battery Annotation Tool.

## Features

### 1. IP Camera Streaming
- Connect to IP cameras via HTTP/RTSP streams
- Real-time frame capture and processing
- Automatic frame capture at configurable intervals
- Manual frame capture capability

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
python test_ip_camera.py
```

## Usage

### 1. Start Streaming
1. Enter your IP camera URL in the "Camera URL" field
2. Set the capture interval (default: 5 seconds)
3. Click "▶ Start Streaming"
4. The tool will automatically capture frames at the specified interval

### 2. Manual Capture
1. Start streaming as above
2. Click "📷 Capture Frame" to manually capture the current frame
3. The frame will be processed and displayed for annotation

### 3. Annotation
- Captured frames are processed exactly like uploaded images
- All annotation tools (holes, text, knobs, etc.) work the same way
- Save annotations using the "Save Annotations" button

### 4. Stop Streaming
- Click "⏹ Stop Streaming" to end the IP camera connection
- The tool will clean up resources automatically

## File Management

### Streamed Frames
- Captured frames are saved as `streamed_frame_[timestamp].jpg`
- Annotations are saved as `streamed_frame_[timestamp]_enhanced_annotation.json`
- Files are saved in the current working directory

### Example File Structure
```
streamed_frame_1703123456.jpg
streamed_frame_1703123456_enhanced_annotation.json
streamed_frame_1703123461.jpg
streamed_frame_1703123461_enhanced_annotation.json
```

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Verify the IP camera URL is correct
   - Check if the camera is accessible from your network
   - Try different URL formats (HTTP vs RTSP)

2. **No Frames Received**
   - Check camera power and network connection
   - Verify camera supports the specified URL format
   - Try the test script to isolate the issue

3. **High CPU Usage**
   - Increase the capture interval to reduce processing frequency
   - Close other applications using the camera

### Testing Your Camera
1. Run `python test_ip_camera.py`
2. The script will try multiple common URL formats
3. Check the console output for success/failure messages

## Technical Details

### Dependencies
- OpenCV (`cv2`) for video capture
- Threading for non-blocking streaming
- PIL for image processing

### Architecture
- `IPCamera` class handles the camera connection
- Streaming runs in a separate thread to avoid blocking the UI
- Frame processing is thread-safe with proper synchronization

### Error Handling
- Automatic retry on connection failures
- Graceful cleanup on application exit
- User-friendly error messages

## Configuration

### Default Settings
- Default camera URL: `http://192.168.1.100:8080/video`
- Default capture interval: 5 seconds
- Frame format: JPEG
- Image processing: BGR to RGB conversion

### Customization
You can modify the default settings in the `__init__` method of `EnhancedBatteryLabelingTool`:
```python
self.camera_url = "your_camera_url_here"
self.capture_interval = 10.0  # 10 seconds
```

## Security Notes

- IP camera credentials are stored in plain text in the URL
- Consider using environment variables for sensitive credentials
- Ensure your network is secure when using IP cameras 
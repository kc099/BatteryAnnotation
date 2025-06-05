# Battery Annotation Project

## Repository Structure and Large Files Management

This repository contains code for battery annotation, but intentionally excludes large media files and extracted frames to keep the repository size manageable. Here's how we handle different types of files:

### Ignored Directories and Files
- `extracted_frames/` - Contains extracted video frames (git-ignored)
- `extracted_frames_9182/` - Contains extracted frames from specific video (git-ignored)
- `labelled_frames/` - Contains annotated frames (git-ignored)
- `*.MOV` - All MOV video files (git-ignored)

### Best Practices for Media Files

1. **Video Files (.MOV)**
   - Store original video files in a separate location (not in git)
   - Use a consistent naming convention for videos
   - Document video metadata (resolution, duration, etc.) in a separate file

2. **Extracted Frames**
   - Frames are extracted from videos for annotation
   - These are stored locally but not in git
   - Use the provided scripts to extract frames consistently
   - Keep frame extraction parameters documented

3. **Labelled Frames**
   - Contains annotation data and processed frames
   - Stored locally but not in git
   - Use consistent annotation format
   - Back up annotation data separately

### Setting Up the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/kc099/BatteryAnnotation.git
   cd BatteryAnnotation
   ```

2. Create required directories (if they don't exist):
   ```bash
   mkdir -p extracted_frames extracted_frames_9182 labelled_frames
   ```

3. Place your video files in the appropriate location (outside git)

### Working with the Repository

1. **Adding New Videos**
   - Place new .MOV files in your working directory
   - They will be automatically ignored by git
   - Use the provided scripts to extract frames

2. **Extracting Frames**
   - Use the provided Python scripts to extract frames
   - Frames will be saved in the appropriate directory
   - These directories are git-ignored

3. **Annotation Process**
   - Annotate frames using the provided tools
   - Save annotations in the labelled_frames directory
   - Keep a backup of your annotations

### Backup Strategy

1. **Local Backups**
   - Regularly backup your media files and annotations
   - Consider using external storage for large files
   - Keep a separate backup of annotation data

2. **Version Control**
   - Only code and configuration files are version controlled
   - Use git for tracking code changes
   - Document any changes to the frame extraction or annotation process

### Troubleshooting

If you accidentally commit large files:
1. Use `git filter-branch` to remove them from history
2. Update .gitignore if needed
3. Force push changes to remote
4. Document the incident and solution

### Contributing

When contributing to this project:
1. Never commit media files or extracted frames
2. Follow the established directory structure
3. Document any changes to the frame extraction process
4. Update this README if you add new features or change the workflow

## Project Structure

```
BatteryAnnotation/
├── .gitignore           # Git ignore rules
├── README.md           # This file
├── scripts/            # Python scripts for processing
├── extracted_frames/   # Extracted frames (git-ignored)
├── extracted_frames_9182/ # Specific video frames (git-ignored)
└── labelled_frames/    # Annotated frames (git-ignored)
```

## Contact

For questions about the project or repository management, please contact the repository maintainer. 

# Dataset Merger Tool README
## Introduction
merge_lerobot_dataset is a powerful Python script dedicated to combining multiple data sets with similar structures. It is particularly applicable to lerobot self-mining datasets containing video, status and action data, which allow efficient integration of decentralized episodes and decentralized mission resources and facilitate subsequent training in models.

## II. OVERVIEW OF FUNCTIONS
1. ** datasets merge **: consolidate multiple source dataset folders into one unified data set and simplify data management.
2. ** index renumbered **: renumbered all episode indexes and task indexes to ensure continuity and consistency of merged data.
3. ** vector dimension fill **: Automatically detect and fill in vector dimension to make all data consistent in terms of observation.state and action.
4. ** statistical information merges **: smartly combines statistical information from multiple datasets and correctly handles complex data structures such as embedded structures for image characteristics.
5. ** Image Video Validation **: Automatically detects the quantitative relationship between the video/ image/ metadata files if a picture `images` folder is available.
6. ** video file processing **: correctly processing and copying video files, maintaining the correct indexing relationship between the video and other data, and supporting multiple video storage structures.
7. ** metadata update **: Update all metadata files to accurately reflect the integrated data set structure.
8. ** image file processing **: Copy the image file and maintain the correct indexing relationship between the picture and other data.


## III. Installation
This script relies on the following Python library:
- numpy
- pandas

Orders to install dependencies are as follows:
```bash
pip install numpy pandas
```

## IV. USE METHODS
### (i) Basic usage
Run tools by command line, for example:
```bash
python dataset_merger.py --sources /path/to/dataset1 /path/to/dataset2 /path/to/dataset3 --output /path/to/output_dataset
```
### (ii) Order line parameters
- ** - sources**: A list of source dataset folder paths needs to be specified at least one source dataset path.
- ** - output**: A folder path for the output data set to specify the storage location of the merged data set.
- ** - state_max_dim**: Maximum dimension of status vector, default 32.
- ** - action_max_dim**: Maximum dimension of the action vector, default 32.
- ** - fps**: Frame rate of the data set, default value 20.
- Whether **-copy_images**: copies and merges images from source folders to output folders.(default: `False`)

### (iii) Examples
```bash
python dataset_merger.py --sources ./robot_dataset_1 ./robot_dataset_2 --output ./merged_dataset --state_max_dim 32 --action_max_dim 18 --fps 30
```

## V. Format of the data set
This tool assumes that the input data set has the following structure:
```
dataset/
├── meta/
│   ├── episodes.jsonl       # Include Everyepisode_Other Organiser
│   ├── episodes_stats.jsonl # EachepisodeStatistics
│   ├── info.json           # Dataset global information
│   ├── stats.json          # Global statistical information
│   └── tasks.jsonl         # Task definition
├── data/                   # OrganisationparquetFormattedepisodeData
│   └── chunk-xxx/
│       └── episode_xxxxxx.parquet
├── images/                   # Optional Photo File
│   └── episode_xxxxxx/
│       └── frame_xxxxxx.png
└── videos/                 # Optional video files
    └── chunk-xxx/
        └── video_key/
            └── episode_xxxxxx.mp4 
```

## VI. FUNCTIONS
1. ** Data Consistency Processing **: Automatically detect and fill state and action vector dimensions to ensure that all data have consistent dimensions to meet machine learning algorithm requirements for data formats.
2. ** Index Management **: Renumber all episode and Job Indexes while maintaining frame index continuity and avoiding data confusion.
3. ** Consolidation of Statistical Information **: Smartly combining statistical data from multiple datasets allows the correct handling of complex data structures, such as embedded structures of image characteristics, to ensure the accuracy and completeness of statistical information.
4. ** video file processing **: correctly copy video files and maintain the correct indexing relationship between the video and other data, support multiple video storage structures and ensure synchronization of video data with other data.
5. ** JobMap **: Automatically detect and merge the same job descriptions, create a new job index map to facilitate unified task management and call.
6. ** data prevalidation **: Implement a full pre-validation of data sets prior to consolidation operations, check the number of video frames, the number of pictures and the length of frames recorded in metadata, ensure the accuracy and completeness of the consolidated data, and automatically repair the recoverable problem (recoding video from pictures).
7. ** image file management **: Copy and organize image files, maintain correct naming rules and directory structure, ensure that images are indexed correctly with video and other data, and support the need to enable image copying to optimize storage space.

## VII. NOTES
1. Ensures that all source data sets are structured in a compatible manner, which could result in merger failure or data error.
2. A consolidated data set may occupy larger disk space and ensure that sufficient storage space is available before consolidation.
3. For very large data sets, the consolidation process may take longer to wait.
4. Image folders occupy large disk space and default `copy_images` parameters. Under `images`, this utility supports **PNG files only**, named `frame_XXXXXX.png`, where XXXXXX is a six-digit index such as `frame_000001.png`. These PNG images are automatically detected and processed during the merger.

## VIII. Common issues
1. What happened to the ** Q: data gathering combining different dimensions?**
The ** A**: tool automatically detects maximum dimensions and fills with zero vectors of smaller dimensions to ensure that all data have consistent dimensions.
2. How does ** Q: handle different FPS data sets?**
** A**: temporarily supports only the consolidation of the same FPS data sets.
3. Could ** Q: only merge certain episodes?**
The current version of ** A**: consolidates all data. If you need more sophisticated control, you can screen the data sets and then merge them. You can also merge all of them, then load specific episode with lerobot.

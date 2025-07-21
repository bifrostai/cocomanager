# COCO Manager

**A powerful Python library for managing, editing, and converting object detection annotations across multiple formats (COCO, YOLO, Pascal VOC).**

## Overview

COCO Manager, developed by **Bifrost**, simplifies working with object detection datasets by providing a unified interface for annotation manipulation, visualization, and format conversion. Whether you're preparing datasets for training, cleaning annotations, or converting between different annotation formats, COCO Manager streamlines your workflow with an intuitive API.

> **⚠️ Note**: This library is designed primarily for **COCO format** annotations, ensuring robust and reliable functionality for COCO JSON files. While YOLO (.txt) and Pascal VOC (.xml) formats are supported for convenience, they are not the main focus and may contain limitations.


## Installation
1. Just dependencies
```
poetry install --without dev
```

2. Dev tools
```
poetry install
```


## Getting Started
To run the annotations manager, you will need:
- COCO annotation file _(.json)_
- Directory containing images of the COCO annotation file _(optional)_

COCO manager allows you to perform several manipulation on the object detection annotations but would require parsing into the `COCOParser` class first.
```python
import cocomanager as cm

coco_parser = cm.COCOParser("/path/to/coco_file", img_dir="path/to/images")

# voc and yolo are supported with limitations
voc_parser = cm.COCOParser.parse_from_voc("/path/to/annotations_files", img_dir="path/to/images")
yolo_parser = cm.COCOParser.parse_from_yolo("/path/to/annotations_files", img_dir="path/to/images")

coco_parser.remove_images(
    ["path/to/image1", "path/to/image2"],
    inplace=True
).remove_categories(
    ["cat_a", "cat_b"],
    inplace=True
).to_coco("path/to/coco_new.json")
```

## Functionalities

COCO Manager provides comprehensive functionality organized into the following categories:

### Standalone Functions
| Function | Description |
|----------|-------------|
| `concat` | Concatenate multiple COCO datasets with intelligent conflict resolution |
| `validate_coco` | Validate COCO annotation files for errors and inconsistencies |
| `annotate_all_images` | Batch annotate all images in a dataset for visualization |

### COCOParser Methods

#### 📊 Data Access & Information
| Method | Description |
|--------|-------------|
| `get_categories_mapping` | Get mapping of category IDs to category names |
| `get_images_mapping` | Get mapping of image IDs to image file names |
| `validate_bbox` | Validate bounding box coordinates and dimensions |

#### ✏️ Edit & Manipulate
| Method | Description |
|--------|-------------|
| `match_categories` | Match category mappings between different datasets |
| `rename_categories` | Rename category labels with custom mapping |
| `update_annotations` | Update specific annotation properties |
| `clip_bbox` | Clip bounding boxes to image boundaries |
| `remove_annotations` | Remove specific annotations by ID |
| `filter_annotations` | Keep only specified annotations |
| `remove_images` | Remove images and their associated annotations |
| `filter_images` | Keep only specified images |
| `remove_categories` | Remove categories and their annotations |
| `filter_categories` | Keep only specified categories |

#### 📈 Visualization & Plotting
| Method | Description |
|--------|-------------|
| `sample_images` | Plot random sample of images with annotations |
| `plot_images` | Plot specific images with their annotations |
| `sample_categories` | Plot random samples from specific categories |
| `plot_annotation` | Plot specific annotations by ID |

#### 💾 Export & Conversion
| Method | Description |
|--------|-------------|
| `to_coco` | Export annotations to COCO JSON format |
| `to_yolo` | Export annotations to YOLO txt format |
| `to_voc` | Export annotations to Pascal VOC XML format |

# Product Detection with SAM2

This repository contains a Python implementation for product detection and tracking using the SAM2 (Segment Anything Model 2) framework. The system evaluates product categories by tracking objects across images and performing COCO-style evaluation metrics.

## Features

- **Object Tracking**: Tracks products across image pairs using SAM2's video prediction capabilities
- **Bounding Box Extraction**: Converts segmentation masks to bounding boxes with dynamic padding
- **COCO Evaluation**: Implements standard COCO evaluation metrics (AP, AP50, AP75, AR)
- **Multi-Category Processing**: Handles multiple product categories with separate evaluations
- **Visualization**: Includes optional visualization of predictions and ground truth

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Pillow (PIL)
- pycocotools
- SAM2 model files (config and checkpoint)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sam2-product-tracking.git
   cd sam2-product-tracking
   ```

2. Install required packages:
   ```bash
   pip install torch numpy matplotlib pillow pycocotools
   ```

3. Download SAM2 model files and place them in the appropriate directory:
   - `sam2_hiera_tiny.pt` (checkpoint)
   - `sam2_hiera_t.yaml` (config)

## Usage

1. About the dataset:
   - The images and masks are present in a flat directory structure.
   - File naming convention: `{category}_{identifier}.jpg` for images and `{category}_{identifier}_gt.png` for masks.

2. Modifying the configuration paths in the script:
   - Update `checkpoint` and `model_cfg` paths to point to your SAM2 model files
   - Set `data_dir` to point to your dataset directory

3. Run the evaluation:
   ```bash
   python product_tracking.py
   ```


## Output

The script:
1. Processes each product category separately.
2. Generate evaluation metrics (AP, AP50, AP75, AR) for each category.
3. Save detailed results to `{category}_results.txt` files.
4. Print a summary of final results

## Example Results

```
=== Final Results ===

product_category1:
  AP: 0.782
  AP50: 0.891
  AP75: 0.843
  AR: 0.812

product_category2:
  AP: 0.723
  AP50: 0.845
  AP75: 0.781
  AR: 0.792
```



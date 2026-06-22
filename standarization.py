import os
import cv2
import numpy as np
from pathlib import Path

def standardize_xray(image_path: Path, target_size=(256, 256), zoom_factor=1.2) -> np.ndarray:
    """
    Executes offline standardization:
    1. Center-crop to a square.
    2. Scale by zoom_factor.
    3. Resize to target dimensions.
    4. Apply CLAHE.
    """
    # 1. Load image in grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # 2. Crop to center square
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    square_img = img[start_y:start_y+min_dim, start_x:start_x+min_dim]
    
    # 3. Apply Zoom Factor
    # We create a size that represents the zoomed-in view
    # e.g., if target is 256 and zoom is 1.2, we want a larger crop area
    zoom_dim = int(min_dim / zoom_factor)
    zoom_start = (min_dim - zoom_dim) // 2
    zoomed_img = square_img[zoom_start : zoom_start + zoom_dim, 
                            zoom_start : zoom_start + zoom_dim]
    
    # 4. Resize to target size (256x256)
    resized_img = cv2.resize(zoomed_img, target_size, interpolation=cv2.INTER_CUBIC)
    
    # 5. Illumination & Contrast: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    standardized_img = clahe.apply(resized_img)
        
    return standardized_img

def process_dataset(raw_dir: str, output_dir: str, target_size=(256, 256), zoom_factor=1.2):
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG']
    image_paths = []
    for ext in extensions:
        image_paths.extend(raw_path.rglob(ext))
        
    print(f"Found {len(image_paths)} images for processing. Zoom factor: {zoom_factor}x.")
    
    for img_path in image_paths:
        relative_path = img_path.relative_to(raw_path)
        dest_path = output_path / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Apply standardizer
        processed_img = standardize_xray(img_path, target_size=target_size, zoom_factor=zoom_factor)
        
        if processed_img is not None:
            cv2.imwrite(str(dest_path), processed_img)

if __name__ == "__main__":    
    # NORMAL
    process_dataset("./assets/train/NORMAL", "./assets/post_norm", target_size=(384, 384), zoom_factor=1)
    # PNEUMONIA
    process_dataset("./assets/train/PNEUMONIA", "./assets/post_pneu", target_size=(384, 384), zoom_factor=1)
    print("Offline standardization complete.")
    
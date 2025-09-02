import cv2
import os
import numpy as np

def zoomin(input_folder, output_folder, scale=4):
    os.makedirs(output_folder, exist_ok=True)
    supported_exts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    
    for filename in os.listdir(input_folder):
        name, ext = os.path.splitext(filename)
        if ext.lower() in supported_exts:
            input_path = os.path.join(input_folder, filename)
            
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Failed to read image file: {filename}")
                continue
            
            resized_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_img)
    
            # statistic
            unique_values = np.unique(resized_img)
            print(f"The number of gray value types in {filename}: {len(unique_values)}, gray values: {unique_values}")

    print("Done.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Zoomin mask')
    parser.add_argument('--input-dir', 
                        type=str, 
                        default=r'..\..\..\datasets\AWSD2025\training\annotations')
    parser.add_argument('--output-dir', 
                        type=str,
                        default=r'..\..\..\datasets\AWSD2025\training\annotations1')
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()
    
    zoomin(args.input_dir, args.output_dir, args.scale)
    
    
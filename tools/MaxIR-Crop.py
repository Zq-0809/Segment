# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 22:29:17 2025

@author: zhouy
"""
import numpy as np
import rasterio
from rasterio.windows import Window

def find_largest_valid_rectangle_large(data, 
                                       nodata=None, 
                                       block_size=1000):
    if nodata is None:
        mask = (data != 0) & (~np.isnan(data))
    else:
        mask = (data != nodata) & (~np.isnan(data))
    
    rows, cols = mask.shape
    max_area = 0
    best_rect = (0, 0, 0, 0)
    
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            i_end = min(i + block_size, rows)
            j_end = min(j + block_size, cols)
            block_mask = mask[i:i_end, j:j_end]
            if np.any(block_mask):
                integral = np.zeros((block_mask.shape[0] + 1, block_mask.shape[1] + 1), dtype=int)
                integral[1:, 1:] = np.cumsum(np.cumsum(block_mask, axis=0), axis=1)
                
                for k in range(block_mask.shape[0]):
                    for l in range(block_mask.shape[1]):
                        if not block_mask[k, l]:
                            continue
                            
                        for m in range(k, block_mask.shape[0]):
                            for n in range(l, block_mask.shape[1]):
                                area = (m - k + 1) * (n - l + 1)
                                rect_sum = integral[m+1, n+1] - integral[k, n+1] - integral[m+1, l] + integral[k, l]
                                
                                if rect_sum == area and area > max_area:
                                    max_area = area
                                    best_rect = (i + k, j + l, i + m, j + n)
    
    return best_rect

def find_largest_valid_rectangle_optimized(data, 
                                           nodata=None):
    if nodata is None:
        mask = (data != 0) & (~np.isnan(data))
    else:
        mask = (data != nodata) & (~np.isnan(data))
    
    rows, cols = mask.shape
    max_area = 0
    best_rect = (0, 0, 0, 0)
    
    hist = np.zeros(cols, dtype=int)
    
    for i in range(rows):
        hist = np.where(mask[i, :], hist + 1, 0)
        stack = []
        for j in range(cols + 1):
            while stack and (j == cols or hist[j] < hist[stack[-1]]):
                h = hist[stack.pop()]
                w = j - stack[-1] - 1 if stack else j
                
                if h * w > max_area:
                    max_area = h * w
                    top = i - h + 1
                    left = stack[-1] + 1 if stack else 0
                    best_rect = (top, left, i, left + w - 1)
            
            stack.append(j)
    
    return best_rect

def crop_to_valid_region_advanced(input_path, 
                                  output_path, 
                                  bands='all', 
                                  output_format='GTiff', 
                                  compress='LZW'):
    with rasterio.open(input_path) as src:
        if bands == 'all':
            band_list = list(range(1, src.count + 1))
        else:
            band_list = bands
            
        band1 = src.read(1)
        nodata = src.nodata
        min_row, min_col, max_row, max_col = \
            find_largest_valid_rectangle_optimized(band1, nodata)
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        window = Window(min_col, min_row, width, height)
        
        data = src.read(band_list, window=window)
        transform = src.window_transform(window)
        
        profile = src.profile
        profile.update({
            'height': height,
            'width': width,
            'transform': transform,
            'count': len(band_list),
            'nodata': nodata,
            'driver': output_format,
            'compress': compress
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
        
        return output_path
    
def verify_georeference(original_path, cropped_path):
    with rasterio.open(original_path) as src_orig:
        with rasterio.open(cropped_path) as src_crop:
            orig_center = src_orig.transform * (src_orig.width / 2, src_orig.height / 2)
            crop_center = src_crop.transform * (src_crop.width / 2, src_crop.height / 2)
            dx = abs(orig_center[0] - crop_center[0])
            dy = abs(orig_center[1] - crop_center[1])
            
            transform_match = src_orig.transform == src_crop.transform
            
            return dx < 1e-6 and dy < 1e-6 and transform_match
        
def batch_process_directory(input_dir, 
                            output_dir, 
                            pattern="*.tif"):
    from glob import glob
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    input_files = glob(os.path.join(input_dir, pattern))
    
    for input_path in input_files:
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, f'crop_{filename}')
        
        print(f"{filename}")
        try:
            crop_to_valid_region_advanced(input_path, output_path)
            is_valid = verify_georeference(input_path, output_path)
            print(f"CRS valid: {'pass' if is_valid else 'failed'}")
        except Exception as e:
            print(f"Process {filename} error: {str(e)}")

    return

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description='GeoTiff crop')
    parser.add_argument('--input-dir', 
                        type=str, 
                        default=r'..\..\..\datasets\AWSD2025\training\images')
    parser.add_argument('--output-dir', 
                        type=str,
                        default=r'..\..\..\datasets\AWSD2025\training\images1')
    args = parser.parse_args()
    
    input_tif = args.input_dir
    output_tif = args.output_dir
    
    if os.path.isfile(input_tif):
        crop_to_valid_region_advanced(
            input_tif, 
            output_tif,
            bands='all',
            compress='LZW'
        )
    elif os.path.isdir(input_tif):
        batch_process_directory(
            input_dir=input_tif,
            output_dir=output_tif,
            pattern="*.tif"
        )
        
    print('Done.')
    
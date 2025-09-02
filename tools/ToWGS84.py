# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 17:08:12 2025

@author: zhouy
"""

import rasterio
from rasterio.warp import calculate_default_transform, \
    reproject, Resampling
import numpy as np
from glob import glob
import os
    
def reproject_geotiff_to_4326(input_path, output_path):
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            'EPSG:4326',
            src.width, 
            src.height,
            *src.bounds
        )
        
        dst_meta = src.meta.copy()
        dst_meta.update({
            'crs': 'EPSG:4326',
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': src.nodata if src.nodata is not None else 0
        })
        
        with rasterio.open(output_path, 'w', **dst_meta) as dst:
            for band in range(1, src.count + 1):
                src_data = src.read(band)
                dst_data = np.zeros((height, width), dtype=src_data.dtype)
                reproject(
                    source=src_data,
                    destination=dst_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs='EPSG:4326',
                    resampling=Resampling.nearest
                )
                
                dst.write(dst_data, band)
                
    return

def batch_process_directory(input_dir, 
                            output_dir, 
                            pattern="*.tif"):
    os.makedirs(output_dir, exist_ok=True)
    input_files = glob(os.path.join(input_dir, pattern))
    
    for input_path in input_files:
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, f'WGS84_{filename}')
        reproject_geotiff_to_4326(input_path, output_path)
        print(f"{filename}")
        
    return

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert to WGS84')
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
        reproject_geotiff_to_4326(input_tif, output_tif)
    elif os.path.isdir(input_tif):
        batch_process_directory(input_tif, output_tif)
    
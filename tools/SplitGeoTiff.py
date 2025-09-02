# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 21:14:45 2025

@author: zhouy
"""

import os
import rasterio
from rasterio.windows import Window
from glob import glob

def split_geotiff(input_path, 
                  output_dir, 
                  tile_width, 
                  tile_height):
    
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        crs = src.crs
        
        width = src.width
        height = src.height
        n_cols = (width + tile_width - 1) // tile_width
        n_rows = (height + tile_height - 1) // tile_height
        
        for row in range(n_rows):
            for col in range(n_cols):
                x_offset = col * tile_width
                y_offset = row * tile_height
                win_width = min(tile_width, width - x_offset)
                win_height = min(tile_height, height - y_offset)
                
                window = Window(x_offset, y_offset, win_width, win_height)
                
                data = src.read(window=window)
                win_transform = src.window_transform(window)
                
                tile_meta = meta.copy()
                tile_meta.update({
                    'width': win_width,
                    'height': win_height,
                    'transform': win_transform
                })
                
                output_path = os.path.join(
                    output_dir,
                    f"{os.path.splitext(os.path.basename(input_path))[0]}_row{row}_col{col}.tif"
                )
                
                with rasterio.open(output_path, 'w', **tile_meta) as dst:
                    dst.write(data)
                    dst.crs = crs
                    
                print(f"{output_path} ({win_width}x{win_height})")
    return

def batch_process_directory(input_dir, 
                            output_dir, 
                            tile_width,
                            tile_height,
                            pattern="*.tif"):
    os.makedirs(output_dir, exist_ok=True)
    input_files = glob(os.path.join(input_dir, pattern))
    
    for input_path in input_files:
        split_geotiff(input_path, output_dir, tile_width, tile_height)
        print(input_path)
        
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
    parser.add_argument('--tile-width', type=int, default=1920)
    parser.add_argument('--tile-height', type=int, default=1920)
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if os.path.isfile(input_dir):
        split_geotiff(
            input_path=input_dir,
            output_dir=output_dir,
            tile_width=args.tile_width,
            tile_height=args.tile_height
        )
    elif os.path.isdir(input_dir):
        batch_process_directory(input_dir, output_dir,
            tile_width=args.tile_width, tile_height=args.tile_height)
        

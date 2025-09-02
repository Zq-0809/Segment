# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 21:43:18 2025

@author: zhouy
"""

import os
from glob import glob
from PIL import Image

def split_png(input_path, 
              output_dir, 
              tile_width, 
              tile_height):
    
    os.makedirs(output_dir, exist_ok=True)
    
    with Image.open(input_path) as src:
        width, height = src.size
        n_cols = (width + tile_width - 1) // tile_width
        n_rows = (height + tile_height - 1) // tile_height
        
        filename_prefix = os.path.splitext(os.path.basename(input_path))[0]
        
        for row in range(n_rows):
            for col in range(n_cols):
                left = col * tile_width
                upper = row * tile_height
                right = min(left + tile_width, width)
                lower = min(upper + tile_height, height)
                
                tile = src.crop((left, upper, right, lower))
                tile_width_actual = right - left
                tile_height_actual = lower - upper

                output_path = os.path.join(
                    output_dir,
                    f"{filename_prefix}_row{row}_col{col}.png"
                )

                tile.save(output_path)
                print(f"Saved: {output_path} ({tile_width_actual}x{tile_height_actual})")
    return

def batch_process_png_directory(input_dir, 
                                output_dir, 
                                tile_width,
                                tile_height,
                                pattern="*.png"):
    os.makedirs(output_dir, exist_ok=True)
    input_files = glob(os.path.join(input_dir, pattern))
    
    for input_path in input_files:
        print(f"\nProcessing: {os.path.basename(input_path)}")
        split_png(input_path, output_dir, tile_width, tile_height)
    return

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split PNG images into tiles')
    parser.add_argument('--input-dir', type=str, required=False,
                        default=r'..\..\..\datasets\AWSD2025\training\annotations')
    parser.add_argument('--output-dir', type=str, required=False,
                        default=r'..\..\..\datasets\AWSD2025\training\annotations')
    parser.add_argument('--tile-width', type=int, default=1920)
    parser.add_argument('--tile-height', type=int, default=1920)
    parser.add_argument('--pattern', type=str, default="*.png")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input_dir):
        split_png(
            input_path=args.input_dir,
            output_dir=args.output_dir,
            tile_width=args.tile_width,
            tile_height=args.tile_height
        )
    elif os.path.isdir(args.input_dir):
        batch_process_png_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            pattern=args.pattern
        )
        
    print('Done.')
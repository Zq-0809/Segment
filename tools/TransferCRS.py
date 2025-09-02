# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 21:55:24 2025

@author: zhouy
"""

from osgeo import gdal
import os

def transfer_georeference_with_gdal(src_path, 
                                    dst_path, 
                                    scale):
    src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise ValueError(f"Failed to open: {src_path}")
        
    src_geotrans = src_ds.GetGeoTransform()
    src_proj = src_ds.GetProjection()
    src_band = src_ds.GetRasterBand(1)
    nodata_value = src_band.GetNoDataValue()
    
    new_geotrans = (
        src_geotrans[0],           # x
        src_geotrans[1] / scale,   # horiz scale
        src_geotrans[2],           # rotate x
        src_geotrans[3],           # y
        src_geotrans[4],           # rotate y
        src_geotrans[5] / scale    # vert scale
    )
    
    dst_ds = gdal.Open(dst_path, gdal.GA_Update)
    if dst_ds is None:
        raise ValueError(f"Failed to open: {dst_path}")
    
    # Setting crs
    dst_ds.SetGeoTransform(new_geotrans)
    dst_ds.SetProjection(src_proj)
    
    # Set nodata
    if nodata_value is not None:
        for i in range(1, dst_ds.RasterCount + 1):
            dst_ds.GetRasterBand(i).SetNoDataValue(nodata_value)
    
    # write metadata
    dst_ds.FlushCache()
    
    # close
    src_ds = None
    dst_ds = None
    
    return

def verify_georeference(src_path, 
                        dst_path, 
                        scale):
    
    src_ds = gdal.Open(src_path)
    dst_ds = gdal.Open(dst_path)
    
    # validate crs
    src_proj = src_ds.GetProjection()
    dst_proj = dst_ds.GetProjection()
    print(f"Same CRS: {src_proj == dst_proj}")
    
    # validate geotransform
    src_gt = src_ds.GetGeoTransform()
    dst_gt = dst_ds.GetGeoTransform()
    
    expected_xres = src_gt[1] / scale
    expected_yres = src_gt[5] / scale
    
    ul_match = (abs(src_gt[0] - dst_gt[0]) < 1e-6 and 
                abs(src_gt[3] - dst_gt[3]) < 1e-6)
    print(f"Top-left equal: {ul_match}")
    
    # resolutions
    res_match = (abs(dst_gt[1] - expected_xres) < 1e-6 and 
                 abs(dst_gt[5] - expected_yres) < 1e-6)
    print(f"Same resolution: {res_match} (expected: {expected_xres}, {expected_yres}; real: {dst_gt[1]}, {dst_gt[5]})")
    
    # right-bottom
    src_lr_x = src_gt[0] + src_gt[1] * src_ds.RasterXSize
    src_lr_y = src_gt[3] + src_gt[5] * src_ds.RasterYSize
    
    dst_lr_x = dst_gt[0] + dst_gt[1] * dst_ds.RasterXSize
    dst_lr_y = dst_gt[3] + dst_gt[5] * dst_ds.RasterYSize
    
    lr_match = (abs(src_lr_x - dst_lr_x) < 1e-6 and 
                abs(src_lr_y - dst_lr_y) < 1e-6)
    print(f"Bottom-right equal: {lr_match}")

def transfer(src_path, super_path, output_path, scale=4):
    if os.path.exists(output_path):
        os.remove(output_path)
    gdal.Translate(output_path, super_path)
    transfer_georeference_with_gdal(src_path, output_path, scale)
    verify_georeference(src_path, output_path, scale)
    
    return

def batch_process_directory(geotiff_dir, 
                            super_dir,
                            output_dir,
                            scale=4):
    from glob import glob
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    geotiff_files = sorted(glob(os.path.join(geotiff_dir, '*.tif')))
    super_files = sorted(glob(os.path.join(super_dir, '*.tif')))
    
    for i, tiff in enumerate(geotiff_files):
        super_file = super_files[i]
        filename = os.path.basename(super_file)
        output_path = os.path.join(output_dir, f'crs_{filename}')
        
        print(f"{filename}")
        try:
            transfer(tiff, super_file, output_path, scale)
        except Exception as e:
            print(f"Process {filename} error: {str(e)}")
            
    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Transfer CRS')
    parser.add_argument('--geotiff-dir', 
                        type=str, 
                        default=r'..\..\..\datasets\AWSD2025\training\images')
    parser.add_argument('--super-dir', 
                        type=str, 
                        default=r'..\..\..\datasets\AWSD2025\training\supers')
    parser.add_argument('--output-dir', 
                        type=str,
                        default=r'..\..\..\datasets\AWSD2025\training\output')
    parser.add_argument('--scale', type=int, default=4)
    
    args = parser.parse_args()
    
    geotiff_dir = args.geotiff_dir
    super_dir = args.super_dir
    output_dir = args.output_dir

    if os.path.isfile(geotiff_dir) and os.path.isfile(super_dir):
        transfer(geotiff_dir, super_dir, output_dir, scale=args.scale)
    elif os.path.isdir(geotiff_dir) and os.path.isdir(super_dir):
        batch_process_directory(geotiff_dir, 
                                super_dir,
                                output_dir,
                                scale=args.scale)
    
    print("Done!")
    
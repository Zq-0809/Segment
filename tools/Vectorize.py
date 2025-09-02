# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 20:06:59 2025

@author: zhouy
"""
import numpy as np
from osgeo import gdal, ogr, osr
import os
import cv2

def apply_georeference_to_raster(segmentation_array, 
                                 reference_raster, 
                                 output_raster):
    """
    Transfer geospatial reference and affine transformation from the 
    reference image to the segmentation outputs
    
    Args:
    segmentation_raster
    reference_raster
    output_raster
    """
    
    # Access georeferencing information from the reference image
    ref_ds = gdal.Open(reference_raster, gdal.GA_ReadOnly)
    if ref_ds is None:
        raise RuntimeError(f"Failed to open reference raster: {reference_raster}")
    srs = ref_ds.GetSpatialRef()
    if srs is None:
        print(f'{reference_raster} has no Spatial Refference.')
        return None
    
    geotransform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    x_size = ref_ds.RasterXSize
    y_size = ref_ds.RasterYSize
    
    # Open segmentation
    if (segmentation_array.shape[0] != y_size) or (segmentation_array.shape[1] != x_size):
        ref_ds = None
        raise RuntimeError(f"Array shape {segmentation_array.shape} does not match reference raster dimensions ({x_size}, {y_size}).")
        
    # Create output raster
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_raster, 
        x_size, 
        y_size, 
        1,
        gdal.GDT_Byte
    )
    
    if out_ds is None:
        ref_ds = None
        raise RuntimeError(f"Can not create output raster: {output_raster}")
    
    # Set CRS
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    
    # Copy data
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(segmentation_array)
    out_band.FlushCache()
    
    # Free
    ref_ds = None
    out_ds = None
    
    return output_raster

def vectorize_segmentation(georef_raster, 
                           output_vector, 
                           min_area=0, 
                           simplify_tolerance=0,
                           class_mapping=None):
    """
    Vectorization of Georeferenced Segmentation Results
    
    Args:
    georef_raster
    output_vector
    min_area
    class_mapping
    """
    
    #gdal.UseExceptions()
    #ogr.UseExceptions()
    
    raster_ds = gdal.Open(georef_raster, gdal.GA_ReadOnly)
    if raster_ds is None:
        raise RuntimeError(f"Failed to open raster file: {georef_raster}")
    
    raster_band = raster_ds.GetRasterBand(1)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_ds.GetProjectionRef())
    
    for ck in class_mapping:
        if ck == 0:
            continue
        cname = class_mapping[ck]
        
        # bg mask
        mdata = raster_band.ReadAsArray()
        mask = (mdata == ck).astype(np.uint8)
        
        driverm = gdal.GetDriverByName("MEM")
        mask_ds = driverm.Create("", raster_ds.RasterXSize, raster_ds.RasterYSize, 1, gdal.GDT_Byte)
        mask_ds.GetRasterBand(1).WriteArray(mask)
        
        # Create output vector datasource
        driver_name = "ESRI Shapefile"  # maybe "ESRI Shapefile", "GeoJSON" et.al.
        driver = ogr.GetDriverByName(driver_name)
        
        path, fname = os.path.split(output_vector)
        output_cur = os.path.join(path, f'{cname}{fname}')
        if os.path.exists(output_cur):
            driver.DeleteDataSource(output_cur)
        
        vector_ds = driver.CreateDataSource(output_cur)
        if vector_ds is None:
            raise RuntimeError(f"Failed to create vector file: {output_cur}")
        
        # Create layer
        layer_name = os.path.splitext(os.path.basename(output_cur))[0]
        layer = vector_ds.CreateLayer(layer_name, raster_srs, ogr.wkbPolygon)
        
        # Add field
        layer.CreateField(ogr.FieldDefn("class_id", ogr.OFTInteger))
        
        if class_mapping:
            class_name_field = ogr.FieldDefn("class_name", ogr.OFTString)
            class_name_field.SetWidth(50)
            layer.CreateField(class_name_field)
        
        # Create temp memory datasource
        mem_driver = ogr.GetDriverByName('Memory')
        mem_ds = mem_driver.CreateDataSource('temp')
        mem_layer = mem_ds.CreateLayer('temp', raster_srs, ogr.wkbPolygon)
        mem_layer.CreateField(ogr.FieldDefn("class_id", ogr.OFTInteger))
        
        # Start polygonize
        gdal.Polygonize(raster_band, mask_ds.GetRasterBand(1), 
                        mem_layer, 0, ["maskValue=0"])
        
        # Process features
        for feature in mem_layer:
            class_id = feature.GetField("class_id")
            geom = feature.GetGeometryRef()
            
            # skip invalid geom
            if geom is None or geom.IsEmpty():
                continue
                
            # fix invalid geom
            if not geom.IsValid():
                geom = geom.MakeValid()
                if geom is None:
                    continue
            
            # filter by area
            area = geom.GetArea()
            if min_area > 0 and area < min_area:
                continue
            
            # simplify geom
            if simplify_tolerance > 0:
                geom = geom.SimplifyPreserveTopology(simplify_tolerance)
            
            # Create feature
            out_feature = ogr.Feature(layer.GetLayerDefn())
            out_feature.SetGeometry(geom)
            out_feature.SetField("class_id", class_id)
            
            if class_mapping:
                class_name = class_mapping.get(class_id, f"unknown_{class_id}")
                out_feature.SetField("class_name", class_name)
            
            layer.CreateFeature(out_feature)
            out_feature = None
        
        # free resouce
        mem_ds = None
        vector_ds = None
        mask_ds = None
        
    raster_ds = None
    
    return output_vector

def morphology(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return closed

def process_segmentation_without_crs(segmentation_raster, 
                                     reference_raster, 
                                     output_vector,
                                     min_area=0, 
                                     class_mapping=None):
    """
    Handling Segmentation Results Without Coordinates: Assigning a Reference Frame and Vectorizing
    
    Args:
    segmentation_raster -- File Path for Segmentation Results Without Coordinates
    reference_raster -- Georeferenced Reference Image Path
    output_vector
    min_area
    class_mapping
    """
    
    # Create Temporary Coordinate-System-Assigned Segmentation Results
    segment_mask = cv2.imread(segmentation_raster, cv2.IMREAD_UNCHANGED)
    
    path, name = os.path.split(segmentation_raster)
    fn, _ = os.path.splitext(name)
    
    for cid in class_mapping:
        if cid == 0:
            continue
        cname = class_mapping[cid]
        
        temp_raster = os.path.join(path, f'{cname}_{fn}.tif')
        mask = (segment_mask == cid).astype(np.uint8)
        mask = morphology(mask)
        mask = (mask != 0).astype(np.uint8)
        mask = (mask * cid).astype(np.uint8)
        
        georef_raster = apply_georeference_to_raster(
            mask, 
            reference_raster, 
            temp_raster
        )
        if georef_raster is None:
            continue
        
        mapping = {cid: cname}
        # Convert Georeferenced Segmentation Outputs to Vector Format
        vectorize_segmentation(
            georef_raster, 
            output_vector, 
            min_area, 
            simplify_tolerance=0,
            class_mapping=mapping
        )
        
        # Remove temp files
        if os.path.exists(temp_raster):
            os.remove(temp_raster)
        
    return

def batch_process_directory(geotiff_dir, 
                            segmentation_dir,
                            output_dir,
                            min_area=10,
                            class_mapping=None):
    from glob import glob
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    geotiff_files = sorted(glob(os.path.join(geotiff_dir, '*.tif')))
    segment_files = sorted(glob(os.path.join(segmentation_dir, '*.png')))
    
    for i, tiff in enumerate(geotiff_files):
        seg = segment_files[i]
        filename = os.path.basename(seg)
        fn, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f'vec{fn}.shp')
        
        print(f"{filename}")
        try:
            process_segmentation_without_crs(
                segmentation_raster=seg,
                reference_raster=tiff,
                output_vector=output_path,
                min_area=10,
                class_mapping=class_mapping
            )
        except Exception as e:
            print(f"Process {filename} error: {str(e)}")

    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Vectorize segmentation')
    parser.add_argument('--geotiff-dir', 
                        type=str, 
                        default=r'..\..\..\datasets\AWSD2025\training\images')
    parser.add_argument('--seg-dir', 
                        type=str, 
                        default=r'..\..\..\datasets\AWSD2025\training\annotations')
    parser.add_argument('--output-dir', 
                        type=str,
                        default=r'..\..\..\datasets\AWSD2025\training\shapes')
    parser.add_argument('--min-area', type=float, default=0)
    
    args = parser.parse_args()
    
    geotiff_dir = args.geotiff_dir
    seg_dir = args.seg_dir
    output_dir = args.output_dir
    
    # Class mapping
    class_mapping = {
        0: "background",
        1: "Farmland",
        2: "Forest"
    }
    
    if os.path.isfile(geotiff_dir) and os.path.isfile(seg_dir):
        process_segmentation_without_crs(
            segmentation_raster=seg_dir,
            reference_raster=geotiff_dir,
            output_vector=output_dir,
            min_area=args.min_area,
            class_mapping=class_mapping
        )
    elif os.path.isdir(geotiff_dir) and os.path.isdir(seg_dir):
        batch_process_directory(
            geotiff_dir=geotiff_dir,
            segmentation_dir=seg_dir,
            output_dir=output_dir,
            min_area=args.min_area,
            class_mapping=class_mapping
            )
    
    print("Done!")
    
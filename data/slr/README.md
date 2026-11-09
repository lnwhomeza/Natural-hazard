# DeltaDTM: A global coastal digital terrain model

_this is v1.1 of DeltaDTM, see CHANGELOG.md for details_

## Introduction
This dataset contains 7,599 tiles of 1 by 1 degree in the Cloud Optimized Geotiff (COG) format (.tif), using ZSTD compression. The names of the tiles are constructed by using the name DeltaDTM, a version x_y and the tile location NYYEXXX, split by underscores _. An example filename is thus DeltaDTM_v1_1_N00E099.tif.

The coordinate reference system (CRS) of the geotiffs is set to EPSG:9518, a compound CRS combining the horizontal CRS WGS84 (EPSG:4326) reference and the vertical CRS EGM2008 (EPSG:3855). All geotiffs also contain metadata describing the dataset, usage notes and a reference to the doi of the dataset.

For each elevation tile, a mask tile is also provided in the mask_tiles.zip. These are the original CopernicusDEM mask tiles, with the addition of value 255 to indicate where DeltaDTM is *clipped* at 30 m +EGM2008.

A Virtual Raster Table (DeltaDTM_v1_1.vrt) file linking all tiles is also provided and is advised to be used for overviews or visualisation. Note that the Virtual Raster Table file cannot represent all the different resolutions of the individual tiles.

Finally, a geopackage (deltadtm_tiles.gpkg) is provided with the bounding boxes of each tile, including an attribute detailing in which zipfile (continent) it can be found.

A Google Earth Engine (GEE) asset is provided at https://code.earthengine.google.com/?asset=users/maartenpronk/deltadtm/v1-1.

For questions or to provide feedback, please contact Maarten Pronk (maarten.pronk@deltares.nl).

## Methods
DeltaDTM corrects the CopernicusDEM with space borne lidar from the ICESat-2 and GEDI missions. Specifically, we correct the elevation bias in CopernicusDEM, apply filters to remove non-terrain cells, and fill the gaps using interpolation.

For more details, please see the linked paper at https://www.nature.com/articles/s41597-024-03091-9. The software used to create DeltaDTM is provided as DeltaDTM.jl at https://zenodo.org/doi/10.5281/zenodo.10051451.

## Data
DeltaDTM is licensed as CC-BY 4.0. DeltaDTM is produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

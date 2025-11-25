# Data Sources - Where to Get LiDAR, BIM, and Basemap Data

This guide provides comprehensive information on obtaining real-world data for developing and testing the Infrastructure Intelligence Platform.

---

## Table of Contents

1. [LiDAR Point Cloud Data](#lidar-point-cloud-data)
2. [BIM/IFC Model Data](#bimifc-model-data)
3. [Basemap Tiles for MapLibre](#basemap-tiles-for-maplibre)
4. [GPR Data (Existing)](#gpr-data-existing)
5. [Sample HS2-Style Infrastructure Data](#sample-hs2-style-infrastructure-data)

---

## LiDAR Point Cloud Data

### UK Open Data Sources (Free & High Quality)

#### 1. Environment Agency LiDAR Data (England)
**Best Option for UK Development**

- **URL**: https://environment.data.gov.uk/DefraDataDownload/
- **Coverage**: England
- **Resolutions**: 25cm, 50cm, 1m, 2m
- **Formats**: LAS, LAZ (compressed), ASC (ASCII grid)
- **File Sizes**: Typically 50MB-500MB per tile
- **License**: Open Government Licence v3.0 (commercial use allowed)

**How to Download**:
1. Visit https://environment.data.gov.uk/DefraDataDownload/
2. Click "Continue to site"
3. Search by location or use the map
4. Select "LIDAR Composite DSM/DTM"
5. Choose resolution (25cm recommended for infrastructure)
6. Select file format (LAZ for smaller downloads)
7. Download tiles covering your area of interest

**Recommended Areas for Testing**:
- London (dense urban infrastructure)
- Birmingham HS2 corridor
- Manchester city center
- Bristol waterfront

#### 2. Scottish Public Sector LiDAR (Scotland)
- **URL**: https://remotesensingdata.gov.scot/
- **Coverage**: Scotland
- **Resolutions**: 1m, 2m
- **Formats**: LAS, DSM/DTM rasters
- **License**: Open Government Licence

#### 3. Natural Resources Wales LiDAR (Wales)
- **URL**: https://datamap.gov.wales/
- **Coverage**: Wales
- **Formats**: LAS, LAZ
- **Search**: Use "LIDAR" in the search bar

---

### Global Open Data Sources

#### 4. OpenTopography (Worldwide)
- **URL**: https://opentopography.org/
- **Coverage**: Global (varies by region)
- **Formats**: LAS, LAZ, point cloud
- **Notable Datasets**:
  - UK urban areas
  - US infrastructure corridors
  - European cities
- **Account**: Free registration required
- **Use Cases**: International projects, research

#### 5. USGS 3DEP (United States)
- **URL**: https://www.usgs.gov/3d-elevation-program
- **Coverage**: Entire United States
- **Resolutions**: Varies (0.5m-2m typical)
- **Formats**: LAS, LAZ
- **Quality**: Very high quality, regularly updated

---

### Commercial/Request Access Sources

#### 6. HS2 Sample Data
**For Real Infrastructure Projects**

- **Contact**: HS2 Ltd Data Management Team
- **URL**: https://www.hs2.org.uk/
- **What to Request**:
  - Sample LiDAR scans along Phase 1 corridor
  - BIM models (if available for testing)
  - Construction progress data
- **Likely Requirements**:
  - NDA (Non-Disclosure Agreement)
  - Academic/commercial use declaration
  - Data handling plan (GDPR compliant)

**Email Template**:
```
Subject: Request for Sample LiDAR and BIM Data for Infrastructure Platform Development

Dear HS2 Data Management Team,

I am developing an Infrastructure Intelligence Platform for construction
progress monitoring and BIM validation. I would like to request access to
sample LiDAR and BIM data for development and testing purposes.

Specifically, I am interested in:
- LiDAR point cloud data (sample corridor section)
- BIM models (IFC format if available)
- Construction progress documentation

I am willing to sign an NDA and provide a data handling plan compliant
with GDPR and data protection requirements.

Best regards,
[Your Name]
[Organization]
[Contact Details]
```

#### 7. Crossrail/Elizabeth Line Data
- **Contact**: Transport for London (TfL)
- **Potential Data**: Construction LiDAR, BIM models
- **Status**: Request via TfL Open Data portal or direct contact

---

### Sample File Sizes & Processing Times

| Resolution | File Size (1km²) | Point Count | Processing Time (Local) |
|------------|------------------|-------------|-------------------------|
| 25cm       | ~400MB (LAZ)     | ~16M points | ~5 minutes              |
| 50cm       | ~100MB (LAZ)     | ~4M points  | ~2 minutes              |
| 1m         | ~25MB (LAZ)      | ~1M points  | ~30 seconds             |
| 2m         | ~6MB (LAZ)       | ~250K points| ~10 seconds             |

---

## BIM/IFC Model Data

### Open-Source Sample IFC Files

#### 1. buildingSMART Sample Files
**Best Starting Point**

- **URL**: https://github.com/buildingSMART/Sample-Test-Files
- **Formats**: IFC2x3, IFC4, IFC4.3
- **File Types**:
  - Architectural models
  - Structural models
  - MEP (Mechanical, Electrical, Plumbing)
  - Infrastructure (roads, bridges)
- **License**: Various (mostly CC-BY or public domain)

**Recommended Files for Testing**:
```
/Sample-Test-Files/
├── IFC 2x3/
│   ├── Duplex_A_20110907.ifc          # Residential building
│   ├── rac_advanced_sample_project.ifc # Complex architecture
│   └── 201030_AC_model.ifc             # ArchiCAD model
└── IFC4/
    ├── FZK-Haus-2012.ifc               # Test house (widely used)
    ├── Office_Building.ifc              # Office model
    └── Bridge_Example.ifc               # Infrastructure example
```

#### 2. OSArch Sample IFC Files
- **URL**: https://wiki.osarch.org/index.php?title=Sample_IFC_files
- **Description**: Curated list of sample IFC files
- **Use Cases**: Testing IFC.js viewer, validation algorithms

#### 3. IfcOpenShell Test Files
- **URL**: https://github.com/IfcOpenShell/IfcOpenShell/tree/v0.7.0/test/input
- **Purpose**: Unit testing, edge cases
- **Formats**: IFC2x3, IFC4

---

### Educational & Research Sources

#### 4. UK BIM Alliance Sample Models
- **URL**: https://ukbimalliance.org/
- **Content**: UK construction industry standards
- **Access**: May require membership or request

#### 5. BIM Forum Sample Models
- **URL**: https://bimforum.org/
- **LOD Examples**: Level of Development 100-500 samples
- **Use Cases**: Understanding BIM maturity levels

---

### Commercial Platforms (Request Access)

#### 6. Autodesk Construction Cloud Samples
- **URL**: https://construction.autodesk.com/
- **What's Available**:
  - Sample Revit models (can export to IFC)
  - Construction project templates
- **Requirements**: Free Autodesk account

#### 7. Asite Sample Projects
- **URL**: https://www.asite.com/
- **Content**: Construction document management samples
- **Access**: Contact for demo data

---

### Infrastructure-Specific BIM Models

#### 8. Road & Bridge Models
**Sources**:
- **Open Infrastructure Initiative**: https://www.openbim.org/
- **Search**: "IFC bridge models" on GitHub
- **Example Projects**:
  - Highway design models
  - Railway infrastructure
  - Tunnel BIM models

---

## Basemap Tiles for MapLibre

### Option 1: Pre-Made MBTiles (Offline, Best for Local Dev)

#### OpenMapTiles
**Recommended for Production Use**

- **URL**: https://openmaptiles.org/
- **What**: Pre-generated vector tiles for offline use
- **Coverage**: Worldwide, by region or country
- **Formats**: MBTiles (SQLite database of vector tiles)
- **Styles Available**:
  - OSM Bright (colorful, detailed)
  - Positron (light, minimal)
  - Dark Matter (dark theme)
  - Klokantech Basic (simple)

**How to Get**:
1. Visit https://openmaptiles.org/downloads/
2. Select region (e.g., "Great Britain")
3. Choose data type: "Vector Tiles" (MBTiles)
4. Download (file size: ~2GB for UK)
5. Serve locally with TileServer-GL

**Serving Locally**:
```bash
# Using TileServer-GL Docker
docker run -it -v $(pwd):/data -p 8080:8080 maptiler/tileserver-gl

# Place your .mbtiles file in current directory
# Access at http://localhost:8080
```

---

### Option 2: Online Tile Services (Requires Internet)

#### MapTiler Cloud
- **URL**: https://www.maptiler.com/
- **Free Tier**: 100,000 tile requests/month
- **Styles**: 20+ pre-made styles
- **API Key**: Required (free registration)
- **MapLibre Compatible**: ✅ Yes

**Integration**:
```javascript
// MapLibre GL JS
const map = new maplibregl.Map({
  container: 'map',
  style: 'https://api.maptiler.com/maps/streets/style.json?key=YOUR_API_KEY',
  center: [-0.1276, 51.5074], // London
  zoom: 12
});
```

#### Stadia Maps
- **URL**: https://stadiamaps.com/
- **Free Tier**: 200,000 tile requests/month (non-commercial)
- **Styles**: Alidade, OSM Bright, Outdoors
- **API Key**: Required

#### Mapbox (Commercial, NOT Recommended)
- **Note**: Mapbox GL JS v2+ requires commercial license
- **Alternative**: Use MapLibre GL JS instead (open-source fork)

---

### Option 3: Generate Your Own Tiles (Advanced)

#### Using Tippecanoe + OS OpenData

**What You Need**:
- OS OpenData (https://www.ordnancesurvey.co.uk/opendatadownload/products.html)
- Tippecanoe (https://github.com/felt/tippecanoe)
- GeoJSON data

**Steps**:
```bash
# 1. Download OS OpenData (e.g., Roads, Buildings, Waterways)
wget https://api.os.uk/downloads/v1/products/OpenMapLocal/...

# 2. Convert to GeoJSON (if needed)
ogr2ogr -f GeoJSON roads.geojson OS_OpenMap_Local.gml

# 3. Generate MBTiles with Tippecanoe
tippecanoe -o uk-map.mbtiles \\
  -Z6 -z14 \\  # Zoom levels 6-14
  --drop-densest-as-needed \\
  --extend-zooms-if-still-dropping \\
  roads.geojson buildings.geojson

# 4. Serve with TileServer-GL
tileserver-gl uk-map.mbtiles
```

**Advantages**:
- Full control over data and styling
- Completely offline
- Custom layers (utilities, infrastructure)

---

## GPR Data (Existing)

### Currently Integrated (Production-Ready)

#### 1. Twente University GPR Dataset
- **Location**: `/datasets/raw/twente_gpr/`
- **Volume**: 125 scans, ~500MB
- **Format**: ZIP files with SEG-Y data
- **Metadata**: 25+ fields (environmental, technical, spatial)
- **Ground Truth**: Trial trench validation included

#### 2. Mojahid GPR Images Dataset
- **Location**: `/datasets/raw/mojahid_images/GPR_data/`
- **Volume**: 2,239+ labeled images
- **Format**: JPEG + YOLO annotations
- **Categories**: 6 categories (cavities, utilities, intact, etc.)

---

## Sample HS2-Style Infrastructure Data

### What HS2-Type Projects Provide

**Typical Data Deliverables**:
1. **LiDAR Scans**:
   - Pre-construction baseline (full corridor)
   - Monthly progress scans
   - High-resolution (10cm-25cm)

2. **BIM Models**:
   - Civil 3D corridor models
   - Revit station models
   - Bridge/viaduct IFC models
   - MEP infrastructure models

3. **Documentation**:
   - Construction drawings (CAD)
   - As-built records
   - Quality assurance reports

4. **GIS Data**:
   - Utility surveys
   - Site boundaries
   - Environmental constraints

---

### Alternative Infrastructure Projects to Contact

**UK Projects**:
1. **Crossrail/Elizabeth Line** - TfL
2. **Thames Tideway Tunnel** - Tideway Ltd
3. **Lower Thames Crossing** - National Highways
4. **Northern Powerhouse Rail** - Transport for the North

**International**:
5. **California High-Speed Rail** - CHSRA (USA)
6. **Grand Paris Express** - Société du Grand Paris (France)

---

## Data Preparation Workflow

### LiDAR Data Preparation

```bash
# 1. Download Environment Agency LiDAR (LAZ format)
# 2. Extract to project directory
mkdir -p data/lidar/raw

# 3. Convert LAZ to LAS if needed (optional)
laszip -i input.laz -o output.las

# 4. Inspect point cloud
pdal info data/lidar/raw/your-file.laz

# 5. Upload via API
curl -X POST http://localhost:8000/api/v1/lidar/upload \\
  -F "file=@data/lidar/raw/your-file.laz" \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### BIM Data Preparation

```bash
# 1. Download sample IFC from buildingSMART
# 2. Validate IFC file
ifcconvert --validate FZK-Haus-2012.ifc

# 3. Upload via API
curl -X POST http://localhost:8000/api/v1/bim/upload \\
  -F "file=@FZK-Haus-2012.ifc" \\
  -F "project_id=proj_123" \\
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Basemap Setup (Local Development)

```bash
# 1. Download UK MBTiles from OpenMapTiles
wget https://openmaptiles.org/download/great-britain.mbtiles

# 2. Place in tiles directory
mkdir -p tiles
mv great-britain.mbtiles tiles/

# 3. Start TileServer-GL (Docker Compose handles this)
docker compose up tileserver

# 4. Access tiles at http://localhost:8080
```

---

## Data Storage Best Practices

### File Organization

```
/data/
├── gpr/
│   ├── raw/           # Original SEG-Y, DZT files
│   └── processed/     # Analyzed data
├── bim/
│   ├── ifc/           # IFC files
│   ├── revit/         # RVT files
│   └── converted/     # glTF for web viewing
├── lidar/
│   ├── raw/           # LAS/LAZ files
│   ├── processed/     # Classified point clouds
│   └── potree/        # Web-optimized format
└── tiles/
    └── uk-map.mbtiles # Basemap tiles
```

### MinIO Bucket Structure (Production)

```
infrastructure-intelligence-platform/
├── gpr-data/          # GPR files
├── bim-models/        # BIM/IFC files
├── lidar-scans/       # Point clouds
├── documents/         # PDFs, CAD
└── reports/           # Generated reports
```

---

## Quick Start Checklist

### For Local Development

- [ ] Download Environment Agency LiDAR for test area (1-2 tiles)
- [ ] Download FZK-Haus-2012.ifc from buildingSMART
- [ ] Download UK MBTiles from OpenMapTiles
- [ ] Place files in appropriate `/data` directories
- [ ] Run `docker compose up -d`
- [ ] Upload sample files via API or web interface

### For Production Testing

- [ ] Request HS2 sample data (via NDA)
- [ ] Set up AWS S3 buckets
- [ ] Configure MapTiler API key
- [ ] Load production LiDAR data (Environment Agency)
- [ ] Integrate commercial tile service

---

## Legal & Licensing Summary

| Data Source | License | Commercial Use | Attribution Required |
|-------------|---------|----------------|----------------------|
| Environment Agency LiDAR | OGL v3.0 | ✅ Yes | ✅ Yes |
| buildingSMART IFC | Varies | ✅ Usually | ✅ Check per file |
| OpenMapTiles | BSD+ODbL | ✅ Yes | ✅ Yes (OSM) |
| MapTiler Cloud | Commercial | ✅ Yes (paid) | ✅ Yes |
| HS2 Data | NDA Required | ⚠️ Restricted | ⚠️ Per agreement |

---

## Support & Resources

**LiDAR Processing**:
- PDAL Documentation: https://pdal.io/
- Open3D Tutorials: http://www.open3d.org/docs/

**BIM/IFC**:
- IfcOpenShell: https://ifcopenshell.org/
- IFC.js: https://ifcjs.github.io/info/

**Mapping**:
- MapLibre GL JS: https://maplibre.org/
- TileServer-GL: https://tileserver.readthedocs.io/

---

Last Updated: 2025-11-24

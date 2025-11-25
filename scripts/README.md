# Utility Scripts

This directory contains utility scripts for the Infrastructure Intelligence Platform.

## Available Scripts

### `generate_secret_key.py`
Generate secure random secret keys for environment variables.

**Usage**:
```bash
python3 scripts/generate_secret_key.py
```

**Output**: Secure random keys for SECRET_KEY, JWT_SECRET_KEY, POSTGRES_PASSWORD, and MINIO_ROOT_PASSWORD.

---

### `setup_env.sh`
Automated setup script that creates `.env` file from `.env.example` and populates it with secure random keys.

**Usage**:
```bash
./scripts/setup_env.sh
```

**What it does**:
1. Copies `.env.example` to `.env`
2. Generates secure random keys
3. Replaces placeholder values in `.env`
4. Displays generated credentials

**Note**: Run this script before `docker compose up` for the first time.

---

## Future Scripts (To Be Added)

### Phase 1 (Weeks 1-3)
- `create_data_dirs.sh` - Create data directory structure
- `download_sample_data.sh` - Download sample LiDAR/BIM files
- `init_database.sh` - Initialize database and run migrations
- `create_admin_user.py` - Create initial admin user

### Phase 2 (Weeks 4-7)
- `test_bim_upload.py` - Test BIM file upload
- `validate_ifc.py` - Validate IFC file format

### Phase 3 (Weeks 8-11)
- `test_lidar_upload.py` - Test LiDAR file upload
- `downsample_point_cloud.py` - Downsample large point clouds

### Phase 4 (Weeks 12-14)
- `generate_sample_embeddings.py` - Generate PGVector embeddings
- `test_llm_report.py` - Test LLM report generation

---

## Development

To add a new script:
1. Create the script in this directory
2. Make it executable: `chmod +x scripts/your_script.sh`
3. Add documentation to this README
4. Test the script locally
5. Commit to version control

---

Last Updated: 2025-11-24

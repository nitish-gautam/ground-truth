# Database Management Scripts

This directory contains utility scripts for managing the database schema and data.

## Available Scripts

### `create_tables.py`
Creates all database tables from SQLAlchemy models.

```bash
# Run from container
docker compose exec backend python scripts/database/create_tables.py

# Run locally (if virtual environment is set up)
python backend/scripts/database/create_tables.py
```

**Output**: Creates 17 tables including:
- GPR data tables (surveys, scans, signal_data, processing_results)
- Environmental data tables (environmental_data, weather_conditions, ground_conditions)
- Validation tables (validation_results, accuracy_metrics, ground_truth_data)
- Utility tables (utility_disciplines, utility_materials, utility_records)
- ML tables (ml_models, training_sessions, feature_vectors, model_performance)

### `drop_tables.py`
Drops all database tables. **USE WITH CAUTION!**

```bash
docker compose exec backend python scripts/database/drop_tables.py
```

**Warning**: This will permanently delete all tables and data. You will be prompted to confirm.

### `reset_database.py`
Drops and recreates all tables. **USE WITH CAUTION!**

```bash
docker compose exec backend python scripts/database/reset_database.py
```

**Warning**: This will permanently delete all data and recreate empty tables. You will be prompted to confirm.

### `list_tables.py`
Lists all tables, columns, indexes, and foreign keys in the database.

```bash
docker compose exec backend python scripts/database/list_tables.py
```

**Output**: Detailed information about all database tables including:
- Column names, types, and nullability
- Indexes and uniqueness constraints
- Foreign key relationships

## Database Schema Overview

### GPR Data Model
```
gpr_surveys (parent)
  ├── gpr_scans
  │   ├── gpr_signal_data
  │   └── gpr_processing_results
  └── environmental_data
```

### Environmental Data Model
```
environmental_data
  ├── weather_conditions
  └── ground_conditions
```

### Validation Data Model
```
validation_results
  ├── accuracy_metrics
  └── ground_truth_data
```

### Utility Reference Data
```
utility_disciplines (reference table)
utility_materials (reference table)
utility_records (actual utility locations)
```

### ML Model Tracking
```
ml_models (parent)
  ├── training_sessions
  ├── feature_vectors
  └── model_performance
```

## Common Operations

### Initial Setup
```bash
# Create all tables
docker compose exec backend python scripts/database/create_tables.py
```

### View Database State
```bash
# List all tables
docker compose exec backend python scripts/database/list_tables.py

# Or use psql directly
docker compose exec postgres psql -U gpr_user -d gpr_db -c "\dt"
```

### Reset Database (Development)
```bash
# Drop and recreate tables
docker compose exec backend python scripts/database/reset_database.py
```

### Direct SQL Access
```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U gpr_user -d gpr_db

# Common commands:
\dt           # List tables
\d table_name # Describe table
\l            # List databases
\q            # Quit
```

## Database Connection Details

When running in Docker Compose:
- **Host**: postgres (internal) or localhost:5433 (external)
- **Database**: gpr_db
- **User**: gpr_user
- **Password**: Stored in .env file
- **Extensions**: PostGIS, PGVector

## Migration Strategy

Currently using direct SQLAlchemy table creation. For production, consider:

1. **Alembic** (recommended for production)
   - Track schema changes over time
   - Support rollback of migrations
   - Team collaboration on schema changes

2. **SQLAlchemy create_all()** (current approach)
   - Simple and fast for development
   - No migration history
   - Suitable for rapid prototyping

## Backup and Restore

```bash
# Backup database
docker compose exec postgres pg_dump -U gpr_user gpr_db > backup.sql

# Restore database
docker compose exec -T postgres psql -U gpr_user -d gpr_db < backup.sql
```

## Troubleshooting

### Tables Already Exist
If you see "table already exists" errors:
```bash
# Drop all tables first
docker compose exec backend python scripts/database/drop_tables.py

# Then create again
docker compose exec backend python scripts/database/create_tables.py
```

### Connection Refused
If database connection fails:
```bash
# Check if PostgreSQL is running
docker compose ps postgres

# Check logs
docker compose logs postgres

# Restart if needed
docker compose restart postgres
```

### Permission Denied
If you get permission errors:
```bash
# Check database user permissions
docker compose exec postgres psql -U gpr_user -d gpr_db -c "\du"
```

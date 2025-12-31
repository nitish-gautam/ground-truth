# Quick Access & Credentials

⚠️ **DEVELOPMENT ONLY** - Never use these credentials in production

---

## 🚀 Quick Access URLs

### Main Services
```
Frontend:    http://localhost:3003/hs2
API Docs:    http://localhost:8002/docs (Swagger)
Health:      http://localhost:8002/health
```

### Service Status

| Service | URL | Auth Required |
|---------|-----|---------------|
| HS2 Dashboard | http://localhost:3003/hs2 | ❌ No |
| Backend API | http://localhost:8002 | ❌ No |
| Neo4j Browser | http://localhost:7475 | ✅ Yes |
| MinIO Console | http://localhost:9011 | ✅ Yes |
| PostgreSQL | localhost:5433 | ✅ Yes |
| Redis | localhost:6380| ❌ No |

---

## 🔑 Credentials

### MinIO S3 Storage
```
URL:         http://localhost:9011
Username:    minioadmin
Password:    mD9E3_kgZJAPRjNvBWOxGQ
```

### PostgreSQL Database
```
Host:        localhost:5433
Database:    gpr_db
Username:    gpr_user
Password:    Lb1RcTOayzhQlwhU2E9dbA

# Connection String
postgresql://gpr_user:Lb1RcTOayzhQlwhU2E9dbA@localhost:5433/gpr_db

# Docker exec (no password needed)
docker exec -it infrastructure-postgres psql -U postgres -d infrastructure_db
```

### Redis Cache
```
Host:        localhost:6379
Password:    (none)
Command:     redis-cli -h localhost -p 6379
```

### Neo4j Graph Database
```
Browser URL: http://localhost:7475
Bolt URI:    bolt://localhost:7688
Username:    neo4j
Password:    hs2_graph_2024

# Connect via Cypher Shell
docker exec -it hs2-neo4j cypher-shell -u neo4j -p hs2_graph_2024
```

---

## Environment Variables

All credentials stored in `.env` file:
```bash
POSTGRES_USER=gpr_user
POSTGRES_PASSWORD=Lb1RcTOayzhQlwhU2E9dbA
POSTGRES_DB=gpr_db
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=mD9E3_kgZJAPRjNvBWOxGQ
SECRET_KEY=jIqpQG25aFDWIOK3cDQZmQY5NLHUTq-ltygbuOx-KOM
```

---

## Production Security

**Generate secure credentials:**
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Use:**
- AWS Secrets Manager / Azure Key Vault
- SSL/TLS for all connections
- SSO/SAML authentication
- Different credentials per environment

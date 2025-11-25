#!/bin/bash
# Setup environment file with generated secrets

set -e

echo "=================================================="
echo "Infrastructure Intelligence Platform - Setup .env"
echo "=================================================="
echo

# Check if .env already exists
if [ -f .env ]; then
    echo "⚠️  .env file already exists!"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Setup cancelled."
        exit 1
    fi
fi

# Copy .env.example to .env
echo "📄 Copying .env.example to .env..."
cp .env.example .env

# Generate secure keys
echo "🔐 Generating secure keys..."
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
POSTGRES_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(16))")
MINIO_ROOT_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(16))")

# Update .env file (macOS and Linux compatible)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/changeme_secure_password_here/$POSTGRES_PASSWORD/g" .env
    sed -i '' "s/changeme_minio_password_at_least_8_chars/$MINIO_ROOT_PASSWORD/g" .env
    sed -i '' "s/changeme_random_secret_key_at_least_32_characters_long/$SECRET_KEY/g" .env
else
    # Linux
    sed -i "s/changeme_secure_password_here/$POSTGRES_PASSWORD/g" .env
    sed -i "s/changeme_minio_password_at_least_8_chars/$MINIO_ROOT_PASSWORD/g" .env
    sed -i "s/changeme_random_secret_key_at_least_32_characters_long/$SECRET_KEY/g" .env
fi

echo "✅ .env file created with secure keys"
echo
echo "=================================================="
echo "📋 Your generated credentials:"
echo "=================================================="
echo "SECRET_KEY: $SECRET_KEY"
echo "POSTGRES_PASSWORD: $POSTGRES_PASSWORD"
echo "MINIO_ROOT_PASSWORD: $MINIO_ROOT_PASSWORD"
echo "=================================================="
echo
echo "⚠️  IMPORTANT: Keep these credentials secure!"
echo "⚠️  NEVER commit .env file to version control!"
echo
echo "✅ Setup complete! Run 'docker compose up -d' to start."

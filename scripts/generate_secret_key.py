#!/usr/bin/env python3
"""
Generate secure random secret keys for .env file
Usage: python scripts/generate_secret_key.py
"""

import secrets

def generate_secret_key(length: int = 32) -> str:
    """Generate a secure random secret key"""
    return secrets.token_urlsafe(length)

def main():
    print("=" * 60)
    print("Secret Key Generator for Infrastructure Intelligence Platform")
    print("=" * 60)
    print()

    print("Copy these values to your .env file:")
    print()

    print("# Application Secret Key")
    print(f"SECRET_KEY={generate_secret_key(32)}")
    print()

    print("# JWT Secret Key (can be same as SECRET_KEY)")
    print(f"JWT_SECRET_KEY={generate_secret_key(32)}")
    print()

    print("# Database Password")
    print(f"POSTGRES_PASSWORD={generate_secret_key(16)}")
    print()

    print("# MinIO Password")
    print(f"MINIO_ROOT_PASSWORD={generate_secret_key(16)}")
    print()

    print("=" * 60)
    print("IMPORTANT: Keep these values secret and secure!")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""Script to verify that the dummy_table in Supabase exists and can be queried."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from supabase import create_client


def main():
    if not settings.supabase_url or not settings.supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        return

    client = create_client(settings.supabase_url, settings.supabase_key)
    try:
        response = client.table("dummy_table").select("*").execute()
        rows = response.data or []
        print(f"Successfully connected to dummy_table! Total rows found: {len(rows)}")
        for i, row in enumerate(rows, 1):
            print(f"  [{i}] Age: {row.get('age')}, Phone: {row.get('phone_no')}, Address: {row.get('address')}")
    except Exception as exc:
        print(f"dummy_table not found or unreachable yet: {exc}")
        print("Remember to run scripts/create_dummy_table.sql in the Supabase SQL Editor first!")


if __name__ == "__main__":
    main()

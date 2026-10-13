# Session 3 (2026-09-04): local setup for Ledgerly

User: Ledgerly uses PostgreSQL 16 as its database.

User: Before running the test suite, start the database container with `docker compose up -d db`, otherwise pytest fails to connect.

User: The migrations in `migrations/` use JSONB columns for the audit log, so don't try to run them against anything else.

Assistant: Noted: Ledgerly is on PostgreSQL 16, tests need the `db` container running first, and the migrations rely on JSONB.

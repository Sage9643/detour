-- Runs once, when the Postgres container initializes an empty data volume.
-- The test suite drops and recreates this database's schema; never point it at real data.
CREATE DATABASE detour_test OWNER detour;

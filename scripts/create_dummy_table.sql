-- ============================================================
-- Supabase SQL: Create Dummy Table (age, phone_no, address)
-- Run this in your Supabase Dashboard:
-- SQL Editor -> "+ New query" -> Paste this script -> Click "Run"
-- ============================================================

-- 1. Create the dummy table
create table if not exists public.dummy_table (
    id          uuid        primary key default gen_random_uuid(),
    age         integer     check (age >= 0 and age <= 150),
    phone_no    text,       -- Standard identifier (or "phone no" if literal space is required)
    address     text,
    created_at  timestamptz not null default now()
);

-- Optional comment describing the table
comment on table public.dummy_table is 'Dummy table with age, phone_no, and address columns.';

-- 2. Enable Row Level Security (RLS)
alter table public.dummy_table enable row level security;

-- 3. Allow service_role (and authenticated/anon if needed for testing)
create policy "Allow all access to service role"
    on public.dummy_table
    for all
    to service_role
    using (true)
    with check (true);

create policy "Allow read access to anon users"
    on public.dummy_table
    for select
    to anon
    using (true);

-- 4. Insert sample dummy records
insert into public.dummy_table (age, phone_no, address)
values
    (28, '+1-555-0199', '742 Evergreen Terrace, Springfield, OR'),
    (34, '+1-555-0142', '221B Baker Street, London, UK'),
    (42, '+1-555-0178', '124 Conch Street, Bikini Bottom, Pacific Ocean')
on conflict do nothing;

-- 5. Verify the data
select * from public.dummy_table;

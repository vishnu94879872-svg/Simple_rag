-- BookRAG Reliability Lab — Supabase observability schema
-- Run this once in the Supabase dashboard: SQL Editor -> New query -> Run.
-- The app writes with the service_role key from the server only (.env, never committed).

create table if not exists public.observability_records (
    kind        text        not null,
    record_id   text        not null,
    payload     jsonb       not null default '{}'::jsonb,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now(),
    constraint observability_records_pkey primary key (kind, record_id)
);

comment on table public.observability_records is
    'BookRAG observability records: job, trace, eval_run, benchmark, experiment, investigation.';
comment on column public.observability_records.kind is
    'Record family; one of job | trace | eval_run | benchmark | experiment | investigation.';
comment on column public.observability_records.payload is
    'Full Pydantic document serialized as JSON.';

create index if not exists observability_records_kind_updated_idx
    on public.observability_records (kind, updated_at desc);

create or replace function public.bookrag_touch_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists observability_records_touch_updated_at on public.observability_records;
create trigger observability_records_touch_updated_at
    before update on public.observability_records
    for each row execute function public.bookrag_touch_updated_at();

-- RLS is enabled as a safety net. The service_role key bypasses RLS, which is
-- what the Streamlit server uses. Do not expose the service_role key client-side.
alter table public.observability_records enable row level security;

-- If you intentionally want to use the anon key instead, uncomment the policy
-- below. It grants full read/write to anyone holding the anon key — only do this
-- for a private demo project, never for production data.
-- create policy "anon demo access"
--     on public.observability_records
--     for all
--     to anon
--     using (true)
--     with check (true);

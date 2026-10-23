-- ============================================================
-- BookRAG Cloud Storage Schema — Run ONCE in Supabase SQL Editor
-- Enables persistent vector + BM25 storage for cloud deployment
-- ============================================================

-- Step 1: Enable pgvector extension
create extension if not exists vector;

-- Step 2: RAG chunks table (replaces local ChromaDB on cloud)
create table if not exists public.rag_chunks (
    chunk_id     text          not null primary key,
    content      text          not null,
    embedding    vector(384),  -- BGE-small-en-v1.5 dimension = 384
    metadata     jsonb         not null default '{}'::jsonb,
    book_title   text          generated always as (metadata->>'book_title') stored,
    created_at   timestamptz   not null default now(),
    updated_at   timestamptz   not null default now()
);

comment on table public.rag_chunks is
    'BookRAG: persistent chunk + vector storage for cloud deployment.';

-- Index for fast cosine similarity search
create index if not exists rag_chunks_embedding_idx
    on public.rag_chunks
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- Index for fast book_title filtering
create index if not exists rag_chunks_book_title_idx
    on public.rag_chunks (book_title);

-- Step 3: BM25 index persistence table (JSON-serialized)
create table if not exists public.rag_bm25_index (
    id          text        not null primary key default 'singleton',
    payload     jsonb       not null default '{}'::jsonb,
    updated_at  timestamptz not null default now()
);

comment on table public.rag_bm25_index is
    'BookRAG: JSON-serialized BM25 index for cloud persistence.';

-- Step 4: pgvector cosine similarity search function
create or replace function public.rag_vector_search(
    query_embedding vector(384),
    top_k           int     default 20,
    filter_book     text    default null
)
returns table (
    chunk_id    text,
    content     text,
    metadata    jsonb,
    distance    float
)
language sql stable
as $$
    select
        chunk_id,
        content,
        metadata,
        (embedding <=> query_embedding) as distance
    from public.rag_chunks
    where
        (filter_book is null or book_title = filter_book)
        and embedding is not null
    order by embedding <=> query_embedding
    limit top_k;
$$;

-- Step 5: updated_at triggers
create or replace function public.rag_touch_updated_at()
returns trigger language plpgsql as $$
begin new.updated_at = now(); return new; end;
$$;

drop trigger if exists rag_chunks_touch_updated_at on public.rag_chunks;
create trigger rag_chunks_touch_updated_at
    before update on public.rag_chunks
    for each row execute function public.rag_touch_updated_at();

drop trigger if exists rag_bm25_touch_updated_at on public.rag_bm25_index;
create trigger rag_bm25_touch_updated_at
    before update on public.rag_bm25_index
    for each row execute function public.rag_touch_updated_at();

-- Step 6: RLS (service_role key bypasses; safe for server-side Streamlit)
alter table public.rag_chunks enable row level security;
alter table public.rag_bm25_index enable row level security;

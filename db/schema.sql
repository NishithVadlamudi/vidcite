-- vidcite schema. Already applied to the Supabase project 'vidcite' (us-west-1).
-- Re-run in the SQL editor only if you create a fresh project.
create extension if not exists vector with schema extensions;

create table if not exists videos (
  id           text primary key,          -- YouTube video ID
  title        text not null,
  channel      text,
  duration_s   int,
  license      text not null,             -- e.g. 'Creative Commons Attribution license (reuse allowed)'
  ingested_at  timestamptz default now()
);

create table if not exists chunks (
  id           bigserial primary key,
  video_id     text not null references videos(id) on delete cascade,
  source       text not null check (source in ('youtube', 'whisper')),  -- compared in the eval
  chunk_index  int  not null,
  start_s      real not null,
  end_s        real not null,
  text         text not null,
  embedding    vector(384) not null,      -- BAAI/bge-small-en-v1.5, normalized
  tsv          tsvector generated always as (to_tsvector('english', text)) stored,
  unique (video_id, source, chunk_index)
);

create index if not exists chunks_embedding_hnsw on chunks using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_tsv_gin        on chunks using gin (tsv);
create index if not exists chunks_source         on chunks (source);

-- Lock the tables away from Supabase's public REST API. Our code connects
-- directly to Postgres as the owner, which bypasses RLS.
alter table videos enable row level security;
alter table chunks enable row level security;

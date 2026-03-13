create extension if not exists vector;

create table if not exists books (
  book_id text primary key,
  title text not null,
  author text not null,
  source_path text not null,
  source_type text not null,
  raw_text text not null,
  ingested_at timestamptz not null
);

create table if not exists chapters (
  chapter_id text primary key,
  book_id text not null references books (book_id) on delete cascade,
  chapter_number integer not null,
  title text not null,
  summary text not null,
  chunk_ids jsonb not null
);

create table if not exists chunks (
  chunk_id text primary key,
  book_id text not null references books (book_id) on delete cascade,
  chapter_id text not null references chapters (chapter_id) on delete cascade,
  chapter_title text not null,
  chapter_number integer not null,
  sequence integer not null,
  text text not null,
  start_word integer not null,
  end_word integer not null,
  source_offsets jsonb not null,
  themes jsonb not null
);

create table if not exists chunk_embeddings (
  chunk_id text primary key references chunks (chunk_id) on delete cascade,
  book_id text not null references books (book_id) on delete cascade,
  embedding vector(8) not null
);

create table if not exists episode_plans (
  episode_id text primary key,
  book_id text not null references books (book_id) on delete cascade,
  payload jsonb not null,
  created_at timestamptz not null default now()
);

create table if not exists episode_scripts (
  episode_id text primary key references episode_plans (episode_id) on delete cascade,
  payload jsonb not null,
  created_at timestamptz not null default now()
);

create table if not exists grounding_reports (
  episode_id text primary key references episode_scripts (episode_id) on delete cascade,
  payload jsonb not null,
  created_at timestamptz not null default now()
);

create table if not exists repair_attempts (
  episode_id text not null references episode_scripts (episode_id) on delete cascade,
  attempt integer not null,
  payload jsonb not null,
  created_at timestamptz not null default now(),
  primary key (episode_id, attempt)
);

create index if not exists idx_chapters_book_id on chapters (book_id, chapter_number);
create index if not exists idx_chunks_book_id on chunks (book_id, chapter_number, sequence);
create index if not exists idx_chunk_embeddings_book_id on chunk_embeddings (book_id);

-- Follow-up: add PostgreSQL full-text and BM25-style ranking over chunks.text for hybrid retrieval.

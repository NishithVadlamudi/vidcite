-- Read-only login for the hosted demo. Run in Supabase > SQL editor.
-- Replace CHOOSE_A_STRONG_PASSWORD with your own; don't commit it anywhere.
--
-- The tables have row-level security on (schema.sql) and the app normally connects as the
-- owner, which bypasses it. This role doesn't, so it gets explicit SELECT-only policies.
-- Supabase's public REST roles (anon, authenticated) stay blocked: the policies name only this role.

create role vidcite_reader login password 'CHOOSE_A_STRONG_PASSWORD';

grant usage on schema public to vidcite_reader;
grant usage on schema extensions to vidcite_reader;          -- the vector type lives here
grant select on public.videos, public.chunks to vidcite_reader;

create policy vidcite_reader_read on public.videos for select to vidcite_reader using (true);
create policy vidcite_reader_read on public.chunks for select to vidcite_reader using (true);

-- Connection string for the Space's DATABASE_URL secret: copy your current DATABASE_URL and change
-- only the user and password (the pooler user is "<role>.<project-ref>"):
--   postgresql://vidcite_reader.<project-ref>:<password>@<same host>:5432/postgres

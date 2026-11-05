# FedAI Web

A read-only web interface for browsing the FedAI research corpus on the Federal Open Market Committee (FOMC): meetings and participants, transcripts with full-text search, Summary of Economic Projections (SEP) answers and projections, the federal funds rate dot plot, pre-meeting economic data, and question answering over the transcripts.

Architecture: React front end (Vite) → Fastify API → PostgreSQL through a read-only role.

## Quick start

```bash
npm install
npm run build
npm start            # http://127.0.0.1:8787
```

Without database settings the server starts in **demo mode**: a private local PostgreSQL instance is filled with clearly fictitious data, and every page shows a demo banner.

To connect to a real database, copy `.env.example` to `.env` and fill in a **read-only** role. Keep the password in `~/.pgpass` rather than in `.env`. The server checks the role at start-up and refuses any role that can write.

For development: `npm run dev:server` and `npm run dev:web`.

## Safety layers

- Only single `SELECT` / `WITH` statements are allowed.
- Sessions run with `default_transaction_read_only = on`.
- The database role is checked at start-up.
- Statement timeout, page-size and search-length limits apply.
- Error messages are stripped of hosts and user names.

## Tests

```bash
npm test             # unit and integration tests against an embedded PostgreSQL instance
npm run typecheck
```

## Benchmarks

`benchmark/` holds the baselines for simulating FOMC participants (individual dot-plot projections, SEP narrative answers,
policy decisions and votes), the AI experiments with look-ahead (memory) tests, and plug-in scoring for other methods.
It reads the same database through a read-only role. See `benchmark/README.md`.

## Deployment

`deploy/` contains a systemd unit and an nginx site configuration (team login through HTTP basic authentication). `deploy/update_from_bundle.sh` updates a server from a git bundle; the server login is read from `FEDAI_HOST_SSH` or from a local `deploy/.host` file that is not part of the repository.

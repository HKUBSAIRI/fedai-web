import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import type { ApiError, Row } from './api';

export function SqlBox({ sql, title = 'SQL and definition' }: { sql: string | string[] | null | undefined; title?: string }) {
  if (!sql) return null;
  const text = Array.isArray(sql) ? sql.join(';\n\n') : sql;
  return (
    <details className="sql">
      <summary>{title}</summary>
      <pre>{text.replace(/\n\s{6,}/g, '\n  ')}</pre>
    </details>
  );
}

export function Loading({ what = 'data' }: { what?: string }) {
  return <p className="muted">Loading {what}…</p>;
}

export function ErrorBox({ error }: { error: ApiError }) {
  return (
    <div className={error.unavailable ? 'notice' : 'error'}>
      <strong>{error.unavailable ? 'Not available:' : 'Error:'}</strong> {error.message}
    </div>
  );
}

export function Pager({ page, pageSize, total, hasMore, onPage }: {
  page: number; pageSize: number; total?: number; hasMore?: boolean; onPage: (p: number) => void;
}) {
  const pages = total !== undefined ? Math.max(1, Math.ceil(total / pageSize)) : undefined;
  const next = pages !== undefined ? page < pages : Boolean(hasMore);
  return (
    <div className="pager">
      <button disabled={page <= 1} onClick={() => onPage(page - 1)}>Previous</button>
      <span>Page {page}{pages !== undefined ? ` of ${pages}` : ''}{total !== undefined ? ` (${total.toLocaleString()} rows)` : ''}</span>
      <button disabled={!next} onClick={() => onPage(page + 1)}>Next</button>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="empty">{children}</p>;
}

export function Status({ value }: { value: unknown }) {
  const v = String(value ?? '—');
  return <span className={`tag tag-${v}`}>{v}</span>;
}

export function Val({ v }: { v: unknown }) {
  if (v === null || v === undefined) return <span className="null">NULL</span>;
  if (v === '') return <span className="null">(empty)</span>;
  return <>{String(v)}</>;
}

/** Generic key/value view of a DB row — shows every column the database returned. */
export function RowTable({ row }: { row: Row }) {
  return (
    <table className="kv">
      <tbody>
        {Object.entries(row).map(([k, v]) => (
          <tr key={k}><th>{k}</th><td className="mono"><Val v={v} /></td></tr>
        ))}
      </tbody>
    </table>
  );
}

export const MeetingLink = ({ id, date }: { id: unknown; date: unknown }) =>
  <Link to={`/meetings/${id}`}>{String(date ?? id).slice(0, 10)}</Link>;
export const PersonLink = ({ id, name }: { id: unknown; name: unknown }) =>
  id ? <Link to={`/participants/${id}`}>{String(name ?? id)}</Link> : <span className="null">unmapped</span>;

/**
 * Highlights keyword matches (case-insensitive) in plain text without using innerHTML.
 * words = full-text mode: highlight each search word and its other forms (same first letters), ignoring OR / -excluded words.
 */
export function Highlight({ text, term, words = false }: { text: string; term: string; words?: boolean }) {
  if (!term) return <>{text}</>;
  const esc = (x: string) => x.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  let pattern: string;
  if (words) {
    const parts = (term.match(/"[^"]+"|\S+/g) ?? [])
      .filter((w) => !/^or$/i.test(w) && !w.startsWith('-'))
      .map((w) => w.replace(/"/g, '').trim()).filter((w) => w.length >= 2);
    if (!parts.length) return <>{text}</>;
    pattern = parts.map((w) => (w.includes(' ') ? esc(w) : `\\b${esc(w.length > 4 ? w.slice(0, w.length - 2) : w)}\\w*`)).join('|');
  } else pattern = esc(term);
  const re = new RegExp(`(${pattern})`, 'gi');
  const split = text.split(re);
  return <>{split.map((p, i) => (i % 2 ? <mark key={i}>{p}</mark> : p))}</>;
}

/** Per-meeting role. Inferred roles (database value empty) are marked so they are not mistaken for recorded ones. */
export function RoleBadge({ role, inferred }: { role: unknown; inferred?: unknown }) {
  const r = String(role ?? '—');
  return <span className={`role role-${r.replace(/\s+/g, '-').toLowerCase()}`} title={inferred ? 'Inferred: not recorded in the database' : undefined}>
    {r}{inferred && r !== 'Staff' ? '*' : ''}</span>;
}

export const ROLE_NOTE = 'Chair, Vice Chair and Voter come from the database. When the database has no role, a person is a non-voting '
  + 'Participant (*) if the meeting falls within their time as an FOMC official, otherwise Staff. The FOMC has at most 12 voters; '
  + 'there are fewer when Board seats are vacant.';

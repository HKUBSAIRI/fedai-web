import { useApi, type Row } from '../api';
import type { StatusData } from '../App';
import { ErrorBox, Loading, SqlBox, Val } from '../components';

interface Check { id: string; severity: 'info' | 'warn' | 'error'; title: string; description: string; sql: string; count: number | null; sample: Row[]; error?: string;
  verified?: { count: number; note: string } }
interface Dist { id: string; title: string; sql: string; rows: Row[]; error?: string }
interface Cross { id: string; title: string; ours: number | null; theirs: number | null; oursSql: string; theirsSql: string; error?: string }
interface QData { distributions: Dist[]; crossChecks: Cross[]; checks: Check[]; coverage: Row[]; coverageSql: string | null; unavailable: { area: string; reason: string }[] }

const SEV = { error: 'Error', warn: 'Warning', info: 'Info' } as const;

export default function Quality({ status }: { status?: StatusData }) {
  const { data, error, loading } = useApi<QData>('/quality');
  return (
    <>
      <h1>Data Quality</h1>
      <p className="muted">Every check is a read-only SQL query run live; problems are listed, never changed. Expand a check for its SQL and the first 20 hits.</p>

      <section>
        <h2>Connection and permissions</h2>
        {status && (
          <table className="kv">
            <tbody>
              <tr><th>Data source</th><td>{status.demo ? <strong className="warn-text">DEMO data (fictitious)</strong> : 'FedAI database'}</td></tr>
              <tr><th>Database configured / connected</th><td>{status.realDb.configured ? 'yes' : 'no'} / {status.realDb.connected ? 'yes' : 'no'}</td></tr>
              <tr><th>Database status</th><td><Val v={status.realDb.error ?? 'OK'} /></td></tr>
              <tr><th>Read-only account check</th><td>{status.role?.ok ? 'Passed (not a superuser, no write privileges, read-only session)' : `Failed: ${status.role?.problems.join('; ')}`}</td></tr>
              <tr><th>Tables in the schema</th><td className="mono small">{status.tables.join(', ')}</td></tr>
            </tbody>
          </table>
        )}
      </section>

      {error && <ErrorBox error={error} />}
      {loading && !data && <Loading />}
      {data && (
        <>
          {data.unavailable.length > 0 && (
            <section>
              <h2>Not covered yet</h2>
              <ul>{data.unavailable.map((u) => <li key={u.area}><strong>{u.area}: </strong>{u.reason}</li>)}</ul>
            </section>
          )}
          {data.crossChecks.length > 0 && (
            <section>
              <h2>Cross-checks</h2>
              <p className="muted small">This site’s “active version” joins compared with the database’s current_* views; equal numbers mean the same definition.</p>
              <table className="data">
                <thead><tr><th>Check</th><th className="num">This site</th><th className="num">View</th><th>Result</th></tr></thead>
                <tbody>{data.crossChecks.map((c) => (
                  <tr key={c.id} className={c.ours !== c.theirs ? 'sev-warn' : ''}>
                    <td>{c.title}</td><td className="num">{c.ours ?? '—'}</td><td className="num">{c.theirs ?? '—'}</td>
                    <td>{c.error ? <span className="error-text">{c.error}</span> : c.ours === c.theirs ? 'Match ✓' : 'Mismatch — check'}</td>
                  </tr>))}</tbody>
              </table>
              <SqlBox sql={data.crossChecks.flatMap((c) => [c.oursSql, c.theirsSql])} />
            </section>
          )}
          {data.distributions.length > 0 && (
            <section>
              <h2>Distributions</h2>
              <div className="grid2">{data.distributions.map((d) => (
                <div key={d.id}>
                  <h3 className="chart-title">{d.title}</h3>
                  {d.error ? <div className="error">{d.error}</div> : (
                    <table className="data narrow">
                      <thead><tr>{Object.keys(d.rows[0] ?? {}).map((k) => <th key={k}>{k}</th>)}</tr></thead>
                      <tbody>{d.rows.map((r, i) => <tr key={i}>{Object.values(r).map((v, j) => <td key={j} className={j ? 'num' : ''}><Val v={v} /></td>)}</tr>)}</tbody>
                    </table>)}
                  <SqlBox sql={d.sql} title="SQL" />
                </div>))}
              </div>
            </section>
          )}
          <section>
            <h2>Checks</h2>
            <table className="data">
              <thead><tr><th>Level</th><th>Check</th><th className="num">Hits</th><th>Description</th></tr></thead>
              <tbody>{data.checks.map((c) => {
                // all hits checked by hand and correct as stored → not highlighted; any new hit keeps the highlight
                const v = c.verified && c.verified.count > 0 ? c.verified : undefined;
                const open = (c.count ?? 0) - (v?.count ?? 0);
                return (
                  <tr key={c.id} className={open > 0 ? `sev-${c.severity}` : ''}>
                    <td><span className={`sev sev-${c.severity}`}>{SEV[c.severity]}</span></td>
                    <td>{c.title}</td>
                    <td className="num">{c.error ? <span className="error-text">query failed</span> : c.count === 0 ? '0 ✓'
                      : !v ? c.count
                      : open === 0 ? <>{c.count}<div className="verified">verified ✓</div></>
                      : <>{c.count}<div className="small">verified {v.count} · <strong>to check {open}</strong></div></>}</td>
                    <td className="small">{c.description}{c.error && <div className="error-text">{c.error}</div>}
                      {v && <div className="verified-note">Verified{open > 0 ? ` (${v.count})` : ''}: {v.note}</div>}</td>
                  </tr>);
              })}</tbody>
            </table>
            {data.checks.map((c) => (c.count || c.error) ? (
              <details key={c.id} className="sql">
                <summary>{c.title} — SQL and sample ({c.sample.length} rows)</summary>
                <pre>{c.sql.replace(/\n\s{6,}/g, '\n  ')}</pre>
                {c.sample.length > 0 && (
                  <div className="matrix-wrap"><table className="data">
                    <thead><tr>{Object.keys(c.sample[0]).map((k) => <th key={k}>{k}</th>)}</tr></thead>
                    <tbody>{c.sample.map((r, i) => <tr key={i}>{Object.values(r).map((v, j) => <td key={j} className="small"><Val v={v} /></td>)}</tr>)}</tbody>
                  </table></div>
                )}
              </details>
            ) : null)}
          </section>
          <section>
            <h2>Coverage by year: meetings vs SEP rounds</h2>
            <p className="muted small">The SEP comes with some meetings only (usually once a quarter), so “no SEP” is not a gap; a whole year at 0 usually means the year is not imported.</p>
            <table className="data narrow">
              <thead><tr><th>Year</th><th className="num">Meetings</th><th className="num">Meetings with an SEP</th></tr></thead>
              <tbody>{data.coverage.map((r) => (
                <tr key={String(r.year)} className={r.meetings_with_sep === 0 ? 'sev-warn' : ''}>
                  <td>{String(r.year)}</td><td className="num">{String(r.meetings)}</td><td className="num">{String(r.meetings_with_sep)}</td>
                </tr>))}</tbody>
            </table>
            <SqlBox sql={data.coverageSql} />
          </section>
        </>
      )}
    </>
  );
}

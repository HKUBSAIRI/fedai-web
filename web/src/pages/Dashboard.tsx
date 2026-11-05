import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useState } from 'react';
import { useApi } from '../api';
import { ErrorBox, Loading, SqlBox } from '../components';

interface Stat { key: string; label: string; value: number | string | null; definition: string; sql: string | null; note?: string }
interface Overview {
  stats: Stat[];
  responseStatus: { rows: { status: string; n: number }[]; sql: string | null };
  byYear: { rows: { year: number; meetings: number; sep_batches: number; sep_answers: number }[]; sql: string[] };
}

const fmt = (v: number | string | null) => (v === null ? 'n/a' : typeof v === 'number' ? v.toLocaleString() : v);

export default function Dashboard() {
  const { data, error, loading } = useApi<Overview>('/overview');
  if (error) return <ErrorBox error={error} />;
  if (loading && !data) return <Loading />;
  const d = data!;
  const totalAns = d.responseStatus.rows.reduce((a, r) => a + r.n, 0);
  return (
    <>
      <h1>Dashboard</h1>
      <p className="muted">Every number is a live read-only query; open “SQL” under a card to see exactly how it is counted.</p>
      <div className="cards">
        {d.stats.map((s) => (
          <div className="card" key={s.key}>
            <div className="card-label">{s.label}</div>
            <div className={`card-value ${s.value === null ? 'null' : ''}`}>{fmt(s.value)}</div>
            <div className="card-def">{s.definition}</div>
            {s.note && <div className="card-note">{s.note}</div>}
            <SqlBox sql={s.sql} title="SQL" />
          </div>
        ))}
      </div>

      <section>
        <h2>SEP narrative answers by status</h2>
        <p className="muted">answered = the respondent wrote an answer; na = “N/A” or “No comment” in the source (not applicable, or chose not to comment);
          missing = the respondent does not appear under that question in the source at all (only in the first SEP, October 2007). Counted separately, never merged.</p>
        <table className="data narrow">
          <thead><tr><th>Status</th><th className="num">Answers</th><th className="num">Share</th></tr></thead>
          <tbody>
            {d.responseStatus.rows.map((r) => (
              <tr key={r.status}><td>{r.status}</td><td className="num">{r.n.toLocaleString()}</td>
                <td className="num">{totalAns ? ((r.n / totalAns) * 100).toFixed(1) + '%' : '—'}</td></tr>
            ))}
            <tr className="total"><td>Total</td><td className="num">{totalAns.toLocaleString()}</td><td /></tr>
          </tbody>
        </table>
        <SqlBox sql={d.responseStatus.sql} />
      </section>

      <section>
        <h2>By year</h2>
        <YearCharts rows={d.byYear.rows} />
        <SqlBox sql={d.byYear.sql} />
      </section>
    </>
  );
}

const METRICS = [
  { key: 'meetings', label: 'Meetings' },
  { key: 'sep_batches', label: 'SEP rounds' },
  { key: 'sep_answers', label: 'SEP narrative answers' },
] as const;

/** One measure per chart (no dual axis); the selector switches the measure. */
function YearCharts({ rows }: { rows: Overview['byYear']['rows'] }) {
  const [metric, setMetric] = useState<(typeof METRICS)[number]['key']>('meetings');
  const [table, setTable] = useState(false);
  const m = METRICS.find((x) => x.key === metric)!;
  return (
    <div className="viz-root">
      <div className="filters">
        {METRICS.map((x) => (
          <button key={x.key} className={x.key === metric ? 'on' : ''} onClick={() => setMetric(x.key)}>{x.label}</button>
        ))}
        <button className="ghost" onClick={() => setTable(!table)}>{table ? 'Chart' : 'Table'}</button>
      </div>
      <h3 className="chart-title">{m.label}</h3>
      {table ? (
        <table className="data narrow">
          <thead><tr><th>Year</th>{METRICS.map((x) => <th key={x.key} className="num">{x.label}</th>)}</tr></thead>
          <tbody>{rows.map((r) => <tr key={r.year}><td>{r.year}</td>{METRICS.map((x) => <td key={x.key} className="num">{r[x.key]}</td>)}</tr>)}</tbody>
        </table>
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={rows} margin={{ top: 16, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis dataKey="year" tickLine={false} axisLine={{ stroke: 'var(--axis)' }} tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
            <YAxis allowDecimals={false} tickLine={false} axisLine={false} tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} width={44} />
            <Tooltip cursor={{ fill: 'var(--hover)' }} formatter={(v) => [Number(v).toLocaleString(), m.label]} labelFormatter={(y) => String(y)} />
            <Bar dataKey={metric} fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={36} label={rows.length <= 16 ? { position: 'top', fill: 'var(--text-secondary)', fontSize: 11 } : undefined} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

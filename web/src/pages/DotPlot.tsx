import { useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
  CartesianGrid, ComposedChart, Legend, Line, LineChart, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from 'recharts';
import { useApi, type Row } from '../api';
import { Empty, ErrorBox, Loading, MeetingLink, PersonLink, SqlBox } from '../components';

interface Dot { horizon: string; projection_year: number | null; is_long_run: boolean; participant_number: number; value: number; person_id: string | null; person_name: string | null }
interface Stat { horizon: string; n: number; median: number; min: number; max: number; q1: number; q3: number; iqr: number; outliers: { participant_number: number; person_name: string | null; value: number }[] }
interface PlotData { batch: Row; dots: Dot[]; stats: Stat[]; sql: string; outlierRule: string }
interface TrackRow { batch_id: string; meeting_start: string; horizon: string; is_long_run: boolean; year_offset: number | null; value: number; median: number; n: number }

const hLabel = (h: string) => (/^(lr|long)/i.test(h.trim()) ? 'Longer run' : h);
const pct = (v: number) => `${v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')}%`;

export default function DotPlot() {
  const [sp, setSp] = useSearchParams();
  const set = (k: string, v: string) => { const n = new URLSearchParams(sp); if (v) n.set(k, v); else n.delete(k); setSp(n, { replace: true }); };
  const batches = useApi<{ items: Row[]; sql: string }>('/dotplot/batches');
  const items = batches.data?.items ?? [];
  const batchId = sp.get('batch') ?? String(items[items.length - 1]?.batch_id ?? '');
  return (
    <>
      <h1>Federal funds rate dot plot</h1>
      <p className="muted">From sep_projections.federal_funds_rate: each participant’s projection of the year-end federal funds rate. The Fed began collecting these projections in January 2012, so there is no dot plot before that.</p>
      {batches.error && <ErrorBox error={batches.error} />}
      {batches.loading && !batches.data && <Loading />}
      {batches.data && !items.length && <Empty>No SEP round contains federal funds rate projections.</Empty>}
      {items.length > 0 && (
        <>
          <MeetingPlot batchId={batchId} batches={items} highlight={sp.get('hl') ?? ''}
            onBatch={(v) => set('batch', v)} onHighlight={(v) => set('hl', v)} />
          <PersonTrack personId={sp.get('person') ?? ''} offset={sp.get('h') ?? '0'}
            onPerson={(v) => set('person', v)} onOffset={(v) => set('h', v)} />
          <SqlBox sql={batches.data!.sql} title="Meeting list SQL" />
        </>
      )}
    </>
  );
}

function MeetingPlot({ batchId, batches, highlight, onBatch, onHighlight }: {
  batchId: string; batches: Row[]; highlight: string; onBatch: (v: string) => void; onHighlight: (v: string) => void;
}) {
  const { data, error, loading } = useApi<PlotData>(batchId ? `/dotplot/${batchId}` : null);
  const [table, setTable] = useState(false);
  // The dot under the pointer, set by each symbol's own mouse events. Recharts resolves the tooltip
  // payload by x position, so dots stacked in one column (same x) would show the first dot's values.
  type Hovered = (Dot & { x: number }) | { x: number; value: number };
  const [hover, setHover] = useState<Hovered | null>(null);
  const hoverProps = {
    onMouseEnter: (pt: { payload?: unknown }) => setHover((pt.payload as Hovered | undefined) ?? null),
    onMouseLeave: () => setHover(null),
  };
  const horizons = useMemo(() => (data ? data.stats.map((s) => s.horizon) : []), [data]);

  // Stack equal values side by side, like the published dot plot.
  const points = useMemo(() => {
    if (!data) return [];
    const out: (Dot & { x: number; hl: boolean })[] = [];
    horizons.forEach((h, i) => {
      const groups = new Map<number, Dot[]>();
      for (const d of data.dots.filter((x) => x.horizon === h)) {
        if (!groups.has(d.value)) groups.set(d.value, []);
        groups.get(d.value)!.push(d);
      }
      for (const g of groups.values()) {
        const step = Math.min(0.07, 0.8 / g.length);
        g.forEach((d, k) => out.push({ ...d, x: i + (k - (g.length - 1) / 2) * step, hl: d.person_id === highlight }));
      }
    });
    return out;
  }, [data, horizons, highlight]);
  const medians = data ? data.stats.map((s, i) => ({ x: i, value: s.median })) : [];
  // Leave room beyond the highest and lowest dots (some projections are negative, e.g. 2015-09) so edge dots
  // are not cut by the plot edge; ticks at a round step.
  const yTicks = useMemo(() => {
    const values = (data?.dots ?? []).map((d) => d.value);
    const top = Math.max(0, ...values) + 0.15;
    const bottom = Math.min(0, ...values.map((v) => v - 0.15));
    const step = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 5].find((st) => Math.ceil(top / st) - Math.floor(bottom / st) <= 6) ?? 5;
    const first = Math.floor(bottom / step);
    return Array.from({ length: Math.ceil(top / step) - first + 1 }, (_, i) => (first + i) * step);
  }, [data]);
  const people = useMemo(() => {
    const m = new Map<string, string>();
    for (const d of data?.dots ?? []) if (d.person_id) m.set(d.person_id, d.person_name ?? d.person_id);
    return [...m.entries()].sort((a, b) => a[1].localeCompare(b[1]));
  }, [data]);
  const hlName = people.find(([id]) => id === highlight)?.[1];

  return (
    <section className="viz-root">
      <h2>By meeting</h2>
      <div className="filters">
        <label>Meeting <select value={batchId} onChange={(e) => onBatch(e.target.value)}>
          {[...batches].reverse().map((b) => <option key={String(b.batch_id)} value={String(b.batch_id)}>{String(b.meeting_start)} ({String(b.participants)} participants)</option>)}
        </select></label>
        <label>Highlight <select value={highlight} onChange={(e) => onHighlight(e.target.value)}>
          <option value="">(none)</option>
          {people.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
        </select></label>
        <button className="ghost" onClick={() => setTable(!table)}>{table ? 'Chart' : 'Table'}</button>
      </div>
      {error && <ErrorBox error={error} />}
      {loading && !data && <Loading />}
      {data && (
        <>
          <p className="small">Meeting <MeetingLink id={data.batch.meeting_id} date={data.batch.meeting_start} /> · {data.dots.length} projections ·
            {' '}<Link to={`/sep/${data.batch.batch_id}`}>Full SEP for this meeting →</Link></p>
          {table ? (
            <table className="data">
              <thead><tr><th>Horizon</th><th className="num">#</th><th>Participant</th><th className="num">Rate</th></tr></thead>
              <tbody>{data.dots.map((d, i) => (
                <tr key={i} className={d.person_id === highlight ? 'hl-row' : ''}>
                  <td>{hLabel(d.horizon)}</td><td className="num">{d.participant_number}</td>
                  <td><PersonLink id={d.person_id} name={d.person_name} /></td><td className="num">{pct(d.value)}</td>
                </tr>))}</tbody>
            </table>
          ) : (
            <ResponsiveContainer width="100%" height={380}>
              <ComposedChart margin={{ top: 16, right: 24, bottom: 8, left: 0 }}>
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis type="number" dataKey="x" domain={[-0.6, horizons.length - 0.4]} ticks={horizons.map((_, i) => i)}
                  tickFormatter={(i: number) => hLabel(horizons[i] ?? '')} tickLine={false} axisLine={{ stroke: 'var(--axis)' }}
                  tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
                <YAxis type="number" dataKey="value" domain={[yTicks[0], yTicks[yTicks.length - 1]]} ticks={yTicks} tickFormatter={(v: number) => `${v}%`} tickLine={false}
                  axisLine={false} width={48} tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
                <Tooltip cursor={false} shared={false} content={() => {
                  const d = hover;
                  if (!d) return null;
                  if (!('participant_number' in d)) return null;
                  // the median bar ignores the pointer (it covers dots), so the tooltip carries the median instead
                  const median = data.stats.find((st) => st.horizon === d.horizon)?.median;
                  return <div className="tip">#{d.participant_number} {d.person_name ?? 'unmapped'} · {hLabel(d.horizon)}: <strong>{pct(d.value)}</strong>
                    {median !== undefined && <span className="muted"> (median {pct(median)})</span>}</div>;
                }} />
                <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }} />
                <Scatter name="Projections (one dot per participant)" data={points.filter((p) => !p.hl)} fill="var(--series-1)" fillOpacity={0.75}
                  stroke="var(--surface-1)" strokeWidth={1.5} isAnimationActive={false} {...hoverProps} />
                {hlName && (
                  <Scatter name={hlName} data={points.filter((p) => p.hl)} fill="var(--series-2)" stroke="var(--surface-1)" strokeWidth={2}
                    isAnimationActive={false} {...hoverProps}
                    shape={(p: { cx?: number; cy?: number }) => (p.cx === undefined || p.cy === undefined ? <g /> :
                      <circle cx={p.cx} cy={p.cy} r={6.5} fill="var(--series-2)" stroke="var(--surface-1)" strokeWidth={2} />)} />
                )}
                <Scatter name="Median" data={medians} isAnimationActive={false} legendType="line" fill="var(--text-primary)"
                  shape={(p: { cx?: number; cy?: number }) => (p.cx === undefined || p.cy === undefined ? <g /> :
                    <rect x={p.cx - 22} y={p.cy - 1.5} width={44} height={3} rx={1.5} fill="var(--text-primary)" pointerEvents="none" />)} />
              </ComposedChart>
            </ResponsiveContainer>
          )}
          <h3 className="chart-title">Dispersion by horizon</h3>
          <table className="data">
            <thead><tr><th>Horizon</th><th className="num">Participants</th><th className="num">Median</th><th className="num">Low – high</th>
              <th className="num">IQR</th><th>Outliers</th></tr></thead>
            <tbody>{data.stats.map((s) => (
              <tr key={s.horizon}>
                <td>{hLabel(s.horizon)}</td><td className="num">{s.n}</td><td className="num">{pct(s.median)}</td>
                <td className="num">{pct(s.min)} – {pct(s.max)}</td><td className="num">{pct(s.iqr)}</td>
                <td className="small">{s.outliers.length ? s.outliers.map((o) => `#${o.participant_number} ${o.person_name ?? 'unmapped'} (${pct(o.value)})`).join('; ') : '—'}</td>
              </tr>))}</tbody>
          </table>
          <p className="muted small">Outlier rule: {data.outlierRule}. A smaller IQR means more agreement. The median matches PostgreSQL percentile_cont.</p>
          <SqlBox sql={data.sql} />
        </>
      )}
    </section>
  );
}

const OFFSETS = [['0', 'End of current year'], ['1', 'End of next year'], ['2', 'End of year after next'], ['lr', 'Longer run']] as const;

function PersonTrack({ personId, offset, onPerson, onOffset }: {
  personId: string; offset: string; onPerson: (v: string) => void; onOffset: (v: string) => void;
}) {
  const persons = useApi<{ items: Row[] }>('/dotplot/persons');
  const list = persons.data?.items ?? [];
  const fallback = list.find((p) => /powell/i.test(String(p.person_name))) ?? list[0];
  const pid = personId || String(fallback?.person_id ?? '');
  const { data, error, loading } = useApi<{ items: TrackRow[]; sql: string }>(pid ? `/dotplot/person/${pid}` : null);
  const [table, setTable] = useState(false);
  const rows = (data?.items ?? []).filter((r) => (offset === 'lr' ? r.is_long_run : !r.is_long_run && String(r.year_offset) === offset));
  const name = String(list.find((p) => String(p.person_id) === pid)?.person_name ?? '');
  const diffs = rows.map((r) => r.value - r.median);
  const avg = diffs.length ? diffs.reduce((a, b) => a + b, 0) / diffs.length : 0;
  const above = diffs.filter((d) => d > 0).length, below = diffs.filter((d) => d < 0).length;

  return (
    <section className="viz-root">
      <h2>Track one participant</h2>
      <div className="filters">
        <label>Participant <select value={pid} onChange={(e) => onPerson(e.target.value)}>
          {list.map((p) => <option key={String(p.person_id)} value={String(p.person_id)}>{String(p.person_name)} ({String(p.meetings)} meetings)</option>)}
        </select></label>
        <label>Horizon <select value={offset} onChange={(e) => onOffset(e.target.value)}>
          {OFFSETS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select></label>
        <button className="ghost" onClick={() => setTable(!table)}>{table ? 'Chart' : 'Table'}</button>
        {pid && <Link to={`/participants/${pid}`}>Person page →</Link>}
      </div>
      {(persons.error || error) && <ErrorBox error={(persons.error ?? error)!} />}
      {loading && !data && <Loading />}
      {data && !rows.length && <Empty>No rate projections for this participant at this horizon.</Empty>}
      {rows.length > 0 && (
        <>
          <p className="small">
            {rows.length} meetings: above the median in {above}, below in {below}, equal in {rows.length - above - below};
            average distance from the median <strong>{avg >= 0 ? '+' : ''}{avg.toFixed(2)}</strong> percentage points (positive = more hawkish, negative = more dovish).
          </p>
          {table ? (
            <table className="data narrow">
              <thead><tr><th>Meeting</th><th>Horizon</th><th className="num">{name}</th><th className="num">Committee median</th><th className="num">Difference</th></tr></thead>
              <tbody>{rows.map((r) => (
                <tr key={r.batch_id}><td><Link to={`/dotplot?batch=${r.batch_id}&hl=${pid}`}>{r.meeting_start}</Link></td><td>{hLabel(r.horizon)}</td>
                  <td className="num">{pct(r.value)}</td><td className="num">{pct(r.median)}</td>
                  <td className="num">{(r.value - r.median >= 0 ? '+' : '') + (r.value - r.median).toFixed(2)}</td></tr>))}</tbody>
            </table>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={rows} margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis dataKey="meeting_start" tickLine={false} axisLine={{ stroke: 'var(--axis)' }} minTickGap={24}
                  tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
                <YAxis tickFormatter={(v: number) => `${v}%`} tickLine={false} axisLine={false} width={48} domain={[0, 'auto']}
                  tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
                <Tooltip formatter={(v, n) => [pct(Number(v)), n]} labelFormatter={(l) => `Meeting ${l}`} />
                <Legend verticalAlign="top" height={28} wrapperStyle={{ fontSize: 12 }} />
                <Line name={name} dataKey="value" stroke="var(--series-2)" strokeWidth={2.5} dot={{ r: 4 }} isAnimationActive={false} />
                <Line name="Committee median (dashed)" dataKey="median" stroke="var(--series-1)" strokeWidth={2} strokeDasharray="6 4"
                  dot={{ r: 2.5 }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
          <p className="muted small">Horizons are aligned by projection year minus meeting year (current year = 0); the longer run is shown separately. Identities come from the SEP respondent mapping.</p>
          <SqlBox sql={data!.sql} />
        </>
      )}
    </section>
  );
}

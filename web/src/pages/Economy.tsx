import { Fragment } from 'react';
import { Link } from 'react-router-dom';
import { useApi, type Row } from '../api';
import { Empty, ErrorBox, Loading, SqlBox } from '../components';

interface Citation { page: number | null; quote: string; url: string | null }
interface Item { topic: string | null; section: string | null; scope: string | null; sourceKind: string | null; claim: string; citations: Citation[] }
interface NumberRow { variable: string; value: string; unit: string | null; period: string | null; basis: string | null; scope: string | null; sourceKind: string | null; citations: Citation[] }
interface Context {
  cutoffDate: string; status: string; warnings: string[]; missingTypes: string[];
  sources: { kind: string | null; title: string | null; documentDate: string | null; ageDays: number | null; url: string | null }[];
  macro: Item[]; macroNumbers: NumberRow[]; policy: Item[]; policyNumbers: NumberRow[];
  districts: { district: string; status: string | null; items: Item[]; numbers: NumberRow[] }[];
}
interface EconomyData { meetingDate: string; indicators: Row | null; indicatorsNote: string | null; context: Context | null; contextNote: string | null; sql: string }

const KIND: Record<string, string> = {
  beige_book: 'Beige Book', greenbook: 'Greenbook', tealbook_a: 'Tealbook A', tealbook_b: 'Tealbook B',
};
const KIND_NOTE: Record<string, string> = {
  beige_book: 'regional conditions reported by the 12 Reserve Banks (public)',
  greenbook: 'staff forecast for the Committee (before mid-2010)',
  tealbook_a: 'staff economic outlook (from mid-2010)',
  tealbook_b: 'staff policy options',
};
const SECTIONS: [string, string][] = [['current_conditions', 'Current conditions'], ['changes', 'Changes since the last meeting'], ['outlook', 'Outlook'], ['risks', 'Risks']];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const month = (d: unknown) => { const s = String(d ?? ''); return /^\d{4}-\d{2}/.test(s) ? `${MONTHS[Number(s.slice(5, 7)) - 1]} ${s.slice(0, 4)}` : '—'; };
const quarter = (d: unknown) => { const s = String(d ?? ''); return /^\d{4}-\d{2}/.test(s) ? `${s.slice(0, 4)} Q${Math.floor((Number(s.slice(5, 7)) - 1) / 3) + 1}` : '—'; };
const pct = (v: unknown, digits = 1) => (v === null || v === undefined || v === '' ? '—' : `${Number(v).toFixed(digits)}%`);
const range = (lo: unknown, hi: unknown) => (lo === null || lo === undefined || lo === '' ? '—'
  : Number(lo) === Number(hi) ? `${Number(lo).toFixed(2)}%` : `${Number(lo).toFixed(2)}–${Number(hi).toFixed(2)}%`);

const INDICATOR_NOTE = 'Each value is what had been published by the day before the meeting (ALFRED vintages: the first-published figures, '
  + 'not today’s revised data). Inflation and GDP growth are changes from a year earlier; the fed funds rate is the monthly average effective rate.';
const SUMMARY_NOTE = 'AI-generated summary of the documents the Committee had before the meeting, produced by the research pipeline. '
  + 'Every point links to the page it cites in the original document — check the original before quoting.';

function Cites({ c }: { c: Citation[] }) {
  // one link per page; the tooltip shows every quote from that page
  const pages = new Map<string, Citation>();
  for (const x of c) {
    const k = String(x.page);
    const had = pages.get(k);
    pages.set(k, had ? { ...had, quote: `${had.quote}\n…\n${x.quote}` } : x);
  }
  return <>{[...pages.values()].map((x, i) => (
    <Fragment key={i}>{i > 0 && ' '}{x.url
      ? <a href={x.url} target="_blank" rel="noreferrer" title={x.quote}>p. {x.page ?? '?'} ↗</a>
      : <span className="muted" title={x.quote}>p. {x.page ?? '?'}</span>}</Fragment>))}</>;
}

function Items({ items, showSource = true }: { items: Item[]; showSource?: boolean }) {
  if (!items.length) return <Empty>None.</Empty>;
  return (
    <table className="data">
      <thead><tr><th>Topic</th><th>Summary</th>{showSource && <th>Source</th>}<th>Pages</th></tr></thead>
      <tbody>{items.map((it, i) => (
        <tr key={i}>
          <td className="small">{it.topic ?? '—'}{it.scope && it.scope !== 'United States' ? <div className="muted">{it.scope}</div> : null}</td>
          <td className="answer">{it.claim}</td>
          {showSource && <td className="small">{KIND[it.sourceKind ?? ''] ?? it.sourceKind ?? '—'}</td>}
          <td className="small"><Cites c={it.citations} /></td>
        </tr>))}</tbody>
    </table>
  );
}

function Numbers({ rows }: { rows: NumberRow[] }) {
  if (!rows.length) return null;
  return (
    <table className="data">
      <thead><tr><th>Variable</th><th className="num">Value</th><th>Unit</th><th>Period</th><th>Basis</th><th>Source</th><th>Pages</th></tr></thead>
      <tbody>{rows.map((n, i) => (
        <tr key={i}>
          <td className="small">{n.variable}{n.scope && n.scope !== 'United States' ? <div className="muted">{n.scope}</div> : null}</td>
          <td className="num">{n.value}</td><td className="small">{n.unit ?? '—'}</td><td className="small">{n.period ?? '—'}</td>
          <td className="small">{n.basis ?? '—'}</td><td className="small">{KIND[n.sourceKind ?? ''] ?? n.sourceKind ?? '—'}</td>
          <td className="small"><Cites c={n.citations} /></td>
        </tr>))}</tbody>
    </table>
  );
}

function Indicators({ x }: { x: Row }) {
  const cards: [string, string, string][] = [
    ['Unemployment rate', pct(x.unrate), month(x.unrate_period)],
    ['PCE inflation', pct(x.inflation), month(x.pce_period)],
    ['Core PCE inflation', pct(x.core_inflation), month(x.core_pce_period)],
    ['Real GDP growth', pct(x.gdp_growth), quarter(x.gdp_period)],
    ['Fed funds rate', pct(x.fedfunds, 2), month(x.fedfunds_period)],
    ['Target range', range(x.target_lower, x.target_upper), 'set at the previous meeting'],
  ];
  return (
    <div className="cards">
      {cards.map(([label, value, period]) => (
        <div className="card" key={label}><div className="card-label">{label}</div>
          <div className="card-value">{value}</div><div className="card-def">{period}</div></div>))}
    </div>
  );
}

/** "Economy before this meeting" block on the meeting page. */
export function EconomySection({ meetingId }: { meetingId: string }) {
  const { data, error, loading } = useApi<EconomyData>(`/meetings/${meetingId}/economy`);
  if (error) return <section><h2>Economy before this meeting</h2><ErrorBox error={error} /></section>;
  if (loading && !data) return <section><h2>Economy before this meeting</h2><Loading what="economic data" /></section>;
  const d = data!;
  const c = d.context;
  return (
    <section>
      <h2>Economy before this meeting</h2>
      {d.indicators ? <>
        <Indicators x={d.indicators} />
        <p className="muted small">As published on {String(d.indicators.vintage_date).slice(0, 10)}. {INDICATOR_NOTE}</p>
      </> : <p className="muted small">{d.indicatorsNote}</p>}

      {c ? <>
        <h3 className="chart-title">Documents available before the meeting</h3>
        <table className="data">
          <thead><tr><th>Document</th><th>What it is</th><th>Dated</th><th className="num">Days before meeting</th><th>Original</th></tr></thead>
          <tbody>{c.sources.map((s, i) => (
            <tr key={i}>
              <td>{KIND[s.kind ?? ''] ?? s.kind}</td><td className="small muted">{KIND_NOTE[s.kind ?? ''] ?? ''}</td>
              <td>{s.documentDate ?? '—'}</td><td className="num">{s.ageDays ?? '—'}</td>
              <td>{s.url ? <a href={s.url} target="_blank" rel="noreferrer">federalreserve.gov ↗</a> : '—'}</td>
            </tr>))}</tbody>
        </table>
        {c.warnings.map((w) => <div className="notice small" key={w}>{w}</div>)}
        <p className="muted small">{SUMMARY_NOTE}</p>

        {SECTIONS.map(([key, label]) => {
          const items = c.macro.filter((it) => it.section === key);
          return items.length ? <Fragment key={key}><h3 className="chart-title">{label}</h3><Items items={items} /></Fragment> : null;
        })}
        {(() => {
          const other = c.macro.filter((it) => !SECTIONS.some(([k]) => k === it.section));
          return other.length ? <><h3 className="chart-title">Other</h3><Items items={other} /></> : null;
        })()}
        {c.macroNumbers.length > 0 && <details className="sql"><summary>Figures quoted in the documents ({c.macroNumbers.length})</summary><Numbers rows={c.macroNumbers} /></details>}

        {(c.policy.length > 0 || c.policyNumbers.length > 0) && <>
          <h3 className="chart-title">Policy background (staff)</h3>
          <Items items={c.policy} />
          {c.policyNumbers.length > 0 && <details className="sql"><summary>Policy figures ({c.policyNumbers.length})</summary><Numbers rows={c.policyNumbers} /></details>}
        </>}

        {c.districts.length > 0 && <>
          <h3 className="chart-title">Beige Book by Federal Reserve District</h3>
          {c.districts.map((dist) => (
            <details className="sql" key={dist.district}>
              <summary>{dist.district} ({dist.items.length} points{dist.status && dist.status !== 'available' ? `, ${dist.status}` : ''})</summary>
              <Items items={dist.items} showSource={false} />
              <Numbers rows={dist.numbers} />
            </details>))}
        </>}
      </> : <p className="muted small">{d.contextNote}</p>}
      <SqlBox sql={d.sql} />
    </section>
  );
}

/** All meetings with their pre-meeting indicators. */
export default function Economy() {
  const { data, error, loading } = useApi<{ items: Row[]; sql: string }>('/economy');
  if (error) return <ErrorBox error={error} />;
  if (loading && !data) return <Loading />;
  const rows = data!.items;
  return (
    <>
      <h1>Economy</h1>
      <p className="muted small">Economic indicators the Committee could see before each meeting. {INDICATOR_NOTE} Click a meeting for the
        Beige Book and staff (Greenbook / Tealbook) summaries.</p>
      {rows.length ? (
        <table className="data">
          <thead><tr><th>Meeting</th><th className="num">Unemployment</th><th className="num">PCE inflation</th><th className="num">Core PCE</th>
            <th className="num">GDP growth</th><th className="num">Fed funds</th><th className="num">Target range</th><th>SEP</th><th>Data as of</th></tr></thead>
          <tbody>{rows.map((x) => (
            <tr key={String(x.meeting_date)}>
              <td>{x.corpus_meeting_id ? <Link to={`/meetings/${x.corpus_meeting_id}`}>{String(x.meeting_date).slice(0, 10)}</Link> : String(x.meeting_date).slice(0, 10)}</td>
              <td className="num" title={month(x.unrate_period)}>{pct(x.unrate)}</td>
              <td className="num" title={month(x.pce_period)}>{pct(x.inflation)}</td>
              <td className="num" title={month(x.core_pce_period)}>{pct(x.core_inflation)}</td>
              <td className="num" title={quarter(x.gdp_period)}>{pct(x.gdp_growth)}</td>
              <td className="num" title={month(x.fedfunds_period)}>{pct(x.fedfunds, 2)}</td>
              <td className="num">{range(x.target_lower, x.target_upper)}</td>
              <td>{x.is_sep_meeting === true || x.is_sep_meeting === 'true' ? 'Yes' : ''}</td>
              <td className="small muted">{String(x.vintage_date).slice(0, 10)}</td>
            </tr>))}</tbody>
        </table>
      ) : <Empty>No indicators.</Empty>}
      <p className="muted small">Meetings after 2020 have no page yet (their transcripts are not public). Hover a value to see the month or quarter it refers to.</p>
      <SqlBox sql={data!.sql} />
    </>
  );
}

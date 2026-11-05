import { useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useApi, type Row } from '../api';
import { Empty, ErrorBox, Highlight, Loading, MeetingLink, Pager, PersonLink, RoleBadge, SqlBox } from '../components';

interface Result {
  items: Row[]; hasMore: boolean; total: number; totalCapped: boolean; page: number; pageSize: number;
  sql: string; sourceColumns: string[]; table: string; roles: string[]; order: string; searchMode: 'fulltext' | 'substring';
}

const ROLE_LABEL: Record<string, string> = { Voting: 'Voting members', Official: 'All FOMC officials' };
const ordinal = (k: number) => `${k}${['th', 'st', 'nd', 'rd'][(k % 100 > 10 && k % 100 < 14) || k % 10 > 3 ? 0 : k % 10]}`;

export default function Transcript() {
  const [sp, setSp] = useSearchParams();
  const get = (k: string) => sp.get(k) ?? '';
  const q = get('q'), personId = get('personId'), meetingId = get('meetingId'), year = get('year');
  const dateFrom = get('dateFrom'), dateTo = get('dateTo'), role = get('role');
  const order = ['asc', 'desc', 'relevance'].includes(get('order')) ? get('order') : '';
  const page = Number(sp.get('page') ?? 1);
  const [input, setInput] = useState(q);
  const filters = { meetingId, year, dateFrom, dateTo, role };
  const { data, error, loading } = useApi<Result>('/transcript', { q, personId, ...filters, order, page, pageSize: 20 });
  const years = useApi<number[]>('/meetings/years');
  const dates = useApi<string[]>('/meetings/dates');
  const speakers = useApi<{ items: Row[] }>('/transcript/speakers', filters);
  const speakerItems = speakers.data?.items ?? [];
  const update = (patch: Record<string, string>) => {
    const n = new URLSearchParams(sp);
    for (const [k, v] of Object.entries(patch)) if (v) n.set(k, v); else n.delete(k);
    if (!('page' in patch)) n.delete('page');
    setSp(n);
  };
  const any = q || personId || meetingId || year || dateFrom || dateTo || role || Boolean(order);
  return (
    <>
      <h1>Transcript</h1>
      <form className="filters" onSubmit={(e) => { e.preventDefault(); update({ q: input.trim() }); }}>
        <label>Year <select value={year} onChange={(e) => update({ year: e.target.value })}>
          <option value="">All</option>
          {[...(years.data ?? [])].map((y) => <option key={y} value={y}>{y}</option>)}
        </select></label>
        {/* meeting dates instead of <input type="date">: the browser's calendar follows the OS language */}
        <label>From <select value={dateFrom} onChange={(e) => update({ dateFrom: e.target.value })}>
          <option value="">Any</option>
          {[...(dates.data ?? [])].reverse().map((d) => <option key={d} value={d}>{d}</option>)}
        </select></label>
        <label>To <select value={dateTo} onChange={(e) => update({ dateTo: e.target.value })}>
          <option value="">Any</option>
          {(dates.data ?? []).filter((d) => !dateFrom || d >= dateFrom).map((d) => <option key={d} value={d}>{d}</option>)}
        </select></label>
        {(data?.roles.length ?? 0) > 0 && (
          <label>Role <select value={role} onChange={(e) => update({ role: e.target.value })}>
            <option value="">All</option>
            {data!.roles.map((r) => <option key={r} value={r}>{ROLE_LABEL[r] ?? r}</option>)}
          </select></label>)}
        <label>Speaker <select value={personId} onChange={(e) => update({ personId: e.target.value })}>
          <option value="">All</option>
          {personId && !speakerItems.some((p) => p.person_id === personId) && (
            <option value={personId}>{speakers.loading ? 'Loading…' : '(no turns under these filters)'}</option>)}
          {speakerItems.map((p) => (
            <option key={String(p.person_id)} value={String(p.person_id)}>{String(p.person_name ?? p.person_id)} ({Number(p.turns).toLocaleString()})</option>))}
        </select></label>
        <input placeholder='Keywords, e.g. inflation wages or "balance sheet"' value={input} onChange={(e) => setInput(e.target.value)} maxLength={100} />
        <button type="submit">Search</button>
        <label>Order <select value={data?.order ?? order} onChange={(e) => update({ order: e.target.value })}>
          {q && data?.searchMode === 'fulltext' && <option value="relevance">Most relevant first</option>}
          <option value="desc">Newest meetings first</option>
          <option value="asc">Oldest meetings first</option>
        </select></label>
        {meetingId && <span className="chip">One meeting <button type="button" className="x" onClick={() => update({ meetingId: '' })}>×</button></span>}
        {any && <button type="button" className="ghost" onClick={() => { setInput(''); setSp(new URLSearchParams()); }}>Clear filters</button>}
      </form>
      {data?.searchMode === 'fulltext' && <p className="muted small">Keyword search finds turns containing all the words in any order, including other forms
        (“rate” also finds “rates”). Use “quotes” for an exact phrase, <code>or</code> for either word, and <code>-word</code> to exclude a word.</p>}
      <p className="muted small">From / To are meeting start dates (inclusive). Turns within a meeting stay in spoken order. 20 per page; counts stop at 5,000.
        “Turn #96 of 312” is the position in the whole meeting and its total number of turns; “8th of 12” is this speaker’s own count in that meeting.</p>
      {error && <ErrorBox error={error} />}
      {loading && !data && <Loading />}
      {data && (
        <>
          <p>{data.totalCapped ? '5,000+' : data.total.toLocaleString()} matching turns{loading && <span className="muted"> · updating…</span>}</p>
          <ol className="utterances">
            {data.items.map((u) => (
              <li key={String(u.utterance_id)}>
                <div className="u-meta">
                  <MeetingLink id={u.meeting_id} date={u.meeting_date} /> · Turn #{String(u.seq ?? '—')}{u.meeting_turns ? ` of ${String(u.meeting_turns)}` : ''} ·{' '}
                  <PersonLink id={u.person_id} name={u.person_name} />
                  {u.role ? <> <RoleBadge role={u.role} /></> : null}
                  {u.speaker_turn ? <span className="muted"> · {ordinal(Number(u.speaker_turn))} of {String(u.speaker_turns)} turns by this speaker</span> : null}
                  {u.speaker_label ? <span className="muted"> · as “{String(u.speaker_label)}”</span> : null}
                  {u.source_page ? <span className="muted"> · page {String(u.source_page)}</span> : null}
                  {u.pdf_url ? <> · <a href={String(u.pdf_url)} target="_blank" rel="noreferrer">Original PDF ↗</a></> : null}
                </div>
                <div className="u-text"><Highlight text={String(u.text)} term={q} words={data.searchMode === 'fulltext'} /></div>
              </li>
            ))}
          </ol>
          {!data.items.length && <Empty>No matching turns.</Empty>}
          <Pager page={data.page} pageSize={data.pageSize} hasMore={data.hasMore}
            total={data.totalCapped ? undefined : data.total} onPage={(p) => update({ page: String(p) })} />
          <SqlBox sql={data.sql} />
        </>
      )}
      {error?.unavailable && (
        <p className="small">To set it up, configure the read-only connection, run <code>npm run introspect</code>, then fill in
          <code>server/transcript-mapping.json</code>. See <Link to="/quality">Data Quality</Link>.</p>
      )}
    </>
  );
}

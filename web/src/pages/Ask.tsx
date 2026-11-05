import { Fragment, useState } from 'react';
import { ApiError, useApi, type Row } from '../api';
import { ErrorBox, MeetingLink, PersonLink, SqlBox } from '../components';

interface Source extends Row { n: number; cited: boolean; content: string; pdf_url: string | null }
interface AskResult { question: string; answer: string; autoDates: { dateFrom: string; dateTo: string } | null; sources: Source[]; embedModel: string; chatModel: string; sql: string }

const EXAMPLES = [
  'Why did Thomas Hoenig dissent in 2010?',
  'How did participants discuss the risks of the balance sheet in 2013?',
  '2020 年疫情期间，委员们对劳动力市场的看法是什么？',
];

const NOTE = 'Answers are written by an AI model from the transcript passages listed under them, and only from those passages. '
  + 'Every [n] links to its passage; check the passage and the original PDF before quoting.';

/** The answer text with each [n] turned into a link to source n. */
function AnswerText({ text, sources }: { text: string; sources: Source[] }) {
  const parts = text.split(/(\[\d+\])/g);
  return <div className="ask-answer">{parts.map((p, i) => {
    const m = /^\[(\d+)\]$/.exec(p);
    const n = m ? Number(m[1]) : 0;
    return m && sources.some((s) => s.n === n)
      ? <a key={i} href={`#source-${n}`} className="cite">[{n}]</a>
      : <Fragment key={i}>{p}</Fragment>;
  })}</div>;
}

export default function Ask() {
  const [question, setQuestion] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [state, setState] = useState<{ data?: AskResult; error?: ApiError; loading: boolean }>({ loading: false });
  const dates = useApi<string[]>('/meetings/dates');

  async function ask(q: string) {
    if (!q.trim()) return;
    setState({ loading: true });
    try {
      const res = await fetch('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, dateFrom, dateTo }) });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new ApiError(body.error ?? `HTTP ${res.status}`, res.status, Boolean(body.unavailable));
      setState({ data: body.data as AskResult, loading: false });
    } catch (e) {
      setState({ error: e instanceof ApiError ? e : new ApiError(String(e), 0), loading: false });
    }
  }

  const d = state.data;
  return (
    <>
      <h1>Ask the transcripts</h1>
      <p className="muted small">Ask a question in English or Chinese. The closest passages from the FOMC transcripts (2007–2020) are found by
        meaning, not only by keyword, and the answer is written from them with citations. A year named in the question limits the search
        to that year unless From / To are chosen. {NOTE}</p>
      <form className="ask-form" onSubmit={(e) => { e.preventDefault(); void ask(question); }}>
        <textarea value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={500} rows={3}
          placeholder="e.g. Why did Thomas Hoenig dissent in 2010?"
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void ask(question); } }} />
        <div className="filters">
          <label>From <select value={dateFrom} onChange={(e) => setDateFrom(e.target.value)}>
            <option value="">Any</option>
            {[...(dates.data ?? [])].reverse().map((x) => <option key={x} value={x}>{x}</option>)}
          </select></label>
          <label>To <select value={dateTo} onChange={(e) => setDateTo(e.target.value)}>
            <option value="">Any</option>
            {(dates.data ?? []).filter((x) => !dateFrom || x >= dateFrom).map((x) => <option key={x} value={x}>{x}</option>)}
          </select></label>
          <button type="submit" disabled={state.loading || !question.trim()}>{state.loading ? 'Answering…' : 'Ask'}</button>
        </div>
      </form>
      {!d && !state.loading && !state.error && (
        <p className="small">Examples: {EXAMPLES.map((x, i) => (
          <Fragment key={x}>{i > 0 && ' · '}<button type="button" className="linkish" onClick={() => { setQuestion(x); void ask(x); }}>{x}</button></Fragment>))}
        </p>)}
      {state.loading && <p className="muted">Finding passages and writing the answer (about 5–15 seconds)…</p>}
      {state.error && <ErrorBox error={state.error} />}
      {d && <>
        <h2 className="chart-title">Answer</h2>
        {d.autoDates && <p className="notice small">The question names {d.autoDates.dateFrom.slice(0, 4) === d.autoDates.dateTo.slice(0, 4)
          ? `the year ${d.autoDates.dateFrom.slice(0, 4)}` : `the years ${d.autoDates.dateFrom.slice(0, 4)}–${d.autoDates.dateTo.slice(0, 4)}`}, so
          only meetings in {d.autoDates.dateFrom.slice(0, 4) === d.autoDates.dateTo.slice(0, 4) ? 'that year' : 'those years'} were searched.
          Choose From / To to search other dates.</p>}
        <AnswerText text={d.answer} sources={d.sources} />
        <h2 className="chart-title">Sources ({d.sources.length} passages, closest first)</h2>
        <table className="data">
          <thead><tr><th>#</th><th>Speaker</th><th>Meeting</th><th className="num">Turn</th><th>Original</th><th>Passage</th></tr></thead>
          <tbody>{d.sources.map((s) => (
            <tr key={s.n} id={`source-${s.n}`} className={s.cited ? undefined : 'muted'}>
              <td>[{s.n}]{!s.cited && <div className="small">not cited</div>}</td>
              <td className="small"><PersonLink id={s.person_id} name={s.person_name ?? s.speaker_label} /></td>
              <td className="small"><MeetingLink id={s.meeting_id} date={s.meeting_date} /></td>
              <td className="num small">#{String(s.seq ?? '—')}</td>
              <td className="small">{s.pdf_url ? <a href={s.pdf_url} target="_blank" rel="noreferrer">PDF p. {String(s.source_page)} ↗</a> : '—'}</td>
              <td className="answer small">{s.content.length > 600
                ? <details><summary>{s.content.slice(0, 600)}…</summary>{s.content.slice(600)}</details> : s.content}</td>
            </tr>))}</tbody>
        </table>
        <p className="muted small">Passages: {d.embedModel} vectors, cosine distance, best passage per turn. Answer: {d.chatModel}.</p>
        <SqlBox sql={d.sql} title="Retrieval SQL" />
      </>}
    </>
  );
}

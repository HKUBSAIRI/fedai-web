import { useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { useApi, type Row } from '../api';
import { Empty, ErrorBox, Loading, MeetingLink, Pager, ROLE_NOTE, RoleBadge, RowTable, SqlBox, Val } from '../components';

interface PersonList { items: Row[]; total: number; page: number; pageSize: number; sql: string }

const KINDS = [['official', 'FOMC officials'], ['staff', 'Staff and others'], ['all', 'Everyone']] as const;

export function Persons() {
  const [sp, setSp] = useSearchParams();
  const q = sp.get('q') ?? '';
  const kind = sp.get('kind') ?? 'official';
  const page = Number(sp.get('page') ?? 1);
  const [input, setInput] = useState(q);
  const { data, error, loading } = useApi<PersonList>('/persons', { q, kind, page });
  const set = (patch: Record<string, string>) => {
    const p = new URLSearchParams(sp);
    for (const [k, v] of Object.entries(patch)) if (v) p.set(k, v); else p.delete(k);
    if (!('page' in patch)) p.delete('page');
    setSp(p);
  };
  const hasRoles = data?.items.some((p) => p.official_meetings !== null && p.official_meetings !== undefined) ?? true;
  return (
    <>
      <h1>Participants</h1>
      <form className="filters" onSubmit={(e) => { e.preventDefault(); set({ q: input.trim() }); }}>
        <div className="seg">{KINDS.map(([k, label]) => (
          <button key={k} type="button" className={kind === k ? 'on' : ''} onClick={() => set({ kind: k === 'official' ? '' : k })}>{label}</button>))}
        </div>
        <input placeholder="Search by name, e.g. Powell" value={input} onChange={(e) => setInput(e.target.value)} maxLength={100} />
        <button type="submit">Search</button>
      </form>
      <p className="muted small">FOMC officials are Board governors and Reserve Bank presidents (or their alternates) — the people the project models.
        Everyone else who attended (secretaries, economists, other staff) is listed under Staff.</p>
      {error && <ErrorBox error={error} />}
      {loading && !data && <Loading />}
      {data && (
        <>
          <table className="data">
            <thead><tr><th>Person</th>
              {hasRoles && <><th className="num">Meetings as official</th><th className="num">of which voting</th></>}
              <th className="num">Meetings attended</th><th className="num">SEP rounds</th><th className="num">Transcript turns</th></tr></thead>
            <tbody>{data.items.map((p) => (
              <tr key={String(p.person_id)}>
                <td><Link to={`/participants/${p.person_id}`}>{String(p.person_name)}</Link></td>
                {hasRoles && <><td className="num">{String(p.official_meetings)}</td><td className="num">{String(p.voter_meetings)}</td></>}
                <td className="num"><Val v={p.meeting_count} /></td>
                <td className="num"><Val v={p.sep_count} /></td>
                <td className="num"><Val v={p.utterance_count} /></td>
              </tr>))}</tbody>
          </table>
          {!data.items.length && <Empty>No matching people.</Empty>}
          <Pager page={data.page} pageSize={data.pageSize} total={data.total} onPage={(p) => set({ page: String(p) })} />
          <SqlBox sql={data.sql} />
        </>
      )}
    </>
  );
}

interface PersonData { person: Row; personName: string | null; meetings: Row[]; sep: Row[]; sameNameOtherIds: number; transcriptByPerson: boolean }

export function PersonDetail() {
  const { id } = useParams();
  const { data, error, loading } = useApi<PersonData>(`/persons/${id}`);
  if (error) return <ErrorBox error={error} />;
  if (loading && !data) return <Loading />;
  const d = data!;
  const hasRoles = d.meetings.some((m) => 'role' in m);
  const byRole = new Map<string, number>();
  for (const m of d.meetings) byRole.set(String(m.role), (byRole.get(String(m.role)) ?? 0) + 1);
  const totalTurns = d.meetings.reduce((a, m) => a + (Number(m.turns) || 0), 0);
  return (
    <>
      <p><Link to="/participants">← Participants</Link></p>
      <h1>{String(d.personName ?? d.person.person_id)}</h1>
      {d.sameNameOtherIds > 0 && <div className="notice">{d.sameNameOtherIds} other person_id(s) have the same name (case-insensitive) and are not merged.</div>}
      <div className="cards">
        <div className="card"><div className="card-label">Meetings attended</div><div className="card-value">{d.meetings.length}</div>
          {hasRoles && <div className="card-def">{[...byRole].map(([r, k]) => `${r} ${k}`).join(' · ')}</div>}</div>
        <div className="card"><div className="card-label">Transcript turns</div><div className="card-value">{hasRoles ? totalTurns : '—'}</div>
          {d.transcriptByPerson && <Link className="small" to={`/transcript?personId=${id}`}>Read all turns →</Link>}</div>
        <div className="card"><div className="card-label">SEP rounds</div><div className="card-value">{d.sep.length}</div>
          {d.sep.length > 0 && <Link className="small" to={`/dotplot?person=${id}`}>Rate projections over time →</Link>}</div>
      </div>

      <section>
        <h2>Meetings ({d.meetings.length})</h2>
        {d.meetings.length ? (hasRoles ? (
          <table className="data">
            <thead><tr><th>Meeting</th><th>Role</th><th>Affiliation</th><th className="num">Turns</th></tr></thead>
            <tbody>{[...d.meetings].reverse().map((m) => (
              <tr key={String(m.meeting_id)}>
                <td><MeetingLink id={m.meeting_id} date={m.meeting_date} /></td>
                <td><RoleBadge role={m.role} inferred={m.role_inferred} /></td>
                <td><Val v={m.affiliation} /></td>
                <td className="num">{m.turns ? <Link to={`/transcript?meetingId=${m.meeting_id}&personId=${id}&order=asc`}>{String(m.turns)}</Link> : <span className="muted">0</span>}</td>
              </tr>))}</tbody>
          </table>
        ) : <ul className="chips">{d.meetings.map((m) => <li key={String(m.meeting_id)}><MeetingLink id={m.meeting_id} date={m.meeting_date} /></li>)}</ul>)
          : <Empty>This person is not on the attendance list of any meeting.</Empty>}
        {hasRoles && <p className="muted small">{ROLE_NOTE}</p>}
      </section>

      <section>
        <h2>SEP rounds ({d.sep.length})</h2>
        <p className="muted small">The respondent number is anonymous and only valid within one SEP round.</p>
        {d.sep.length ? (
          <table className="data">
            <thead><tr><th>SEP (meeting start)</th><th className="num">Respondent #</th><th>Name in the source</th><th className="num">Answers (answered / all)</th></tr></thead>
            <tbody>{[...d.sep].reverse().map((s) => (
              <tr key={String(s.batch_id)}>
                <td><Link to={`/sep/${s.batch_id}`}>{String(s.meeting_start)}</Link></td>
                <td className="num">{String(s.participant_number)}</td>
                <td>{String(s.source_name)}</td>
                <td className="num">{String(s.answered)} / {String(s.answers)}</td>
              </tr>))}</tbody>
          </table>
        ) : <Empty>No SEP records (only FOMC officials submit SEPs).</Empty>}
      </section>
      <details className="sql"><summary>Database record (persons)</summary><RowTable row={d.person} /></details>
    </>
  );
}

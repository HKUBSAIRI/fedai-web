import { Link, useParams, useSearchParams } from 'react-router-dom';
import { EconomySection } from './Economy';
import { useApi, type Row } from '../api';
import { Empty, ErrorBox, Loading, MeetingLink, Pager, PersonLink, ROLE_NOTE, RoleBadge, RowTable, SqlBox, Val } from '../components';

interface MeetingList { items: Row[]; total: number; page: number; pageSize: number; sql: string }

const n = (v: unknown) => (v === null || v === undefined ? <span className="null">—</span> : String(v));

export function Meetings() {
  const [sp, setSp] = useSearchParams();
  const year = sp.get('year') ?? '';
  const page = Number(sp.get('page') ?? 1);
  const years = useApi<number[]>('/meetings/years');
  const { data, error, loading } = useApi<MeetingList>('/meetings', { year, page, pageSize: 25 });
  const set = (k: string, v: string) => { const p = new URLSearchParams(sp); if (v) p.set(k, v); else p.delete(k); if (k !== 'page') p.delete('page'); setSp(p); };
  return (
    <>
      <h1>Meetings</h1>
      <div className="filters">
        <label>Year <select value={year} onChange={(e) => set('year', e.target.value)}>
          <option value="">All</option>
          {years.data?.map((y) => <option key={y} value={y}>{y}</option>)}
        </select></label>
        <span className="muted">Dates are the first and last day of the meeting.</span>
      </div>
      {error && <ErrorBox error={error} />}
      {loading && !data && <Loading />}
      {data && (
        <>
          <table className="data">
            <thead><tr><th>Start</th><th>End</th><th className="num">Voters</th><th className="num">Non-voting participants</th>
              <th className="num">Staff</th><th className="num">SEP</th><th className="num">Transcript turns</th></tr></thead>
            <tbody>
              {data.items.map((m) => (
                <tr key={String(m.meeting_id)}>
                  <td><MeetingLink id={m.meeting_id} date={m.meeting_date} /></td>
                  <td><Val v={m.meeting_end_date} /></td>
                  <td className="num">{n(m.voter_count)}</td>
                  <td className="num">{n(m.nonvoting_count)}</td>
                  <td className="num">{n(m.staff_count)}</td>
                  <td className="num">{m.sep_batch_count ? <strong>Yes</strong> : <span className="muted">—</span>}</td>
                  <td className="num">{n(m.utterance_count)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!data.items.length && <Empty>No meetings match this filter.</Empty>}
          <Pager page={data.page} pageSize={data.pageSize} total={data.total} onPage={(p) => set('page', String(p))} />
          <p className="muted small">{ROLE_NOTE}</p>
          <SqlBox sql={data.sql} />
        </>
      )}
    </>
  );
}

interface MeetingData {
  meeting: Row; participants: Row[]; participantsSql: string | null; sepBatches: Row[]; utteranceCount: number | null; sql: string;
}

const GROUPS = [
  ['Voting members', ['Chair', 'Vice Chair', 'Voter']],
  ['Non-voting participants', ['Participant']],
  ['Staff and other attendees', null],
] as const;

export function MeetingDetail() {
  const { id } = useParams();
  const { data, error, loading } = useApi<MeetingData>(`/meetings/${id}`);
  if (error) return <ErrorBox error={error} />;
  if (loading && !data) return <Loading />;
  const d = data!;
  const hasRoles = d.participants.some((p) => 'role' in p);
  const inGroup = (roles: readonly string[] | null) => (p: Row) => {
    const r = String(p.role);
    return roles ? roles.includes(r) : !['Chair', 'Vice Chair', 'Voter', 'Participant'].includes(r);
  };
  return (
    <>
      <p><Link to="/meetings">← Meetings</Link></p>
      <h1>Meeting {String(d.meeting.meeting_date).slice(0, 10)}{d.meeting.meeting_end_date && d.meeting.meeting_end_date !== d.meeting.meeting_date
        ? ` – ${String(d.meeting.meeting_end_date).slice(0, 10)}` : ''}</h1>
      {hasRoles && (
        <div className="cards">
          {GROUPS.map(([label, roles]) => (
            <div className="card" key={label}><div className="card-label">{label}</div>
              <div className="card-value">{d.participants.filter(inGroup(roles)).length}</div></div>))}
          <div className="card"><div className="card-label">Transcript turns</div>
            <div className="card-value">{d.utteranceCount ?? '—'}</div>
            {d.utteranceCount !== null && <Link className="small" to={`/transcript?meetingId=${id}&order=asc`}>Read this meeting →</Link>}</div>
          <div className="card"><div className="card-label">SEP</div>
            <div className="card-value">{d.sepBatches.length ? 'Yes' : 'No'}</div>
            {d.sepBatches.map((b) => <Link key={String(b.batch_id)} className="small" to={`/sep/${b.batch_id}`}>Open this SEP →</Link>)}</div>
        </div>
      )}

      {hasRoles ? GROUPS.map(([label, roles]) => {
        const rows = d.participants.filter(inGroup(roles));
        return (
          <section key={label}>
            <h2>{label} ({rows.length})</h2>
            {rows.length ? (
              <table className="data">
                <thead><tr><th>Person</th><th>Role</th><th>Affiliation</th><th className="num">Turns in this meeting</th></tr></thead>
                <tbody>{rows.map((p) => (
                  <tr key={String(p.person_id)}>
                    <td><PersonLink id={p.person_id} name={p.person_name} /></td>
                    <td><RoleBadge role={p.role} inferred={p.role_inferred} /></td>
                    <td><Val v={p.affiliation} /></td>
                    <td className="num">{p.turns ? <Link to={`/transcript?meetingId=${id}&personId=${p.person_id}&order=asc`}>{String(p.turns)}</Link> : <span className="muted">0</span>}</td>
                  </tr>))}</tbody>
              </table>
            ) : <Empty>None.</Empty>}
          </section>
        );
      }) : (
        <section>
          <h2>Attendees ({d.participants.length})</h2>
          {d.participants.length ? (
            <table className="data">
              <thead><tr><th>Person</th>{Object.keys(d.participants[0]).filter((k) => !['person_id', 'person_name'].includes(k)).map((k) => <th key={k}>{k}</th>)}</tr></thead>
              <tbody>{d.participants.map((p, i) => (
                <tr key={i}>
                  <td><PersonLink id={p.person_id} name={p.person_name} /></td>
                  {Object.entries(p).filter(([k]) => !['person_id', 'person_name'].includes(k)).map(([k, v]) => <td key={k} className="mono small"><Val v={v} /></td>)}
                </tr>))}</tbody>
            </table>
          ) : <Empty>No attendees recorded.</Empty>}
        </section>
      )}
      <p className="muted small">{ROLE_NOTE}</p>
      <SqlBox sql={d.participantsSql} title="Attendee SQL" />
      <EconomySection meetingId={String(id)} />
      <details className="sql"><summary>Database record (meetings)</summary><RowTable row={d.meeting} /><pre>{d.sql}</pre></details>
    </>
  );
}

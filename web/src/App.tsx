import { NavLink, Route, Routes } from 'react-router-dom';
import { useApi } from './api';
import Dashboard from './pages/Dashboard';
import { MeetingDetail, Meetings } from './pages/Meetings';
import { PersonDetail, Persons } from './pages/Persons';
import Transcript from './pages/Transcript';
import { Sep, SepDetail } from './pages/Sep';
import Quality from './pages/Quality';
import DotPlot from './pages/DotPlot';
import Economy from './pages/Economy';
import Ask from './pages/Ask';

export interface StatusData {
  mode: 'postgres' | 'demo';
  demo: boolean;
  schema: string;
  role: { database: string; user: string; serverVersion: string; ok: boolean; problems: string[] } | null;
  realDb: { configured: boolean; connected: boolean; error: string | null };
  availability: { transcript: { ok: boolean; reason?: string }; projections: { ok: boolean; reason?: string } };
  tables: string[];
}

const NAV = [
  ['/', 'Dashboard'], ['/meetings', 'Meetings'], ['/participants', 'Participants'],
  ['/transcript', 'Transcript'], ['/ask', 'Ask AI'], ['/economy', 'Economy'], ['/sep', 'SEP'], ['/dotplot', 'Dot plot'], ['/quality', 'Data Quality'],
] as const;

export default function App() {
  const { data: status } = useApi<StatusData>('/status');
  return (
    <div className="app">
      <header className="top">
        <div className="brand">FedAI Data Platform <small>internal research tool</small></div>
        <nav>
          {NAV.map(([to, label]) => <NavLink key={to} to={to} end={to === '/'}>{label}</NavLink>)}
        </nav>
      </header>
      {status?.demo && (
        <div className="demo-banner" role="alert">
          <strong>DEMO DATA</strong> — not connected to the FedAI database. Every meeting, person, turn, SEP and number on these pages is fictitious,
          for testing only; <u>do not cite as research results</u>. Reason: {status.realDb.error ?? '—'}
        </div>
      )}
      <main>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/meetings" element={<Meetings />} />
          <Route path="/meetings/:id" element={<MeetingDetail />} />
          <Route path="/participants" element={<Persons />} />
          <Route path="/participants/:id" element={<PersonDetail />} />
          <Route path="/transcript" element={<Transcript />} />
          <Route path="/ask" element={<Ask />} />
          <Route path="/economy" element={<Economy />} />
          <Route path="/sep" element={<Sep />} />
          <Route path="/sep/:id" element={<SepDetail />} />
          <Route path="/dotplot" element={<DotPlot />} />
          <Route path="/quality" element={<Quality status={status} />} />
          <Route path="*" element={<p>Page not found.</p>} />
        </Routes>
      </main>
      <footer>Read-only: every query is a SELECT · each number comes with its SQL so it can be checked</footer>
    </div>
  );
}

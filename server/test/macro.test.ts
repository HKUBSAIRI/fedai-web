import { describe, expect, it } from 'vitest';
import { officialMacroUrl, shapeContext } from '../src/repo.js';

const P = 'data/FOMC Meetings/Documents';
const FED = 'https://www.federalreserve.gov';

describe('officialMacroUrl', () => {
  it('links Greenbook and Tealbook files by name, with an optional page', () => {
    expect(officialMacroUrl(`${P}/Greenbook/2007/FOMC20070131gbpt120070124.pdf`)).toBe(`${FED}/monetarypolicy/files/FOMC20070131gbpt120070124.pdf`);
    expect(officialMacroUrl(`${P}/Tealbook/2020/FOMC20200429tealbooka20200417.pdf`, '2020-04-17', 12))
      .toBe(`${FED}/monetarypolicy/files/FOMC20200429tealbooka20200417.pdf#page=12`);
  });
  it('follows the Beige Book folders over the years', () => {
    expect(officialMacroUrl(`${P}/Beige Book/2008/fullreport20080305.pdf`)).toBe(`${FED}/fomc/beigebook/2008/20080305/fullreport20080305.pdf`);
    expect(officialMacroUrl(`${P}/Beige Book/2012/fullreport20120411.pdf`)).toBe(`${FED}/monetarypolicy/beigebook/files/fullreport20120411.pdf`);
    expect(officialMacroUrl(`${P}/Beige Book/2016/Beigebook_20160413.pdf`)).toBe(`${FED}/monetarypolicy/beigebook/files/Beigebook_20160413.pdf`);
    expect(officialMacroUrl(`${P}/Beige Book/2017/Beigebook_20170906.pdf`)).toBe(`${FED}/monetarypolicy/files/Beigebook_20170906.pdf`);
    expect(officialMacroUrl(`${P}/Beige Book/2014/BeigeBook_20140416.pdf`)).toBe(`${FED}/monetarypolicy/beigebook/files/BeigeBook_20140416.pdf`);
    expect(officialMacroUrl(`${P}/Beige Book/2020/BeigeBook_20200415.pdf`)).toBe(`${FED}/monetarypolicy/files/BeigeBook_20200415.pdf`);
    expect(officialMacroUrl(`${P}/Beige Book/2007/default.htm`, new Date('2007-11-28T00:00:00Z'))).toBe(`${FED}/fomc/beigebook/2007/20071128/default.htm`);
  });
  it('returns null for unknown names', () => {
    expect(officialMacroUrl(`${P}/Beige Book/2007/default.htm`)).toBeNull();
    expect(officialMacroUrl('x/y/other.pdf')).toBeNull();
  });
});

describe('shapeContext', () => {
  it('links each citation to its page and keeps the sections', () => {
    const sid = '11111111-1111-1111-1111-111111111111';
    const out = shapeContext({
      source_ids: [sid],
      coverage: { source_details: [{ kind: 'tealbook_a', source_id: sid, document_date: '2020-04-17', age_days_at_event: 11 }] },
      macro: { items: [{ claim: 'GDP falls', topic: 'Outlook', section: 'outlook', source_id: sid, source_kind: 'tealbook_a', citations: [{ page: 3, quote: 'q' }] }] },
    }, [{ source_id: sid, kind: 'tealbook_a', title: 'Tealbook A', source_path: `${P}/Tealbook/2020/FOMC20200429tealbooka20200417.pdf`, document_date: '2020-04-17' }],
    '2020-04-28', 'partial_sources');
    expect(out.sources[0]).toMatchObject({ kind: 'tealbook_a', ageDays: 11, url: `${FED}/monetarypolicy/files/FOMC20200429tealbooka20200417.pdf` });
    expect(out.macro[0].citations[0].url).toBe(`${FED}/monetarypolicy/files/FOMC20200429tealbooka20200417.pdf#page=3`);
    expect(out.districts).toEqual([]);
  });
});

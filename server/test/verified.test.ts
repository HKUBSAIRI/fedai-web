// Verified data-quality hits must match exactly; anything new stays unverified.
import { describe, expect, it } from 'vitest';
import { VERIFIED_ROWS } from '../src/verified.js';

describe('verified data-quality hits', () => {
  it('SEP grid gap: only 2009-01-27 with exactly 8 missing answers (Tarullo)', () => {
    const m = VERIFIED_ROWS.sep_grid_gap.match;
    expect(m({ meeting_start: '2009-01-27', expected: '136', actual: '128' })).toBe(true);
    expect(m({ meeting_start: '2009-01-27', expected: '136', actual: '127' })).toBe(false);
    expect(m({ meeting_start: '2010-01-26', expected: '136', actual: '128' })).toBe(false);
  });

  it('chair / vice chair gap: only the two 2008 meetings with one chair and no vice chair', () => {
    const m = VERIFIED_ROWS.meeting_role_gap.match;
    expect(m({ meeting_date: '2008-09-16', chairs: '1', vice_chairs: '0', voters: '9' })).toBe(true);
    expect(m({ meeting_date: '2008-12-15', chairs: '1', vice_chairs: '0', voters: '9' })).toBe(true);
    expect(m({ meeting_date: '2008-12-15', chairs: '0', vice_chairs: '0', voters: '9' })).toBe(false);
    expect(m({ meeting_date: '2009-06-23', chairs: '1', vice_chairs: '0', voters: '8' })).toBe(false);
  });

  it('same-surname groups: verified only with the checked group size', () => {
    const m = VERIFIED_ROWS.person_same_surname.match;
    expect(m({ surname: 'smith', ids: '4', names: '' })).toBe(true);
    expect(m({ surname: 'smith', ids: '5', names: '' })).toBe(false);
    expect(m({ surname: 'yellen', ids: '2', names: '' })).toBe(false);
  });
});

// Hits of data-quality checks that have been checked by hand against the sources and are correct as stored.
// A hit counts as verified only if it matches an entry here exactly; anything new stays "to check",
// so this list can never hide a new problem. Evidence: docs/sepfix/SEP全量核对报告.md and the notes below.
import type { Row } from './db/queryer.js';

export interface VerifiedRule {
  /** True if this hit row is one of the verified cases. */
  match: (r: Row) => boolean;
  /** One line shown on the Data Quality page. */
  note: string;
}

/** Hits of these checks are judged one row at a time (all rows are fetched; these checks return few rows). */
export const VERIFIED_ROWS: Record<string, VerifiedRule> = {
  sep_grid_gap: {
    // Tarullo was sworn in on the last day of the meeting: numbered in the Key, but no projections or answers in the source.
    match: (r) => r.meeting_start === '2009-01-27' && Number(r.expected) - Number(r.actual) === 8,
    note: '2009-01-27: respondent 17 (Tarullo) joined the Board on the last day of the meeting; all 8 of his answers are blank in the source, so having no rows is correct.',
  },
  meeting_role_gap: {
    // Vice Chairman Geithner was absent; First VP Christine Cumming voted as New York's alternate.
    match: (r) => ['2008-09-16', '2008-12-15'].includes(String(r.meeting_date))
      && Number(r.chairs) === 1 && Number(r.vice_chairs) === 0,
    note: '2008-09-16 and 2008-12-15: Vice Chair Geithner was absent (Lehman crisis / nomination as Treasury Secretary); New York Fed First Vice President Cumming voted as alternate, so no Vice Chair attended.',
  },
  person_same_surname: {
    // Each group checked: different people (mostly staff); "ii" is a name suffix, not a surname.
    match: (r) => VERIFIED_SURNAME_GROUPS.get(String(r.surname)) === Number(r.ids),
    note: 'Checked group by group: different people who share a surname (mostly staff); the "ii" group is a name suffix (Sangster II, Knotek II), not one person.',
  },
};

/** Surname → number of distinct people with that surname, as checked on 2026-10-23. */
const VERIFIED_SURNAME_GROUPS = new Map<string, number>([
  ['smith', 4], ['ii', 2], ['anderson', 2], ['meyer', 3], ['li', 2], ['aaronson', 2], ['bowman', 2], ['kim', 2],
  ['nelson', 2], ['mertens', 2], ['campbell', 2], ['williams', 2], ['robertson', 2], ['fisher', 2], ['goldberg', 2],
  ['lewis', 2], ['raskin', 2], ['feldman', 2], ['clark', 2],
]);

/**
 * SEP missing-field hits are verified by reason: every listed reason was confirmed against the source tables
 * (SEP fix steps 01 and 13). Rows with any other reason stay unverified.
 */
export const VERIFIED_MISSING_REASONS = {
  reasons: ['not_collected_before_2012', 'not_collected_for_long_run', 'blank_in_source_table'],
  note: 'Every missing value has a checked reason: the rate was not collected before 2012, longer-run projections have no core PCE, or the respondent left the cell blank. Correct as stored; do not fill in.',
};

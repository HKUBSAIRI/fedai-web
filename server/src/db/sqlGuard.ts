// Defence-in-depth guard: every statement the app sends must be a single SELECT/WITH query.
// The primary protections are the read-only DB role and read-only sessions (see queryer.ts);
// this guard catches programming mistakes before they reach the database.

const FORBIDDEN = [
  'insert', 'update', 'delete', 'merge', 'create', 'alter', 'drop', 'truncate',
  'grant', 'revoke', 'copy', 'call', 'do', 'vacuum', 'analyze', 'reindex', 'cluster',
  'lock', 'refresh', 'set', 'reset', 'comment', 'security', 'execute', 'prepare',
  'listen', 'notify', 'discard', 'import', 'into',
];

/** Remove comments and the contents of string literals / quoted identifiers. */
export function stripLiterals(sql: string): string {
  let out = '';
  let i = 0;
  while (i < sql.length) {
    const c = sql[i];
    const n = sql[i + 1];
    if (c === '-' && n === '-') {
      while (i < sql.length && sql[i] !== '\n') i++;
      out += ' ';
    } else if (c === '/' && n === '*') {
      const end = sql.indexOf('*/', i + 2);
      i = end === -1 ? sql.length : end + 2;
      out += ' ';
    } else if (c === "'" || c === '"') {
      const q = c;
      i++;
      while (i < sql.length) {
        if (sql[i] === q && sql[i + 1] === q) { i += 2; continue; }
        if (sql[i] === q) break;
        i++;
      }
      i++;
      out += q === "'" ? "''" : '"x"';
    } else if (c === '$' && /[A-Za-z_$]/.test(n ?? '') && sql.slice(i).match(/^\$[A-Za-z_]*\$/)) {
      throw new SqlGuardError('dollar-quoted strings are not allowed');
    } else {
      out += c;
      i++;
    }
  }
  return out;
}

export class SqlGuardError extends Error {}

export function assertSelectOnly(sql: string): void {
  const body = stripLiterals(sql).trim().replace(/;\s*$/, '');
  if (body.includes(';')) throw new SqlGuardError('multiple statements are not allowed');
  const lower = body.toLowerCase();
  if (!/^(select|with)\b/.test(lower)) throw new SqlGuardError('only SELECT / WITH queries are allowed');
  for (const word of FORBIDDEN) {
    if (new RegExp(`\\b${word}\\b`).test(lower)) {
      throw new SqlGuardError(`forbidden keyword in query: ${word.toUpperCase()}`);
    }
  }
}

/** Quote an identifier that has already been validated against the discovered schema. */
export function quoteIdent(name: string): string {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) throw new SqlGuardError(`unsafe identifier: ${name}`);
  return `"${name}"`;
}

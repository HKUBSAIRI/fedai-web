import { describe, expect, it } from 'vitest';
import { assertSelectOnly, quoteIdent, SqlGuardError } from '../src/db/sqlGuard.js';
import { loadConfig } from '../src/config.js';
import { sanitizeDbError } from '../src/app.js';
import { clampPage, escapeLike } from '../src/repo.js';

describe('assertSelectOnly', () => {
  it.each([
    'SELECT 1',
    'select * from fedai_corpus.meetings where meeting_id::text = $1',
    'WITH x AS (SELECT 1 AS a) SELECT a FROM x;',
    "SELECT 'please delete; drop table x' AS s",
    'SELECT "update" FROM t',
    'SELECT 1 -- delete from x\n',
    'SELECT p.offset_days, p.reset_flag FROM t p',
  ])('allows %s', (sql) => expect(() => assertSelectOnly(sql)).not.toThrow());

  it.each([
    'DELETE FROM fedai_corpus.persons',
    'UPDATE fedai_corpus.persons SET canonical_name = $1',
    'INSERT INTO t VALUES (1)',
    'DROP TABLE fedai_corpus.meetings',
    'TRUNCATE fedai_corpus.sep_answers',
    'CREATE TABLE x (a int)',
    'ALTER TABLE x ADD COLUMN b int',
    'SELECT 1; DELETE FROM t',
    'WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d',
    'SELECT * INTO new_table FROM t',
    'SELECT 1 /* x */ ; DROP TABLE t',
    'COPY t TO STDOUT',
    'GRANT SELECT ON t TO public',
    'SET ROLE postgres',
    'SELECT $$x$$',
    '  vacuum t',
  ])('rejects %s', (sql) => expect(() => assertSelectOnly(sql)).toThrow(SqlGuardError));
});

describe('helpers', () => {
  it('quoteIdent rejects unsafe identifiers', () => {
    expect(quoteIdent('sep_answers')).toBe('"sep_answers"');
    expect(() => quoteIdent('a"; drop')).toThrow();
  });
  it('escapeLike escapes wildcards', () => expect(escapeLike('50%_a\\')).toBe('50\\%\\_a\\\\'));
  it('clampPage bounds page size', () => {
    expect(clampPage('0', '100000')).toEqual({ page: 1, pageSize: 100, offset: 0 });
    expect(clampPage('3', '10')).toEqual({ page: 3, pageSize: 10, offset: 20 });
  });
  it('refuses to bind to non-loopback hosts', () => {
    expect(() => loadConfig({ FEDAI_HOST: '0.0.0.0' })).toThrow(/127\.0\.0\.1/);
    expect(loadConfig({}).host).toBe('127.0.0.1');
  });
  it('sanitizes host / user details from DB errors', () => {
    const msg = sanitizeDbError({ code: '28P01', message: 'password authentication failed for user "alice" at 10.1.2.3:5432' });
    expect(msg).not.toContain('alice');
    expect(msg).not.toContain('10.1.2.3');
  });
});

import { resolveSsl } from '../src/db/queryer.js';
describe('resolveSsl', () => {
  it('never silently downgrades to plaintext', () => {
    expect(resolveSsl({ PGSSLMODE: 'prefer' }).transport).toBe('tls-unverified');
    expect(resolveSsl({}).transport).toBe('tls-unverified');
    expect(() => resolveSsl({ PGSSLMODE: 'disable' })).toThrow(/FEDAI_ALLOW_PLAINTEXT_DB/);
    expect(resolveSsl({ PGSSLMODE: 'disable', FEDAI_ALLOW_PLAINTEXT_DB: 'yes' }).transport).toBe('plaintext-acknowledged');
    expect(() => resolveSsl({ PGSSLMODE: 'verify-full' })).toThrow(/PGSSLROOTCERT/);
  });
});

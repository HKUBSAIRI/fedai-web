import type { Queryer } from './queryer.js';

export interface RoleCheck {
  database: string;
  user: string;
  serverVersion: string;
  sessionReadOnly: boolean;
  superuser: boolean;
  writableTables: number;
  canCreateInSchema: boolean;
  ok: boolean;
  problems: string[];
}

/**
 * Verifies the connected role cannot modify the business schema. The app refuses to serve
 * real data when this fails, even though every query also runs in a READ ONLY transaction.
 */
export async function verifyReadOnlyRole(q: Queryer, schema: string): Promise<RoleCheck> {
  const [r] = await q.query<Record<string, unknown>>(
    `SELECT current_database() AS database, current_user AS usr,
            current_setting('server_version') AS server_version,
            current_setting('transaction_read_only') AS ro,
            COALESCE((SELECT rolsuper FROM pg_catalog.pg_roles WHERE rolname = current_user), false) AS superuser,
            (SELECT COUNT(*) FROM information_schema.tables t
              WHERE t.table_schema = $1
                AND (has_table_privilege(quote_ident(t.table_schema) || '.' || quote_ident(t.table_name), 'INSERT')
                  OR has_table_privilege(quote_ident(t.table_schema) || '.' || quote_ident(t.table_name), 'UPDATE')
                  OR has_table_privilege(quote_ident(t.table_schema) || '.' || quote_ident(t.table_name), 'DELETE')
                  OR has_table_privilege(quote_ident(t.table_schema) || '.' || quote_ident(t.table_name), 'TRUNCATE'))
            ) AS writable_tables,
            COALESCE((SELECT has_schema_privilege(n.oid, 'CREATE') FROM pg_catalog.pg_namespace n WHERE n.nspname = $1), false) AS can_create`,
    [schema],
  );
  const problems: string[] = [];
  const superuser = Boolean(r.superuser);
  const writableTables = Number(r.writable_tables);
  const canCreate = Boolean(r.can_create);
  if (superuser) problems.push('the database role is a superuser');
  if (writableTables > 0) problems.push(`the role can INSERT/UPDATE/DELETE/TRUNCATE ${writableTables} tables in ${schema}`);
  if (canCreate) problems.push(`the role can CREATE in schema ${schema}`);
  return {
    database: String(r.database),
    user: String(r.usr),
    serverVersion: String(r.server_version),
    sessionReadOnly: r.ro === 'on',
    superuser,
    writableTables,
    canCreateInSchema: canCreate,
    ok: problems.length === 0,
    problems,
  };
}

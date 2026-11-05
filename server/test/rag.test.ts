import { describe, expect, it } from 'vitest';
import type { Queryer, Row } from '../src/db/queryer.js';
import { assertSelectOnly } from '../src/db/sqlGuard.js';
import type { SchemaInfo, TranscriptMapping } from '../src/db/schema.js';
import { buildUserPrompt, chunkTurn, citedNumbers, embeddingInput, loadRagConfig, modelLabel, toVectorLiteral, yearsInQuestion } from '../src/rag.js';
import { BadRequestError, Repo } from '../src/repo.js';
import mapping from '../transcript-mapping.json' with { type: 'json' };

const words = (n: number) => Array.from({ length: n }, (_, i) => `w${i}`).join(' ');

describe('chunkTurn', () => {
  it('skips short turns and keeps a medium turn whole', () => {
    expect(chunkTurn('Thank you. President Evans.')).toEqual([]);
    expect(chunkTurn(words(100))).toEqual([words(100)]);
  });

  it('splits long turns into overlapping passages that cover every word', () => {
    const parts = chunkTurn(words(1000), 375, 40);
    expect(parts.map((p) => p.split(' ').length)).toEqual([375, 375, 330]);
    expect(parts[1].split(' ')[0]).toBe('w335');                 // starts 40 words before the end of passage 0
    expect(parts.at(-1)!.split(' ').at(-1)).toBe('w999');
  });

  it('normalises whitespace', () => {
    expect(chunkTurn(`  ${words(20).replace(/ /g, '\n  ')}  `)).toEqual([words(20)]);
  });
});

describe('prompt and citations', () => {
  it('puts speaker and date in front of the embedded passage', () => {
    expect(embeddingInput('Jerome H. Powell', '2020-12-15', 'Welcome.')).toBe('Jerome H. Powell — FOMC meeting 2020-12-15\nWelcome.');
  });

  it('numbers the excerpts with speaker, date, turn and page', () => {
    const p = buildUserPrompt('Why?', [{ n: 1, speaker: 'Thomas M. Hoenig', meetingDate: '2010-04-27', seq: 96, page: 140, content: 'I dissent.' }]);
    expect(p).toContain('[1] Thomas M. Hoenig, FOMC meeting 2010-04-27, turn #96, transcript p. 140\nI dissent.');
    expect(p.endsWith('Question: Why?')).toBe(true);
  });

  it('limits the search to the years a question names', () => {
    expect(yearsInQuestion('Why did Hoenig dissent in 2010?')).toEqual({ dateFrom: '2010-01-01', dateTo: '2010-12-31' });
    expect(yearsInQuestion('2020年疫情期间')).toEqual({ dateFrom: '2020-01-01', dateTo: '2020-12-31' });
    expect(yearsInQuestion('between 2012 and 2008')).toEqual({ dateFrom: '2008-01-01', dateTo: '2012-12-31' });
    expect(yearsInQuestion('Was the 1994 tightening mentioned?')).toBeNull();
    expect(yearsInQuestion('What about 20100 or 2010s?')).toBeNull();
    expect(yearsInQuestion('Why did Hoenig dissent?')).toBeNull();
  });

  it('lists cited numbers in order of first use', () => {
    expect(citedNumbers('A [2][1]. B [2]. C [10].')).toEqual([2, 1, 10]);
    expect(citedNumbers('No citations.')).toEqual([]);
  });

  it('reads Azure settings only when endpoint and key are set', () => {
    expect(loadRagConfig({})).toBeNull();
    const c = loadRagConfig({ AZURE_OPENAI_ENDPOINT: 'https://x.openai.azure.com/', AZURE_OPENAI_API_KEY: 'k' })!;
    expect(c.endpoint).toBe('https://x.openai.azure.com');
    expect(modelLabel(c)).toBe('text-embedding-3-large@1536');
    expect(toVectorLiteral([0.5, -1])).toBe('[0.5,-1]');
    expect(toVectorLiteral([-0.012345678912345678, 0.1 + 0.2])).toBe('[-0.01234568,0.3]');
  });
});

describe('ragSearch SQL', () => {
  const info: SchemaInfo = new Map([
    ['meetings', ['meeting_id', 'meeting_date', 'active_version_id']],
    ['persons', ['person_id', 'display_name']],
    ['transcript_turns', ['turn_id', 'version_id', 'body_clean', 'person_id', 'speaker_raw', 'sequence_number', 'source_page', 'document_id', 'cleaner_version']],
    ['documents', ['document_id', 'source_path']],
    ['rag_chunks', ['turn_id', 'chunk_index', 'version_id', 'meeting_date', 'person_id', 'content', 'text_hash', 'model', 'embedding', 'created_at']],
  ]);
  const row = (turn: string, chunk: number, distance: number): Row => ({
    turn_id: turn, chunk_index: chunk, content: `${turn}/${chunk}`, distance, meeting_id: 'm', meeting_date: '2010-04-27',
    person_id: 'p', person_name: 'Thomas M. Hoenig', speaker_label: 'MR. HOENIG.', seq: 96,
    source_file: 'FOMC20100428meeting.pdf', source_page: 140,
  });
  const fake = (rows: Row[]) => {
    const seen: { sql: string; params: unknown[] }[] = [];
    const q: Queryer = {
      async query<T extends Row>(sql: string, params: unknown[] = []) { assertSelectOnly(sql); seen.push({ sql, params }); return rows as T[]; },
      async close() { /* nothing */ },
    };
    return { q, seen };
  };

  it('is available only when rag_chunks is readable', () => {
    expect(new Repo(fake([]).q, 'fedai_corpus', info, mapping as TranscriptMapping).avail.rag).toBe(true);
    const without = new Map(info); without.delete('rag_chunks');
    expect(new Repo(fake([]).q, 'fedai_corpus', without, mapping as TranscriptMapping).avail.rag).toBe(false);
  });

  it('is a single guarded SELECT using the index, keeps the closest passage per turn and links the PDF page', async () => {
    const { q, seen } = fake([row('a', 0, 0.1), row('a', 1, 0.2), row('b', 0, 0.3)]);
    const r = await new Repo(q, 'fedai_corpus', info, mapping as TranscriptMapping).ragSearch('[1,2]', 'text-embedding-3-large@1536', {});
    expect(r.items.map((x) => x.content)).toEqual(['a/0', 'b/0']);
    expect(r.items[0].pdf_url).toBe('https://www.federalreserve.gov/monetarypolicy/files/FOMC20100428meeting.pdf#page=140');
    expect(seen[0].sql).toContain('ORDER BY c.embedding <=> $1::vector');
    expect(seen[0].sql).toContain('md5(tr."body_clean") = h.text_hash');
    expect(seen[0].params).toEqual(['[1,2]', 'text-embedding-3-large@1536']);
    expect(r.sql).not.toContain('$1::vector');
  });

  it('ranks exactly within a date filter and validates dates', async () => {
    const { q, seen } = fake([]);
    const repo = new Repo(q, 'fedai_corpus', info, mapping as TranscriptMapping);
    await repo.ragSearch('[1]', 'm', { dateFrom: '2010-01-01', dateTo: '2010-12-31' });
    expect(seen[0].sql).toContain('ORDER BY (c.embedding <=> $1::vector) + 0');
    expect(seen[0].params).toEqual(['[1]', 'm', '2010-01-01', '2010-12-31']);
    await expect(repo.ragSearch('[1]', 'm', { dateFrom: '2010; DROP' })).rejects.toBeInstanceOf(BadRequestError);
  });
});

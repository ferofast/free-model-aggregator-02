// Reading the brief: the free model list and the prompt list.
//
// A trial hands this entry a brief file (the path in MAKEFUN_PROMPT_FILE). The brief is prose
// wrapped around two JSON documents, and only the documents matter — the prose is never parsed.
// Each document is found by the `schema` member it carries, under this contest's prefix:
//
//   - the model list (`free-model-aggregator-02/models/v1`): one row per model, keyed by the id
//     the gateway routes (`openrouter/<vendor>/<model>:free`), carrying the model's public rating
//     in milli-points, its capabilities (tools, vision, JSON mode, context length) and its free
//     flag. Only free models may serve a prompt; a paid model is never routed, whatever the brief
//     says about it.
//   - the prompt list (`free-model-aggregator-02/prompts/v1`): the six prompts the house's client
//     will send, in order, each with its id, its wire shape (`openai` or `anthropic`), whether it
//     streams, its weight (1 or 2) and the capabilities a model must have to serve it. The client
//     sends the bodies; this file routes each arrival from its row, which is why the prompt list
//     is read before the first request is served.
//
// A brief may name the lists `models` or `ratings` and `prompts` or `manifest` (the sibling -01
// contest's names), and a single document may carry both lists; a document is found by the arrays
// it carries, never by a name alone.
//
// Nothing here reads the clock or rolls dice, so `node main.ts --list` and the register body build
// the same document, byte for byte, on a live pass and on its replay.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** The schema prefix every document in this contest's brief carries; the brief's JSON is found by it. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The public base URL of the provider the free models belong to: the endpoint the gateway reaches. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable an entrant's OpenRouter key is pushed under (an entry secret, kept out of the sandbox). */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface ModelRow {
  id: string;
  /** The provider the model belongs to, as the brief names it (`openrouter`). */
  provider: string;
  display_name: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as this baseline reads it. `error` is set when a document could not be read. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** What a model can do, from the brief's row (null: the brief did not say). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** A string field, or '' when the value is not a string. */
function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

/** A finite number, or null when the value is not one. */
function asFiniteNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** A boolean, or null when the value is not one. */
function asBoolean(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * Every JSON object in `text` that carries a string `schema`, in the order they appear in the
 * text. A `{` that does not open a parseable object on its own is skipped and the scan moves on.
 */
function findSchemaDocuments(text: string): Json[] {
  const found: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = matchingCloseBrace(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return found;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function matchingCloseBrace(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/**
 * The first document under this contest's prefix that carries one of `keys` as an array: the
 * document and the key it was found under. A brief that spells its lists in the sibling's names
 * still reads.
 */
function firstDocumentWithArray(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!asString(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return { doc, key };
    }
  }
  return null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider that names it: a row
 * that gives the bare OpenRouter id (`vendor/model:free`) and names `openrouter` is routed as the
 * gateway spells it (`openrouter/vendor/model:free`). The brief lists free models, so a row is free
 * unless it says `free: false`; a paid row is never routed.
 */
function parseModelRow(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = asString(v.id) !== '' ? asString(v.id) : asString(v.model);
  if (raw === '') return null;
  const named = asString(v.provider) !== '' ? asString(v.provider) : asString(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: asString(v.display_name) !== '' ? asString(v.display_name) : raw,
    rating_milli: asFiniteNumber(v.rating_milli) ?? 0,
    context_length: asFiniteNumber(v.context_length),
    tools: asBoolean(v.tools),
    vision: asBoolean(v.vision),
    json: asBoolean(v.json),
    free: v.free !== false,
  };
}

/** One prompt row: only the fields the brief actually sets are kept. */
function parsePromptRow(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)
        ? { min_context: requires.min_context }
        : {}),
    },
  };
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = findSchemaDocuments(text);
  const modelsDoc = firstDocumentWithArray(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = parseModelRow(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = firstDocumentWithArray(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = parsePromptRow(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** What a model can do, as the brief's row says it. */
export function capabilitiesOf(model: ModelRow): Capabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) {
    return false;
  }
  return true;
}

/**
 * The catalog the brief's free model list makes: one provider, the models it serves.
 *
 * The base_url is the provider's public endpoint, never the gateway's trial-local address: the
 * gateway's origin differs between a live pass and its replay, and the catalog is part of the state
 * export a replay must reproduce byte for byte.
 *
 * This baseline lists every free model in the brief — exactly the fourteen dialable ids the catalog
 * column scores — so keep it that way: register this set and keep GET /v1/models (both auth
 * headers) exactly that set, since listing fewer loses the column as surely as listing more.
 */
export function buildCatalog(brief: Brief): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}

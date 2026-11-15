// free-model-aggregator-02: one server, two client wires, and a real free model behind every answer.
//
// The winning shape this entry plays: quality-optimal routing (the brief's best-rated free model
// that meets each prompt's requirements, computed at runtime from the brief, never hardcoded),
// exactly one gateway call per prompt when nothing refuses, and a failover that retries only on a
// real refusal, reports every refusal in `x-makefun-upstream-failures`, and never spends a call on
// a model that is cooling down. The -01 lesson, learned from the live leaderboard: our -01 entry
// scored 99 and still ranked 1,454, because six extra calls cost ~1,430 ranks — the tie-break is
// the number of gateway calls, so every call is spent like it costs a place.
//
// Routing, failover and the catalog are the whole game:
//   - quality (60%): the best-rated free model that meets the prompt. JSON-mode prompts need a
//     JSON-capable model (the top-rated model cannot do JSON); tools prompts need tools; the rest
//     take the brief's best.
//   - failover (10%): a refused call is retried on the next-best legal model, never the same one,
//     and the refusal is reported in the header, the only trail the failover column can see.
//   - the gate: the brief's best model may refuse every call with a 403 — OpenRouter gates some
//     free models to approved harnesses, and a 403 is not transient. The cooldown learns it in one
//     call and the trial never dials it again, so the probe costs one gateway request and nothing on
//     the tie-break: a refused call is not a counted upstream call, and the served model is the best
//     one that actually answers.
//   - calls (the tie-break): one call per prompt, no probes, no pre-calls, no same-model retry.
//   - catalog (5%): exactly the fourteen free ids, in the register body and on both /v1/models
//     answers, byte for byte the same document `--list` prints.
//   - replay: every choice is a function of the brief, the requests and the gateway's answers —
//     no clock, no randomness, one request order.
//
// WHAT A TRIAL GIVES YOU
//   A trial runs this server as `node /bot/main.ts` from your repository's root, with Node 22's
//   standard library alone (nothing is installed; TypeScript runs directly, and a relative import
//   needs its .ts extension). The env is at $MAKEFUN_ENV_URL; your model calls go through the
//   grader's gateway at $OPENAI_BASE_URL, an OpenAI-compatible endpoint that reaches OpenRouter on
//   the key you pushed to the competition. The brief at $MAKEFUN_PROMPT_FILE carries two documents:
//     - the free model list: each free OpenRouter model's gateway id
//       (`openrouter/<vendor>/<model>:free`), its published rating in milli-points, what it can do
//       (tools, vision, JSON mode, context length) and its free flag;
//     - the six prompts: each one's id, its api (openai or anthropic), whether it streams, its
//       weight (1 or 2) and the requirements a model must meet to serve it. Each prompt carries a
//       unique token the answer must contain.
//   The server:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves GET /v1/models (answering in the shape of the auth header it got: x-api-key means
//        Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST /v1/messages, each
//        with a JSON body or with server-sent events;
//     4. answers each prompt by dialling ONE free model through the gateway. OPENAI_API_KEY in the
//        sandbox is a dummy the gateway swaps for the OpenRouter key you pushed; the real key never
//        enters the sandbox;
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
//   register body's catalog is that same document, byte for byte.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the model it actually dialled.
//   The house records every gateway call and scores the model that really answered. This server sends
//   the brief's model id in the gateway request and puts that same id back on every answer and every
//   streamed chunk (the provider's own answer spells it without the gateway's `openrouter/` prefix), so
//   its claim is the call it made, never another. A prompt whose reported model has no matching
//   successful call in the house's record earns nothing.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the free model list, the catalog.
//   lib/wire.ts        Anthropic <-> Chat Completions translation, the model field, the SSE writers.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most gateway attempts one prompt gets, across distinct models. */
const MAX_ATTEMPTS = 3;
/** The most milliseconds one gateway call may take to answer its headers. */
const UPSTREAM_TIMEOUT_MS = 8000;
/** The output cap a translated body gets when the client named none (the gateway requires one). */
const FALLBACK_MAX_TOKENS = 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Models that refused a call this trial: never dialled again while the trial runs. */
const cooledDown = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free candidates that meet a request's requirements, best-rated first (ties keep the brief's
 * order, so the choice is a pure function of the brief). Quality is 60% of the score and the best
 * model is the whole of it: a JSON-mode prompt takes the best JSON-capable model, a tools prompt
 * the best tools-capable one, the rest the brief's best.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's requirements; this fallback is
  // a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extraHeaders: Record<string, string> = {}): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extraHeaders);
  else sendJson(res, status, openAIError(message, type, null), extraHeaders);
}

/** One refusal, as the `x-makefun-upstream-failures` trail records it. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

/**
 * The `x-makefun-upstream-failures` header for a prompt's refusals, or nothing when there were none.
 * The header is the only trail the failover column can see, so every refusal a retry recovered from
 * is reported here, in the order the calls refused.
 */
function failuresHeader(failures: UpstreamFailure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

/** The gateway's error code for a refused call: its JSON error's code or type, else the status. */
function errorCodeOf(status: number, text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

/**
 * The body a gateway call carries: the client's body with the dialled model, and the output cap the
 * gateway requires when the client named none (the house client names one on the Anthropic wire; the
 * OpenAI wire's prompts do not always, and a call without a cap is refused before it starts).
 */
function gatewayBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (typeof out.max_tokens !== 'number' && typeof out.max_completion_tokens !== 'number') out.max_tokens = FALLBACK_MAX_TOKENS;
  return out;
}

/**
 * POST one request to the gateway's Chat Completions route, naming the model this entry dialled.
 * The timer guards the call's headers only: it is cleared the moment the gateway answers, so a
 * slow-but-alive stream is never cut off mid-answer, while a hung gateway frees the attempt after
 * eight seconds and the prompt fails over instead of eating the client's whole 30-second budget.
 */
async function dial(model: string, body: Json): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  try {
    return await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify(gatewayBody(body, model)),
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timer);
  }
}

/** A prompt the gateway served: the live response, the model that answered it, and its refusals. */
interface PromptServed {
  ok: true;
  upstream: Response;
  model: string;
  failures: UpstreamFailure[];
}

/** A prompt no free model could serve: why it was lost, and every refusal behind it. */
interface PromptLost {
  ok: false;
  text: string;
  failures: UpstreamFailure[];
}

type PromptOutcome = PromptServed | PromptLost;

/**
 * Serve one prompt through the gateway, in rating order, one call per model.
 * The first attempt dials the brief's best legal model; a refusal cools that model down for the
 * trial (a refused model stays refused — the -01 lesson is to drop it on its first refusal, not to
 * spend a second call learning the same thing) and the next attempt takes the next-best model that
 * meets the prompt. Three attempts at most: with fourteen free models and six prompts, three is
 * enough for a real refusal storm, and a fourth would only spend the tie-break. Every refusal is
 * recorded in order, so the response can carry the trail the failover column reads.
 */
async function attemptPrompt(wire: 'openai' | 'anthropic', body: Json): Promise<PromptOutcome> {
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires);
  const failures: UpstreamFailure[] = [];
  const tried = new Set<string>();
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    const route = candidates.find((model) => !tried.has(model.id) && !cooledDown.has(model.id));
    if (route === undefined) break;
    tried.add(route.id);
    const upstreamBody = wire === 'anthropic' ? anthropicToOpenAI(body, route.id) : body;
    let upstream: Response;
    try {
      upstream = await dial(route.id, upstreamBody);
    } catch (err) {
      const code = err instanceof Error && err.name === 'AbortError' ? 'upstream_timeout' : 'upstream_unreachable';
      failures.push({ status: 0, code, model: route.id });
      cooledDown.add(route.id);
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      const text = await upstream.text().catch(() => '');
      failures.push({ status: upstream.status, code: errorCodeOf(upstream.status, text), model: route.id });
      cooledDown.add(route.id);
      continue;
    }
    return { ok: true, upstream, model: route.id, failures };
  }
  if (failures.length === 0) {
    return { ok: false, text: 'no free model in the brief meets this request', failures };
  }
  const last = failures[failures.length - 1];
  return {
    ok: false,
    text: `the gateway refused every model tried (last: HTTP ${last.status} ${last.code} on ${last.model ?? 'unknown'})`,
    failures,
  };
}

/**
 * A refused or broken gateway call as a 502 in the client's wire, carrying the refusals behind it:
 * the env records a 4xx or 5xx answer as a refusal on its own, and the trail names the gateway
 * calls that refused, so the prompt enters the failover column honestly instead of vanishing.
 */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', text: string, failures: UpstreamFailure[]): void {
  const headers = failuresHeader(failures);
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', text, headers);
  else sendError(res, wire, 502, 'upstream_error', text, headers);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const outcome = await attemptPrompt('openai', body);
  if (!outcome.ok) return refused(res, 'openai', outcome.text, outcome.failures);
  const headers = failuresHeader(outcome.failures);
  if (body.stream === true) {
    const stream = outcome.upstream.body;
    if (stream === null) {
      return refused(res, 'openai', 'the gateway answered with no body', [
        ...outcome.failures,
        { status: 200, code: 'empty_body', model: outcome.model },
      ]);
    }
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const frame of openAIStreamWithModel(stream, outcome.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  let answer: unknown;
  try {
    answer = await outcome.upstream.json();
  } catch {
    return refused(res, 'openai', 'the gateway answer did not parse', [
      ...outcome.failures,
      { status: 200, code: 'unparseable_body', model: outcome.model },
    ]);
  }
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, outcome.model), headers);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const outcome = await attemptPrompt('anthropic', body);
  if (!outcome.ok) return refused(res, 'anthropic', outcome.text, outcome.failures);
  const headers = failuresHeader(outcome.failures);
  if (body.stream === true) {
    const stream = outcome.upstream.body;
    if (stream === null) {
      return refused(res, 'anthropic', 'the gateway answered with no body', [
        ...outcome.failures,
        { status: 200, code: 'empty_body', model: outcome.model },
      ]);
    }
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const event of anthropicEventsFromOpenAI(stream, outcome.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  let answer: unknown;
  try {
    answer = await outcome.upstream.json();
  } catch {
    return refused(res, 'anthropic', 'the gateway answer did not parse', [
      ...outcome.failures,
      { status: 200, code: 'unparseable_body', model: outcome.model },
    ]);
  }
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, outcome.model), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

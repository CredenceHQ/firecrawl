import { parseArgs } from "node:util";
import fs from "node:fs/promises";
import path from "node:path";
import { JSDOM } from "jsdom";
import OpenAI from "openai";

/**
 * Shape of the request object accepted by the standalone extractor script.
 * This mirrors the most important pieces of the public extract endpoint
 * without any of the queueing, billing, or telemetry layers from the
 * production service.
 */
export interface StandaloneExtractRequest {
  urls?: string[];
  prompt?: string;
  schema?: any;
  systemPrompt?: string;
  model?: string;
  searchLimit?: number;
  includeHtml?: boolean;
  includeMarkdown?: boolean;
  timeoutMs?: number;
}

interface CliArguments {
  command: "extract" | "search";
  request?: StandaloneExtractRequest;
  query?: string;
  limit?: number;
  requestInput?: string;
  schemaInput?: string;
}

interface ScrapedDocument {
  url: string;
  markdown?: string;
  html?: string;
  metadata: {
    title?: string | null;
    description?: string | null;
  };
}

interface SimpleSearchResult {
  url: string;
  title: string;
  description: string;
}

/**
 * Normalises a user provided URL. If the scheme is missing we default to
 * HTTPS. Invalid URLs are discarded by returning null.
 */
function normalizeCandidateUrl(url: string): string | null {
  try {
    return new URL(url).toString();
  } catch (_) {
    try {
      return new URL(`https://${url}`).toString();
    } catch (error) {
      return null;
    }
  }
}

interface ExtractExecutionResult {
  success: boolean;
  documents: ScrapedDocument[];
  data?: unknown;
  errors?: { url: string; message: string }[];
  urls: string[];
}

const DEFAULT_USER_AGENT =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36";
const DEFAULT_TIMEOUT = 45_000;
const DEFAULT_SEARCH_LIMIT = 5;

let turndownServicePromise: Promise<any> | null = null;

/**
 * Lazily loads Turndown and its GitHub-Flavoured Markdown helpers.
 * The production API relies on a Go shared object for conversion, but for
 * the standalone script we stick to the JS implementation to keep
 * dependencies minimal and easy to port to Kotlin.
 */
async function getTurndownService(): Promise<any> {
  if (!turndownServicePromise) {
    turndownServicePromise = (async () => {
      const [{ default: TurndownService }, gfmModule] = await Promise.all([
        import("turndown"),
        import("joplin-turndown-plugin-gfm"),
      ]);
      const service = new TurndownService({ headingStyle: "atx" });
      if (gfmModule?.gfm) {
        service.use(gfmModule.gfm);
      }
      return service;
    })();
  }
  return turndownServicePromise;
}

/**
 * Parses a CLI provided JSON argument. The value can be a raw JSON string
 * or a filesystem path pointing to a JSON file.
 */
async function loadJsonInput<T>(input?: string): Promise<T | undefined> {
  if (!input) return undefined;
  const trimmed = input.trim();
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    return JSON.parse(trimmed) as T;
  }
  const absolutePath = path.resolve(trimmed);
  const contents = await fs.readFile(absolutePath, "utf-8");
  return JSON.parse(contents) as T;
}

/**
 * Small helper to build the argument object expected by the script. The
 * Node.js built-in `parseArgs` keeps the parsing logic dependency free
 * and mirrors how a future Kotlin port could parse CLI arguments.
 */
function parseCliArguments(): CliArguments {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      request: { type: "string" },
      query: { type: "string" },
      limit: { type: "string" },
      urls: { type: "string" },
      prompt: { type: "string" },
      schema: { type: "string" },
      model: { type: "string" },
      includeHtml: { type: "boolean" },
      includeMarkdown: { type: "boolean" },
      timeout: { type: "string" },
      systemPrompt: { type: "string" },
      searchLimit: { type: "string" },
    },
  });

  if (positionals.length === 0) {
    throw new Error(
      "Missing command. Use either 'extract' or 'search'. Pass --help for options.",
    );
  }

  const command = positionals[0];
  if (command !== "extract" && command !== "search") {
    throw new Error(`Unknown command '${command}'. Use 'extract' or 'search'.`);
  }

  const limit = values.limit ? Number.parseInt(values.limit, 10) : undefined;
  if (values.limit && Number.isNaN(limit)) {
    throw new Error(
      `Invalid numeric value provided to --limit: ${values.limit}`,
    );
  }

  const searchLimit = values.searchLimit
    ? Number.parseInt(values.searchLimit, 10)
    : undefined;
  if (values.searchLimit && Number.isNaN(searchLimit)) {
    throw new Error(
      `Invalid numeric value provided to --searchLimit: ${values.searchLimit}`,
    );
  }

  const timeoutMs = values.timeout
    ? Number.parseInt(values.timeout, 10)
    : undefined;
  if (values.timeout && Number.isNaN(timeoutMs)) {
    throw new Error(
      `Invalid numeric value provided to --timeout: ${values.timeout}`,
    );
  }

  const requestFromArgs: StandaloneExtractRequest = {};

  if (values.prompt !== undefined) requestFromArgs.prompt = values.prompt;
  if (values.model !== undefined) requestFromArgs.model = values.model;
  if (values.systemPrompt !== undefined)
    requestFromArgs.systemPrompt = values.systemPrompt;
  if (values.includeHtml !== undefined)
    requestFromArgs.includeHtml = values.includeHtml;
  if (values.includeMarkdown !== undefined)
    requestFromArgs.includeMarkdown = values.includeMarkdown;
  if (timeoutMs !== undefined) requestFromArgs.timeoutMs = timeoutMs;
  if (searchLimit !== undefined) requestFromArgs.searchLimit = searchLimit;

  if (values.urls) {
    requestFromArgs.urls = values.urls
      .split(",")
      .map(url => url.trim())
      .filter(Boolean);
  }

  return {
    command,
    request: requestFromArgs,
    query: values.query,
    limit,
    requestInput: values.request,
    schemaInput: values.schema,
  };
}

/**
 * When the user specifies wildcard URLs (e.g. `https://example.com/*`), we
 * approximate the behaviour of the API by issuing a focused Google search
 * against that domain and returning the discovered URLs.
 */
async function expandWildcardUrl(
  rawUrl: string,
  prompt: string | undefined,
  limit: number,
): Promise<string[]> {
  const normalized = rawUrl.replace(/\/\*+$/, "");
  try {
    const target = new URL(normalized);
    const queryComponents = ["site:" + target.hostname];
    if (prompt) {
      queryComponents.push(prompt);
    }
    const query = queryComponents.join(" ");
    const results = await runSimpleSearch(query, limit);
    return results
      .map(result => result.url)
      .filter(url => url.startsWith(target.origin));
  } catch (error) {
    throw new Error(`Invalid wildcard URL '${rawUrl}': ${String(error)}`);
  }
}

/**
 * Performs a very small Google scrape. This mirrors the behaviour of the
 * production `googleSearch` helper but omits logging, retries, and
 * per-team rate limiting so the logic stays portable.
 */
async function runSimpleSearch(
  query: string,
  limit: number = DEFAULT_SEARCH_LIMIT,
): Promise<SimpleSearchResult[]> {
  const results: SimpleSearchResult[] = [];
  let start = 0;
  const maxAttempts = 10;
  let attempts = 0;

  while (results.length < limit && attempts < maxAttempts) {
    const searchUrl = new URL("https://www.google.com/search");
    searchUrl.searchParams.set("q", query);
    searchUrl.searchParams.set("num", String(Math.min(10, limit)));
    searchUrl.searchParams.set("hl", "en");
    searchUrl.searchParams.set("start", String(start));

    const response = await fetch(searchUrl, {
      headers: {
        "User-Agent": DEFAULT_USER_AGENT,
        Accept:
          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      },
    });

    if (!response.ok) {
      throw new Error(
        `Failed to execute search request: ${response.status} ${response.statusText}`,
      );
    }

    const html = await response.text();
    const dom = new JSDOM(html);
    const document = dom.window.document;

    const blocks = document.querySelectorAll<HTMLDivElement>("div.g");
    if (blocks.length === 0) {
      attempts += 1;
      start += 10;
      continue;
    }

    for (let index = 0; index < blocks.length; index += 1) {
      const block = blocks.item(index);
      if (!block) continue;

      const anchor = block.querySelector<HTMLAnchorElement>("a[href]");
      const titleEl = block.querySelector("h3");
      const descriptionEl = block.querySelector("div.IsZvec");
      if (!anchor || !titleEl || !descriptionEl) continue;

      let url = anchor.href;
      if (url.startsWith("https://www.google.com/url")) {
        const parsed = new URL(url);
        url = parsed.searchParams.get("q") ?? url;
      }
      const title = titleEl.textContent?.trim() ?? "";
      const description = descriptionEl.textContent?.trim() ?? "";
      if (!url || !title) continue;

      const normalized = normalizeCandidateUrl(url);
      if (!normalized) continue;

      if (!results.find(result => result.url === normalized)) {
        results.push({ url: normalized, title, description });
      }
      if (results.length >= limit) break;
    }

    start += blocks.length;
    attempts = 0;
  }

  return results.slice(0, limit);
}

/**
 * Converts HTML into Markdown by stripping away noisy tags and running the
 * Turndown conversion pipeline. The implementation intentionally avoids any
 * global state so that the logic ports cleanly to other languages.
 */
async function htmlToMarkdown(html: string): Promise<string> {
  const service = await getTurndownService();
  return service.turndown(html);
}

/**
 * Minimal DOM post-processing. Removes script-like tags and returns the
 * cleaned HTML and extracted metadata (title, description).
 */
function sanitizeDom(dom: JSDOM): {
  html: string;
  title: string | null;
  description: string | null;
} {
  const { document } = dom.window;
  document
    .querySelectorAll("script, style, noscript, iframe")
    .forEach(node => node.remove());
  const title = document.querySelector("title")?.textContent?.trim() ?? null;
  const description =
    document.querySelector<HTMLMetaElement>("meta[name=description]")
      ?.content ?? null;
  return {
    html: document.body?.innerHTML ?? "",
    title,
    description,
  };
}

/**
 * Fetches a single URL, sanitises the DOM, and converts the resulting HTML
 * to Markdown. Timeouts are implemented with an AbortController so the
 * script does not hang forever when a site is unresponsive.
 */
async function scrapeSingleUrl(
  url: string,
  options: {
    includeHtml?: boolean;
    includeMarkdown?: boolean;
    timeoutMs?: number;
  },
): Promise<ScrapedDocument> {
  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    options.timeoutMs ?? DEFAULT_TIMEOUT,
  );

  try {
    const response = await fetch(url, {
      headers: {
        "User-Agent": DEFAULT_USER_AGENT,
      },
      redirect: "follow",
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(
        `Request failed: ${response.status} ${response.statusText}`,
      );
    }

    const html = await response.text();
    const dom = new JSDOM(html, { url });
    const cleaned = sanitizeDom(dom);
    const markdown =
      options.includeMarkdown !== false
        ? await htmlToMarkdown(cleaned.html)
        : undefined;

    return {
      url,
      markdown,
      html: options.includeHtml ? cleaned.html : undefined,
      metadata: {
        title: cleaned.title,
        description: cleaned.description,
      },
    };
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Prepares the concatenated document text that will be passed into the LLM
 * extractor when a schema or free-form prompt is provided.
 */
function buildDocumentBundle(documents: ScrapedDocument[]): string {
  return documents
    .map(doc => {
      const title = doc.metadata.title ? `Title: ${doc.metadata.title}\n` : "";
      const description = doc.metadata.description
        ? `Description: ${doc.metadata.description}\n`
        : "";
      const markdown = doc.markdown ?? "";
      return `URL: ${doc.url}\n${title}${description}\n${markdown}`.trim();
    })
    .join("\n\n---\n\n");
}

/**
 * Runs a schema based extraction against OpenAI's Responses API. The
 * implementation keeps the prompt straightforward so the same idea can be
 * ported to Kotlin using any HTTP client.
 */
async function runSchemaExtraction(
  documents: ScrapedDocument[],
  request: StandaloneExtractRequest,
): Promise<unknown> {
  if (!request.schema) return undefined;
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error(
      "OPENAI_API_KEY is required for schema based extraction, but it was not found in the environment.",
    );
  }

  const client = new OpenAI({ apiKey });
  const model = request.model ?? "gpt-4o-mini";
  const schemaJson = JSON.stringify(request.schema, null, 2);
  const bundle = buildDocumentBundle(documents);
  const promptSections = [
    request.prompt
      ? `Instruction: ${request.prompt}`
      : "Instruction: Extract the requested information from the documents.",
    "Return a JSON object that strictly matches the provided schema. Use null when the information is missing.",
    `JSON Schema:\n${schemaJson}`,
    `Source Documents:\n${bundle}`,
  ];

  const systemPrompt =
    request.systemPrompt ??
    "You are a careful data extraction assistant. Only respond with valid JSON that matches the provided schema.";

  const response = await client.responses.create({
    model,
    input: [
      {
        role: "system",
        content: [{ type: "input_text", text: systemPrompt }],
      },
      {
        role: "user",
        content: [{ type: "input_text", text: promptSections.join("\n\n") }],
      },
    ],
  });

  const outputText = response.output_text?.trim();
  if (!outputText) {
    throw new Error("LLM response did not contain any text output");
  }

  try {
    return JSON.parse(outputText);
  } catch (error) {
    throw new Error(
      `Failed to parse model response as JSON. Raw response: ${outputText}. Error: ${String(error)}`,
    );
  }
}

/**
 * When a prompt is provided without a schema we ask the model to answer the
 * question directly. The result is returned as free-form markdown so the
 * caller can decide how to use it.
 */
async function runPromptOnlyExtraction(
  documents: ScrapedDocument[],
  request: StandaloneExtractRequest,
): Promise<string | undefined> {
  if (!request.prompt || request.schema) return undefined;
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.warn(
      "OPENAI_API_KEY is not set. Skipping LLM extraction and returning raw documents only.",
    );
    return undefined;
  }

  const client = new OpenAI({ apiKey });
  const model = request.model ?? "gpt-4o-mini";
  const bundle = buildDocumentBundle(documents);
  const systemPrompt =
    request.systemPrompt ??
    "You answer questions about the provided documents using concise markdown.";

  const response = await client.responses.create({
    model,
    input: [
      {
        role: "system",
        content: [{ type: "input_text", text: systemPrompt }],
      },
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: `Question: ${request.prompt}\n\nDocuments:\n${bundle}`,
          },
        ],
      },
    ],
  });

  return response.output_text?.trim();
}

/**
 * Determines the set of URLs that should be scraped based on the incoming
 * request. Supports explicit URL lists, wildcard URLs, or falling back to a
 * search generated list when only a prompt is supplied.
 */
async function resolveTargetUrls(
  request: StandaloneExtractRequest,
): Promise<string[]> {
  const limit = request.searchLimit ?? DEFAULT_SEARCH_LIMIT;
  const urls = new Set<string>();

  if (request.urls) {
    for (const candidate of request.urls) {
      if (candidate.endsWith("/*")) {
        const expanded = await expandWildcardUrl(
          candidate,
          request.prompt,
          limit,
        );
        expanded.forEach(url => urls.add(url));
      } else {
        const normalized = normalizeCandidateUrl(candidate);
        if (normalized) {
          urls.add(normalized);
        }
      }
    }
  }

  if (urls.size === 0 && request.prompt) {
    const searchResults = await runSimpleSearch(request.prompt, limit);
    searchResults.forEach(result => urls.add(result.url));
  }

  return Array.from(urls);
}

/**
 * Coordinates the scraping and optional LLM extraction phases. Any network
 * errors are captured and returned alongside the successful documents so the
 * caller can decide how to handle partial results.
 */
async function executeExtract(
  request: StandaloneExtractRequest,
): Promise<ExtractExecutionResult> {
  const targetUrls = await resolveTargetUrls(request);
  if (targetUrls.length === 0) {
    throw new Error(
      "No URLs to process. Provide explicit URLs or a prompt that can be used for search.",
    );
  }

  const documents: ScrapedDocument[] = [];
  const errors: { url: string; message: string }[] = [];

  for (const url of targetUrls) {
    try {
      const doc = await scrapeSingleUrl(url, {
        includeHtml: request.includeHtml,
        includeMarkdown: request.includeMarkdown,
        timeoutMs: request.timeoutMs,
      });
      documents.push(doc);
    } catch (error) {
      errors.push({
        url,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }

  if (documents.length === 0) {
    return {
      success: false,
      urls: targetUrls,
      documents,
      errors,
    };
  }

  let data: unknown = undefined;
  if (request.schema) {
    data = await runSchemaExtraction(documents, request);
  } else if (request.prompt) {
    data = await runPromptOnlyExtraction(documents, request);
  }

  return {
    success: true,
    urls: targetUrls,
    documents,
    data,
    errors: errors.length > 0 ? errors : undefined,
  };
}

/**
 * Entry point that wires argument parsing and command execution together.
 */
async function main(): Promise<void> {
  const cli = parseCliArguments();

  if (cli.command === "search") {
    const query = cli.query;
    if (!query) {
      throw new Error("The search command requires --query to be provided.");
    }
    const limit = cli.limit ?? DEFAULT_SEARCH_LIMIT;
    const results = await runSimpleSearch(query, limit);
    console.log(
      JSON.stringify(
        {
          success: true,
          query,
          results,
        },
        null,
        2,
      ),
    );
    return;
  }

  const requestArg = cli.request ?? {};
  const fileRequest =
    (await loadJsonInput<StandaloneExtractRequest>(cli.requestInput)) ?? {};
  const schemaFromCli = await loadJsonInput<any>(cli.schemaInput);

  const merged: StandaloneExtractRequest = {
    ...fileRequest,
    ...requestArg,
  };

  if (schemaFromCli !== undefined) {
    merged.schema = schemaFromCli;
  }

  const result = await executeExtract(merged);
  console.log(JSON.stringify(result, null, 2));
}

main().catch(error => {
  console.error(
    JSON.stringify({ success: false, error: error.message ?? String(error) }),
  );
  process.exitCode = 1;
});

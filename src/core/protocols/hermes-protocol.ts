import type {
  LanguageModelV3Content,
  LanguageModelV3FunctionTool,
  LanguageModelV3StreamPart,
  LanguageModelV3ToolCall,
} from "@ai-sdk/provider";
import { parse as parseRJSON } from "../../rjson";
import { logParseFailure } from "../utils/debug";
import { getPotentialStartIndex } from "../utils/get-potential-start-index";
import { generateId, generateToolCallId } from "../utils/id";
import {
  addTextSegment,
  formatToolsWithPromptTemplate,
} from "../utils/protocol-utils";
import { escapeRegExp } from "../utils/regex";
import {
  emitToolInputProgressDelta,
  shouldEmitRawToolCallTextOnError,
  stringifyToolInputWithSchema,
} from "../utils/tool-input-streaming";
import type { ParserOptions, TCMProtocol } from "./protocol-interface";

interface HermesProtocolOptions {
  toolCallEnd?: string;
  toolCallStart?: string;
}

function canonicalizeToolInput(argumentsValue: unknown): string {
  return JSON.stringify(argumentsValue ?? {});
}

/**
 * Extract known argument key names from the tool schema matching the
 * tool name found in a (possibly malformed) JSON string.
 */
function extractKnownArgKeys(
  tools: LanguageModelV3FunctionTool[],
  toolCallJson: string
): string[] | undefined {
  const toolName = extractTopLevelStringProperty(toolCallJson, "name");
  if (!toolName) {
    return undefined;
  }
  const tool = tools.find((t) => t.name === toolName);
  if (!tool?.inputSchema?.properties) {
    return undefined;
  }
  return Object.keys(tool.inputSchema.properties as Record<string, unknown>);
}

/** Maximum size (in UTF-16 code units) for the arguments body before bailing out of repair. */
const REPAIR_MAX_ARGS_BODY_SIZE = 102_400;

const WHITESPACE_RE = /\s/;
const FIRST_KEY_RE = /^\s*"([^"]+)"\s*:\s*/;
const KV_PATTERN_RE = /,\s*"([^"]+)"\s*:\s*/g;
const TRAILING_COMMA_RE = /,\s*$/;

/**
 * Returns true if `position` in `argsBody` is at nesting depth 0
 * (not inside a nested `{}` or `[]`).
 */
function isAtTopLevel(argsBody: string, position: number): boolean {
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = 0; i < position; i++) {
    const ch = argsBody[i];
    if (esc) {
      esc = false;
      continue;
    }
    if (ch === "\\" && inStr) {
      esc = true;
      continue;
    }
    if (ch === '"') {
      inStr = !inStr;
      continue;
    }
    if (!inStr) {
      if (ch === "{" || ch === "[") {
        depth++;
      }
      if (ch === "}" || ch === "]") {
        depth--;
      }
    }
  }
  return depth === 0;
}

/**
 * Attempt to repair a malformed tool-call JSON string where the model
 * failed to escape double-quotes inside string values.  Returns the
 * parsed result or null when repair is not possible.
 */
// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: JSON repair requires manual character-level scanning with multiple heuristic passes.
function repairToolCallJson(
  raw: string,
  knownArgKeys?: string[]
): { name: string; arguments: Record<string, unknown> } | null {
  // 1. Extract tool name (depth-aware to avoid matching nested "name" keys)
  const toolName = extractTopLevelStringProperty(raw, "name");
  if (!toolName) {
    return null;
  }

  // 2. Find arguments object boundaries (top-level aware, like name extraction)
  const argsValueStart = findTopLevelPropertyValueStart(raw, "arguments");
  if (argsValueStart == null || raw.charAt(argsValueStart) !== "{") {
    return null;
  }
  const argsStart = argsValueStart + 1;

  // 3. Find closing braces from end (arguments + outer object).
  //    Uses backwards scan because forward brace-balance is unreliable
  //    when quotes are broken (which is the premise of this repair path).
  let outerClose = -1;
  for (let i = raw.length - 1; i >= argsStart; i--) {
    if (raw.charAt(i) === "}") {
      outerClose = i;
      break;
    }
    if (!WHITESPACE_RE.test(raw.charAt(i))) {
      break;
    }
  }
  if (outerClose === -1) {
    return null;
  }

  let argsClose = -1;
  for (let j = outerClose - 1; j >= argsStart; j--) {
    if (raw.charAt(j) === "}") {
      argsClose = j;
      break;
    }
    if (!WHITESPACE_RE.test(raw.charAt(j))) {
      break;
    }
  }
  if (argsClose === -1) {
    return null;
  }

  const argsBody = raw.slice(argsStart, argsClose);

  // Size guard: bail out on unreasonably large argument bodies
  if (argsBody.length > REPAIR_MAX_ARGS_BODY_SIZE) {
    return null;
  }

  // 4. Try standard parse first
  try {
    return {
      name: toolName,
      arguments: JSON.parse(`{${argsBody}}`) as Record<string, unknown>,
    };
  } catch {
    /* fall through to repair */
  }

  // 5. Collect key positions
  const firstKeyMatch = argsBody.match(FIRST_KEY_RE);
  if (!firstKeyMatch) {
    return null;
  }
  let allKeys: Array<{
    key: string;
    matchStart: number;
    valueStart: number;
  }> = [
    {
      key: firstKeyMatch[1],
      matchStart: 0,
      valueStart: firstKeyMatch[0].length,
    },
  ];
  for (const m of argsBody.matchAll(KV_PATTERN_RE)) {
    allKeys.push({
      key: m[1],
      matchStart: m.index,
      valueStart: m.index + m[0].length,
    });
  }

  // 5b. Filter candidates to prevent false boundary splits.
  //     Boundary detection always uses top-level position — dropping
  //     schema-unknown keys from the candidate list corrupts neighbouring
  //     value slices, because their ,"extra":... text gets merged into
  //     the previous value. Schema filtering is applied later when
  //     assigning parsed values into args (step 8 below).
  const knownKeySet =
    knownArgKeys && knownArgKeys.length > 0 ? new Set(knownArgKeys) : null;
  allKeys = allKeys.filter(
    (entry) =>
      entry.matchStart === 0 || isAtTopLevel(argsBody, entry.matchStart)
  );

  // 7. Handle duplicate key names with scoring heuristic
  const firstByKey: Record<string, number> = {};
  const lastByKey: Record<string, number> = {};
  for (let idx = 0; idx < allKeys.length; idx++) {
    if (!(allKeys[idx].key in firstByKey)) {
      firstByKey[allKeys[idx].key] = idx;
    }
    lastByKey[allKeys[idx].key] = idx;
  }
  const firstPositions = allKeys.filter(
    (_, i) => firstByKey[allKeys[i].key] === i
  );
  const lastPositions = allKeys.filter(
    (_, i) => lastByKey[allKeys[i].key] === i
  );

  let keyPositions: typeof allKeys;
  if (
    firstPositions.length === lastPositions.length &&
    firstPositions.every(
      (fp, i) => fp.matchStart === lastPositions[i].matchStart
    )
  ) {
    keyPositions = firstPositions;
  } else {
    // biome-ignore lint/complexity/noExcessiveCognitiveComplexity: Scoring heuristic requires multi-stage fallback parsing.
    function scorePositions(positions: typeof allKeys): [number, number] {
      let rawOk = 0;
      let repaired = 0;
      for (let si = 0; si < positions.length; si++) {
        const svs = positions[si].valueStart;
        const sve =
          si + 1 < positions.length
            ? positions[si + 1].matchStart
            : argsBody.length;
        const srv = argsBody.slice(svs, sve).replace(TRAILING_COMMA_RE, "");
        try {
          JSON.parse(srv);
          rawOk++;
          continue;
        } catch {
          /* skip */
        }
        if (srv.charAt(0) === '"') {
          let seq = srv.length - 1;
          while (seq > 0 && srv.charAt(seq) !== '"') {
            seq--;
          }
          if (seq > 0) {
            const sinner = srv.slice(1, seq);
            let sesc = "";
            let sbs = 0;
            for (const sch of sinner) {
              if (sch === "\\") {
                sbs++;
                sesc += sch;
              } else if (sch === '"' && sbs % 2 === 0) {
                sbs = 0;
                sesc += '\\"';
              } else {
                sbs = 0;
                sesc += sch;
              }
            }
            sesc = sesc
              .replace(/\n/g, "\\n")
              .replace(/\r/g, "\\r")
              .replace(/\t/g, "\\t");
            try {
              JSON.parse(`"${sesc}"`);
              repaired++;
            } catch {
              /* skip */
            }
          }
        }
      }
      return [rawOk, repaired];
    }
    const fs = scorePositions(firstPositions);
    const ls = scorePositions(lastPositions);
    keyPositions =
      ls[0] > fs[0] || (ls[0] === fs[0] && ls[1] > fs[1])
        ? lastPositions
        : firstPositions;
  }
  allKeys = keyPositions;
  if (allKeys.length === 0) {
    return null;
  }

  // 8. Repair each value by escaping unescaped quotes.
  //    Schema-unknown keys are skipped here (their slice was still needed
  //    for correct boundary detection in step 5b).
  const args: Record<string, unknown> = {};
  for (let i = 0; i < allKeys.length; i++) {
    const kp = allKeys[i];
    if (knownKeySet && !knownKeySet.has(kp.key)) {
      continue;
    }
    const vs = kp.valueStart;
    const ve =
      i + 1 < allKeys.length ? allKeys[i + 1].matchStart : argsBody.length;
    const rv = argsBody.slice(vs, ve).replace(TRAILING_COMMA_RE, "");
    try {
      args[kp.key] = JSON.parse(rv);
      continue;
    } catch {
      /* needs repair */
    }
    if (rv.charAt(0) === '"') {
      let eq = rv.length - 1;
      while (eq > 0 && rv.charAt(eq) !== '"') {
        eq--;
      }
      if (eq <= 0) {
        // String literal with no closing quote — repair cannot handle this
        return null;
      }
      const inner = rv.slice(1, eq);
      let esc = "";
      let bs = 0;
      for (const ch of inner) {
        if (ch === "\\") {
          bs++;
          esc += ch;
        } else if (ch === '"' && bs % 2 === 0) {
          bs = 0;
          esc += '\\"';
        } else {
          bs = 0;
          esc += ch;
        }
      }
      esc = esc
        .replace(/\n/g, "\\n")
        .replace(/\r/g, "\\r")
        .replace(/\t/g, "\\t");
      try {
        args[kp.key] = JSON.parse(`"${esc}"`);
      } catch {
        // Repaired string still invalid — bail out
        return null;
      }
    } else {
      // Non-string value that failed JSON.parse — repair cannot handle this
      return null;
    }
  }
  // Guard: if the schema filter skipped every parsed key, we have nothing
  // meaningful to emit. Returning {arguments: {}} here would trigger a
  // tool invocation with empty args — worse than reporting parse failure.
  if (knownKeySet && Object.keys(args).length === 0) {
    return null;
  }
  return { name: toolName, arguments: args };
}

function processToolCallJson(
  toolCallJson: string,
  fullMatch: string,
  processedElements: LanguageModelV3Content[],
  tools: LanguageModelV3FunctionTool[],
  options?: ParserOptions
) {
  try {
    const parsedToolCall = parseRJSON(toolCallJson) as {
      name: string;
      arguments: unknown;
    };
    processedElements.push({
      type: "tool-call",
      toolCallId: generateToolCallId(),
      toolName: parsedToolCall.name,
      input: canonicalizeToolInput(parsedToolCall.arguments),
    });
  } catch (error) {
    // Attempt repair for unescaped quotes
    const repaired = repairToolCallJson(
      toolCallJson,
      extractKnownArgKeys(tools, toolCallJson)
    );
    if (repaired) {
      processedElements.push({
        type: "tool-call",
        toolCallId: generateToolCallId(),
        toolName: repaired.name,
        input: canonicalizeToolInput(repaired.arguments),
      });
      return;
    }
    logParseFailure({
      phase: "generated-text",
      reason: "Failed to parse tool call JSON segment",
      snippet: fullMatch,
      error,
    });
    options?.onError?.(
      "Could not process JSON tool call, keeping original text.",
      { toolCall: fullMatch, error }
    );
    processedElements.push({ type: "text", text: fullMatch });
  }
}

interface ParseContext {
  currentIndex: number;
  match: RegExpExecArray;
  options?: ParserOptions;
  processedElements: LanguageModelV3Content[];
  text: string;
  tools: LanguageModelV3FunctionTool[];
}

function processMatchedToolCall(context: ParseContext): number {
  const { match, text, currentIndex, processedElements, tools, options } =
    context;
  const startIndex = match.index;
  const toolCallJson = match[1];

  if (startIndex > currentIndex) {
    const textSegment = text.slice(currentIndex, startIndex);
    addTextSegment(textSegment, processedElements);
  }

  if (toolCallJson) {
    processToolCallJson(
      toolCallJson,
      match[0],
      processedElements,
      tools,
      options
    );
  }

  return startIndex + match[0].length;
}

interface StreamState {
  activeToolInput: {
    id: string;
    toolName: string;
    emittedInput: string;
  } | null;
  buffer: string;
  currentTextId: string | null;
  currentToolCallJson: string;
  hasEmittedTextStart: boolean;
  isInsideToolCall: boolean;
}

type StreamController =
  TransformStreamDefaultController<LanguageModelV3StreamPart>;

interface TagProcessingContext {
  controller: StreamController;
  options?: ParserOptions;
  state: StreamState;
  toolCallEnd: string;
  toolCallStart: string;
  tools: LanguageModelV3FunctionTool[];
}

const WHITESPACE_JSON_REGEX = /\s/;

function skipJsonWhitespace(text: string, fromIndex: number): number {
  let index = fromIndex;
  while (index < text.length && WHITESPACE_JSON_REGEX.test(text[index])) {
    index += 1;
  }
  return index;
}

// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: Streaming JSON key/value scanning requires explicit string-depth state tracking.
function findTopLevelPropertyValueStart(
  text: string,
  property: string
): number | null {
  const objectStart = skipJsonWhitespace(text, 0);
  if (objectStart >= text.length || text.charAt(objectStart) !== "{") {
    return null;
  }

  let depth = 0;
  let inString = false;
  let escaping = false;

  for (let index = objectStart; index < text.length; index += 1) {
    const char = text.charAt(index);

    if (inString) {
      if (escaping) {
        escaping = false;
        continue;
      }
      if (char === "\\") {
        escaping = true;
        continue;
      }
      if (char === '"') {
        inString = false;
      }
      continue;
    }

    if (char === "{") {
      depth += 1;
      continue;
    }
    if (char === "}") {
      depth = Math.max(0, depth - 1);
      continue;
    }

    if (char !== '"') {
      continue;
    }

    if (depth !== 1) {
      inString = true;
      continue;
    }

    const keyStart = index + 1;
    let keyEnd = keyStart;
    let keyEscaped = false;
    while (keyEnd < text.length) {
      const keyChar = text.charAt(keyEnd);
      if (keyEscaped) {
        keyEscaped = false;
      } else if (keyChar === "\\") {
        keyEscaped = true;
      } else if (keyChar === '"') {
        break;
      }
      keyEnd += 1;
    }

    if (keyEnd >= text.length || text.charAt(keyEnd) !== '"') {
      return null;
    }

    const key = text.slice(keyStart, keyEnd);
    let valueCursor = skipJsonWhitespace(text, keyEnd + 1);
    if (valueCursor >= text.length || text.charAt(valueCursor) !== ":") {
      index = keyEnd;
      continue;
    }

    valueCursor = skipJsonWhitespace(text, valueCursor + 1);
    if (key === property) {
      return valueCursor < text.length ? valueCursor : null;
    }

    index = valueCursor - 1;
  }

  return null;
}

function extractTopLevelStringProperty(
  text: string,
  property: string
): string | undefined {
  const valueStart = findTopLevelPropertyValueStart(text, property);
  if (valueStart == null || valueStart >= text.length) {
    return undefined;
  }
  if (text.charAt(valueStart) !== '"') {
    return undefined;
  }

  let valueEnd = valueStart + 1;
  let escaped = false;
  while (valueEnd < text.length) {
    const char = text.charAt(valueEnd);
    if (escaped) {
      escaped = false;
    } else if (char === "\\") {
      escaped = true;
    } else if (char === '"') {
      return text.slice(valueStart + 1, valueEnd);
    }
    valueEnd += 1;
  }

  return undefined;
}

// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: Streaming JSON value slicing must handle nested arrays/objects and escaped strings.
function extractJsonValueSlice(
  text: string,
  valueStart: number
): {
  text: string;
  complete: boolean;
} | null {
  if (valueStart >= text.length) {
    return null;
  }

  const first = text.charAt(valueStart);
  if (first === "{" || first === "[") {
    const stack: string[] = [first];
    let inString = false;
    let escaped = false;

    for (let index = valueStart + 1; index < text.length; index += 1) {
      const char = text.charAt(index);
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (char === "\\") {
          escaped = true;
        } else if (char === '"') {
          inString = false;
        }
        continue;
      }

      if (char === '"') {
        inString = true;
        continue;
      }

      if (char === "{" || char === "[") {
        stack.push(char);
        continue;
      }

      if (char === "}" || char === "]") {
        const open = stack.at(-1);
        if ((open === "{" && char === "}") || (open === "[" && char === "]")) {
          stack.pop();
          if (stack.length === 0) {
            return {
              text: text.slice(valueStart, index + 1),
              complete: true,
            };
          }
        }
      }
    }

    return {
      text: text.slice(valueStart),
      complete: false,
    };
  }

  if (first === '"') {
    let escaped = false;
    for (let index = valueStart + 1; index < text.length; index += 1) {
      const char = text.charAt(index);
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === '"') {
        return {
          text: text.slice(valueStart, index + 1),
          complete: true,
        };
      }
    }
    return {
      text: text.slice(valueStart),
      complete: false,
    };
  }

  let index = valueStart;
  while (index < text.length) {
    const char = text.charAt(index);
    if (char === "," || char === "}" || WHITESPACE_JSON_REGEX.test(char)) {
      break;
    }
    index += 1;
  }

  return {
    text: text.slice(valueStart, index),
    complete: index < text.length,
  };
}

function extractStreamingToolCallProgress(toolCallJson: string): {
  toolName: string | undefined;
  argumentsText: string | undefined;
  argumentsComplete: boolean;
} {
  const toolName = extractTopLevelStringProperty(toolCallJson, "name");
  const argsValueStart = findTopLevelPropertyValueStart(
    toolCallJson,
    "arguments"
  );
  if (argsValueStart == null) {
    return {
      toolName,
      argumentsText: undefined,
      argumentsComplete: false,
    };
  }

  const argsSlice = extractJsonValueSlice(toolCallJson, argsValueStart);
  return {
    toolName,
    argumentsText: argsSlice?.text,
    argumentsComplete: argsSlice?.complete ?? false,
  };
}

function ensureToolInputStart(
  state: StreamState,
  controller: StreamController,
  toolName: string
) {
  if (!state.activeToolInput) {
    const id = generateToolCallId();
    state.activeToolInput = {
      id,
      toolName,
      emittedInput: "",
    };
    controller.enqueue({
      type: "tool-input-start",
      id,
      toolName,
    } as LanguageModelV3StreamPart);
  }
}

function emitToolInputDelta(
  state: StreamState,
  controller: StreamController,
  fullInput: string
) {
  const active = state.activeToolInput;
  if (!active) {
    return;
  }

  emitToolInputProgressDelta({
    controller,
    id: active.id,
    state: active,
    fullInput,
    mode: "full-json",
  });
}

function closeToolInput(state: StreamState, controller: StreamController) {
  if (!state.activeToolInput) {
    return;
  }
  controller.enqueue({
    type: "tool-input-end",
    id: state.activeToolInput.id,
  } as LanguageModelV3StreamPart);
  state.activeToolInput = null;
}

function emitToolCallFromParsed(
  state: StreamState,
  controller: StreamController,
  parsedToolCall: { name: string; arguments: unknown },
  tools: LanguageModelV3FunctionTool[]
) {
  closeTextBlock(state, controller);
  const toolName =
    typeof parsedToolCall.name === "string"
      ? parsedToolCall.name
      : (state.activeToolInput?.toolName ?? "unknown");
  const input = stringifyToolInputWithSchema({
    toolName,
    args: parsedToolCall.arguments,
    tools,
    fallback: canonicalizeToolInput,
  });
  ensureToolInputStart(state, controller, toolName);
  emitToolInputDelta(state, controller, input);
  const toolCallId = state.activeToolInput?.id ?? generateToolCallId();
  closeToolInput(state, controller);
  controller.enqueue({
    type: "tool-call",
    toolCallId,
    toolName,
    input,
  } as LanguageModelV3StreamPart);
}

function canonicalizeArgumentsProgressInput(
  progress: {
    argumentsText: string | undefined;
    argumentsComplete: boolean;
  },
  toolName: string,
  tools: LanguageModelV3FunctionTool[]
): string | undefined {
  if (progress.argumentsText === undefined || !progress.argumentsComplete) {
    return undefined;
  }

  try {
    const parsedArguments = parseRJSON(progress.argumentsText);
    return stringifyToolInputWithSchema({
      toolName,
      args: parsedArguments,
      tools,
      fallback: canonicalizeToolInput,
    });
  } catch {
    return undefined;
  }
}

function emitToolInputProgress(
  state: StreamState,
  controller: StreamController,
  tools: LanguageModelV3FunctionTool[]
) {
  if (!(state.isInsideToolCall && state.currentToolCallJson)) {
    return;
  }

  const progress = extractStreamingToolCallProgress(state.currentToolCallJson);
  if (!progress.toolName) {
    return;
  }

  ensureToolInputStart(state, controller, progress.toolName);
  const canonicalProgressInput = canonicalizeArgumentsProgressInput(
    progress,
    progress.toolName,
    tools
  );
  if (canonicalProgressInput !== undefined) {
    emitToolInputDelta(state, controller, canonicalProgressInput);
  }
}

function flushBuffer(
  state: StreamState,
  controller: StreamController,
  toolCallStart: string
) {
  if (state.buffer.length === 0) {
    return;
  }

  if (!state.currentTextId) {
    state.currentTextId = generateId();
    controller.enqueue({
      type: "text-start",
      id: state.currentTextId,
    } as LanguageModelV3StreamPart);
    state.hasEmittedTextStart = true;
  }

  const deltaContent = state.isInsideToolCall
    ? `${toolCallStart}${state.buffer}`
    : state.buffer;

  controller.enqueue({
    type: "text-delta",
    id: state.currentTextId,
    delta: deltaContent,
  } as LanguageModelV3StreamPart);
  state.buffer = "";
}

function closeTextBlock(state: StreamState, controller: StreamController) {
  if (state.currentTextId && state.hasEmittedTextStart) {
    controller.enqueue({
      type: "text-end",
      id: state.currentTextId,
    } as LanguageModelV3StreamPart);
    state.currentTextId = null;
    state.hasEmittedTextStart = false;
  }
}

function emitIncompleteToolCall(
  state: StreamState,
  controller: StreamController,
  toolCallStart: string,
  trailingBuffer: string,
  tools: LanguageModelV3FunctionTool[],
  options?: ParserOptions
) {
  if (!state.currentToolCallJson && trailingBuffer.length === 0) {
    state.isInsideToolCall = false;
    return;
  }

  if (state.currentToolCallJson) {
    try {
      const parsedToolCall = parseRJSON(state.currentToolCallJson) as {
        name: string;
        arguments: unknown;
      };
      emitToolCallFromParsed(state, controller, parsedToolCall, tools);
      state.currentToolCallJson = "";
      state.isInsideToolCall = false;
      return;
    } catch {
      // Incomplete tool calls (no closing </tool_call>) are not candidates
      // for repair — the JSON may be genuinely truncated.
      // Fall through to text/error fallback.
    }
  }

  const rawToolCallContent = `${state.currentToolCallJson}${trailingBuffer}`;
  const errorContent = `${toolCallStart}${rawToolCallContent}`;
  const shouldEmitRawFallback = shouldEmitRawToolCallTextOnError(options);

  logParseFailure({
    phase: "stream",
    reason: shouldEmitRawFallback
      ? "Incomplete streaming tool call segment emitted as text"
      : "Incomplete streaming tool call segment suppressed without raw text fallback",
    snippet: errorContent,
  });

  if (shouldEmitRawFallback) {
    const errorId = generateId();
    controller.enqueue({
      type: "text-start",
      id: errorId,
    } as LanguageModelV3StreamPart);
    controller.enqueue({
      type: "text-delta",
      id: errorId,
      delta: errorContent,
    } as LanguageModelV3StreamPart);
    controller.enqueue({
      type: "text-end",
      id: errorId,
    } as LanguageModelV3StreamPart);
  }
  closeToolInput(state, controller);
  options?.onError?.(
    shouldEmitRawFallback
      ? "Could not complete streaming JSON tool call at finish; emitting original text."
      : "Could not complete streaming JSON tool call at finish.",
    { toolCall: errorContent }
  );
  state.currentToolCallJson = "";
  state.isInsideToolCall = false;
}

function handleFinishChunk(
  state: StreamState,
  controller: StreamController,
  toolCallStart: string,
  tools: LanguageModelV3FunctionTool[],
  options: ParserOptions | undefined,
  chunk: LanguageModelV3StreamPart
) {
  if (state.isInsideToolCall) {
    const trailingBuffer = state.buffer;
    state.buffer = "";
    emitIncompleteToolCall(
      state,
      controller,
      toolCallStart,
      trailingBuffer,
      tools,
      options
    );
  } else if (state.buffer.length > 0) {
    flushBuffer(state, controller, toolCallStart);
  }
  closeTextBlock(state, controller);
  controller.enqueue(chunk);
}

function publishText(
  text: string,
  state: StreamState,
  controller: StreamController,
  tools: LanguageModelV3FunctionTool[]
) {
  if (state.isInsideToolCall) {
    closeTextBlock(state, controller);
    state.currentToolCallJson += text;
    emitToolInputProgress(state, controller, tools);
  } else if (text.length > 0) {
    if (!state.currentTextId) {
      state.currentTextId = generateId();
      controller.enqueue({
        type: "text-start",
        id: state.currentTextId,
      } as LanguageModelV3StreamPart);
      state.hasEmittedTextStart = true;
    }
    controller.enqueue({
      type: "text-delta",
      id: state.currentTextId,
      delta: text,
    } as LanguageModelV3StreamPart);
  }
}

function emitToolCall(context: TagProcessingContext) {
  const { state, controller, toolCallStart, toolCallEnd, options, tools } =
    context;
  try {
    const parsedToolCall = parseRJSON(state.currentToolCallJson) as {
      name: string;
      arguments: unknown;
    };
    emitToolCallFromParsed(state, controller, parsedToolCall, tools);
  } catch (error) {
    // Attempt repair for unescaped quotes
    const repaired = repairToolCallJson(
      state.currentToolCallJson,
      extractKnownArgKeys(tools, state.currentToolCallJson)
    );
    if (repaired) {
      emitToolCallFromParsed(state, controller, repaired, tools);
      return;
    }

    const errorContent = `${toolCallStart}${state.currentToolCallJson}${toolCallEnd}`;
    const shouldEmitRawFallback = shouldEmitRawToolCallTextOnError(options);

    logParseFailure({
      phase: "stream",
      reason: "Failed to parse streaming tool call JSON segment",
      snippet: errorContent,
      error,
    });
    if (shouldEmitRawFallback) {
      const errorId = generateId();
      controller.enqueue({
        type: "text-start",
        id: errorId,
      } as LanguageModelV3StreamPart);
      controller.enqueue({
        type: "text-delta",
        id: errorId,
        delta: errorContent,
      } as LanguageModelV3StreamPart);
      controller.enqueue({
        type: "text-end",
        id: errorId,
      } as LanguageModelV3StreamPart);
    }
    closeToolInput(state, controller);
    options?.onError?.(
      shouldEmitRawFallback
        ? "Could not process streaming JSON tool call; emitting original text."
        : "Could not process streaming JSON tool call.",
      {
        toolCall: errorContent,
      }
    );
  }
}

function processTagMatch(context: TagProcessingContext) {
  const { state } = context;
  if (state.isInsideToolCall) {
    emitToolCall(context);
    state.currentToolCallJson = "";
    state.isInsideToolCall = false;
  } else {
    state.currentToolCallJson = "";
    state.isInsideToolCall = true;
    state.activeToolInput = null;
  }
}

function processBufferTags(context: TagProcessingContext) {
  const { state, controller, toolCallStart, toolCallEnd, tools } = context;
  let startIndex = getPotentialStartIndex(
    state.buffer,
    state.isInsideToolCall ? toolCallEnd : toolCallStart
  );

  while (startIndex != null) {
    const tag = state.isInsideToolCall ? toolCallEnd : toolCallStart;
    if (startIndex + tag.length > state.buffer.length) {
      break;
    }

    publishText(state.buffer.slice(0, startIndex), state, controller, tools);
    state.buffer = state.buffer.slice(startIndex + tag.length);
    processTagMatch(context);

    startIndex = getPotentialStartIndex(
      state.buffer,
      state.isInsideToolCall ? toolCallEnd : toolCallStart
    );
  }
}

function handlePartialTag(
  state: StreamState,
  controller: StreamController,
  toolCallStart: string,
  toolCallEnd: string,
  tools: LanguageModelV3FunctionTool[]
) {
  if (state.isInsideToolCall) {
    const potentialEndIndex = getPotentialStartIndex(state.buffer, toolCallEnd);
    if (
      potentialEndIndex != null &&
      potentialEndIndex + toolCallEnd.length > state.buffer.length
    ) {
      publishText(
        state.buffer.slice(0, potentialEndIndex),
        state,
        controller,
        tools
      );
      state.buffer = state.buffer.slice(potentialEndIndex);
    } else {
      publishText(state.buffer, state, controller, tools);
      state.buffer = "";
    }
    return;
  }

  const potentialIndex = getPotentialStartIndex(state.buffer, toolCallStart);
  if (
    potentialIndex != null &&
    potentialIndex + toolCallStart.length > state.buffer.length
  ) {
    publishText(
      state.buffer.slice(0, potentialIndex),
      state,
      controller,
      tools
    );
    state.buffer = state.buffer.slice(potentialIndex);
  } else {
    publishText(state.buffer, state, controller, tools);
    state.buffer = "";
  }
}

export const hermesProtocol = ({
  toolCallStart = "<tool_call>",
  toolCallEnd = "</tool_call>",
}: HermesProtocolOptions = {}): TCMProtocol => ({
  formatTools({
    tools,
    toolSystemPromptTemplate,
  }: {
    tools: LanguageModelV3FunctionTool[];
    toolSystemPromptTemplate: (tools: LanguageModelV3FunctionTool[]) => string;
  }) {
    return formatToolsWithPromptTemplate({ tools, toolSystemPromptTemplate });
  },

  formatToolCall(toolCall: LanguageModelV3ToolCall) {
    let args: unknown = {};
    if (toolCall.input != null) {
      try {
        args = JSON.parse(toolCall.input);
      } catch {
        args = toolCall.input;
      }
    }
    return `${toolCallStart}${JSON.stringify({
      name: toolCall.toolName,
      arguments: args,
    })}${toolCallEnd}`;
  },

  parseGeneratedText({
    text,
    tools,
    options,
  }: {
    text: string;
    tools: LanguageModelV3FunctionTool[];
    options?: ParserOptions;
  }) {
    const startEsc = escapeRegExp(toolCallStart);
    const endEsc = escapeRegExp(toolCallEnd);
    const toolCallRegex = new RegExp(
      `${startEsc}([\u0000-\uFFFF]*?)${endEsc}`,
      "gs"
    );

    const processedElements: LanguageModelV3Content[] = [];
    let currentIndex = 0;
    let match = toolCallRegex.exec(text);

    while (match !== null) {
      currentIndex = processMatchedToolCall({
        match,
        text,
        currentIndex,
        processedElements,
        tools,
        options,
      });
      match = toolCallRegex.exec(text);
    }

    if (currentIndex < text.length) {
      const remainingText = text.slice(currentIndex);
      addTextSegment(remainingText, processedElements);
    }

    return processedElements;
  },

  createStreamParser({
    tools,
    options,
  }: {
    tools: LanguageModelV3FunctionTool[];
    options?: ParserOptions;
  }) {
    const state: StreamState = {
      isInsideToolCall: false,
      buffer: "",
      currentToolCallJson: "",
      currentTextId: null,
      hasEmittedTextStart: false,
      activeToolInput: null,
    };

    return new TransformStream<
      LanguageModelV3StreamPart,
      LanguageModelV3StreamPart
    >({
      transform(chunk, controller) {
        if (chunk.type === "finish") {
          handleFinishChunk(
            state,
            controller,
            toolCallStart,
            tools,
            options,
            chunk
          );
          return;
        }

        if (chunk.type !== "text-delta") {
          controller.enqueue(chunk);
          return;
        }

        const textContent = (chunk as { delta?: string }).delta ?? "";
        state.buffer += textContent;
        processBufferTags({
          state,
          controller,
          toolCallStart,
          toolCallEnd,
          options,
          tools,
        });
        handlePartialTag(state, controller, toolCallStart, toolCallEnd, tools);
      },
    });
  },

  extractToolCallSegments({ text }: { text: string }) {
    const startEsc = escapeRegExp(toolCallStart);
    const endEsc = escapeRegExp(toolCallEnd);
    const regex = new RegExp(`${startEsc}([\u0000-\uFFFF]*?)${endEsc}`, "gs");
    const segments: string[] = [];
    let m = regex.exec(text);
    while (m != null) {
      segments.push(m[0]);
      m = regex.exec(text);
    }
    return segments;
  },
});

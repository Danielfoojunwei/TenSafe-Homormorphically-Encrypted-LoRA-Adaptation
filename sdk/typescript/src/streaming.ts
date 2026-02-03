/**
 * TenSafe SDK Streaming Support
 *
 * This module provides utilities for handling streaming responses
 * from the TenSafe API, similar to the OpenAI SDK streaming interface.
 */

import type { StreamChunk, StreamChoice, StreamDelta, ChatMessage, Usage } from './types.js';
import { TenSafeError, ConnectionError } from './errors.js';

/**
 * Server-Sent Events (SSE) line parser.
 */
function parseSSELine(line: string): { event?: string; data?: string } | null {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith(':')) {
    return null;
  }

  if (trimmed.startsWith('event:')) {
    return { event: trimmed.slice(6).trim() };
  }

  if (trimmed.startsWith('data:')) {
    return { data: trimmed.slice(5).trim() };
  }

  return null;
}

/**
 * Async iterator that yields stream chunks from a Response.
 */
async function* iterateSSEStream(
  response: Response
): AsyncGenerator<StreamChunk, void, undefined> {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new ConnectionError('Response body is not readable');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const parsed = parseSSELine(line);

        if (parsed?.data) {
          if (parsed.data === '[DONE]') {
            return;
          }

          try {
            const chunk = JSON.parse(parsed.data) as StreamChunk;
            yield chunk;
          } catch {
            // Skip malformed JSON lines
            continue;
          }
        }
      }
    }

    // Process any remaining buffer
    if (buffer.trim()) {
      const parsed = parseSSELine(buffer);
      if (parsed?.data && parsed.data !== '[DONE]') {
        try {
          const chunk = JSON.parse(parsed.data) as StreamChunk;
          yield chunk;
        } catch {
          // Ignore
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * A stream of chat completion chunks.
 *
 * This class implements the AsyncIterable interface, allowing you to
 * iterate over streaming responses using for-await-of loops.
 *
 * @example
 * ```typescript
 * const stream = await client.chat.completions.create({
 *   model: 'llama-3-8b',
 *   messages: [{ role: 'user', content: 'Hello!' }],
 *   stream: true,
 * });
 *
 * for await (const chunk of stream) {
 *   const content = chunk.choices[0]?.delta?.content;
 *   if (content) {
 *     process.stdout.write(content);
 *   }
 * }
 * ```
 */
export class Stream implements AsyncIterable<StreamChunk> {
  private response: Response;
  private controller: AbortController;
  private iterator: AsyncGenerator<StreamChunk, void, undefined> | null = null;

  constructor(response: Response, controller: AbortController) {
    this.response = response;
    this.controller = controller;
  }

  /**
   * Returns an async iterator for the stream.
   */
  [Symbol.asyncIterator](): AsyncIterator<StreamChunk> {
    if (!this.iterator) {
      this.iterator = iterateSSEStream(this.response);
    }
    return this.iterator;
  }

  /**
   * Aborts the stream.
   */
  abort(): void {
    this.controller.abort();
  }

  /**
   * Converts the stream to a complete response by concatenating all chunks.
   * This consumes the stream.
   */
  async toResponse(): Promise<StreamedChatCompletion> {
    const chunks: StreamChunk[] = [];

    for await (const chunk of this) {
      chunks.push(chunk);
    }

    return StreamedChatCompletion.fromChunks(chunks);
  }

  /**
   * Collects all text content from the stream.
   * This consumes the stream.
   */
  async toText(): Promise<string> {
    let text = '';

    for await (const chunk of this) {
      const content = chunk.choices[0]?.delta?.content;
      if (content) {
        text += content;
      }
    }

    return text;
  }
}

/**
 * A completed chat completion assembled from streamed chunks.
 */
export class StreamedChatCompletion {
  /** Response ID */
  readonly id: string;
  /** Object type */
  readonly object: string;
  /** Creation timestamp (Unix) */
  readonly created: number;
  /** Model used */
  readonly model: string;
  /** The complete message */
  readonly message: ChatMessage;
  /** Finish reason */
  readonly finishReason: string | null;
  /** Token usage (if available) */
  readonly usage?: Usage;

  constructor(data: {
    id: string;
    object: string;
    created: number;
    model: string;
    message: ChatMessage;
    finishReason: string | null;
    usage?: Usage;
  }) {
    this.id = data.id;
    this.object = data.object;
    this.created = data.created;
    this.model = data.model;
    this.message = data.message;
    this.finishReason = data.finishReason;
    this.usage = data.usage;
  }

  /**
   * Assembles a StreamedChatCompletion from an array of stream chunks.
   */
  static fromChunks(chunks: StreamChunk[]): StreamedChatCompletion {
    if (chunks.length === 0) {
      throw new TenSafeError('Cannot create completion from empty chunks');
    }

    const firstChunk = chunks[0];
    if (!firstChunk) {
      throw new TenSafeError('Cannot create completion from empty chunks');
    }

    let role = 'assistant';
    let content = '';
    let finishReason: string | null = null;

    for (const chunk of chunks) {
      const choice = chunk.choices[0];
      if (choice) {
        if (choice.delta.role) {
          role = choice.delta.role;
        }
        if (choice.delta.content) {
          content += choice.delta.content;
        }
        if (choice.finishReason) {
          finishReason = choice.finishReason;
        }
      }
    }

    return new StreamedChatCompletion({
      id: firstChunk.id,
      object: 'chat.completion',
      created: firstChunk.created,
      model: firstChunk.model,
      message: {
        role: role as 'assistant',
        content,
      },
      finishReason,
    });
  }
}

/**
 * Options for creating a stream.
 */
export interface StreamOptions {
  /** Abort signal for cancellation */
  signal?: AbortSignal;
  /** Callback for each chunk */
  onChunk?: (chunk: StreamChunk) => void;
  /** Callback for stream start */
  onStart?: () => void;
  /** Callback for stream end */
  onEnd?: () => void;
  /** Callback for errors */
  onError?: (error: Error) => void;
}

/**
 * Creates a stream from a fetch response.
 */
export function createStream(response: Response, controller: AbortController): Stream {
  return new Stream(response, controller);
}

/**
 * Helper class for building streaming requests.
 */
export class StreamingRequestBuilder {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(baseUrl: string, headers: Record<string, string>) {
    this.baseUrl = baseUrl;
    this.headers = headers;
  }

  /**
   * Makes a streaming request and returns a Stream.
   */
  async request(
    path: string,
    body: unknown,
    options: StreamOptions = {}
  ): Promise<Stream> {
    const controller = new AbortController();

    // Combine with user-provided signal if present
    if (options.signal) {
      options.signal.addEventListener('abort', () => controller.abort());
    }

    const response = await fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: {
        ...this.headers,
        'Accept': 'text/event-stream',
        'Cache-Control': 'no-cache',
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new TenSafeError(`Stream request failed: ${text}`, {
        status: response.status,
      });
    }

    options.onStart?.();

    const stream = createStream(response, controller);

    // Wrap in a proxy to handle callbacks
    if (options.onChunk || options.onEnd || options.onError) {
      return new StreamWithCallbacks(stream, options);
    }

    return stream;
  }
}

/**
 * Stream wrapper that calls callbacks on events.
 */
class StreamWithCallbacks implements AsyncIterable<StreamChunk> {
  private stream: Stream;
  private options: StreamOptions;

  constructor(stream: Stream, options: StreamOptions) {
    this.stream = stream;
    this.options = options;
  }

  async *[Symbol.asyncIterator](): AsyncIterator<StreamChunk> {
    try {
      for await (const chunk of this.stream) {
        this.options.onChunk?.(chunk);
        yield chunk;
      }
      this.options.onEnd?.();
    } catch (error) {
      this.options.onError?.(error as Error);
      throw error;
    }
  }

  abort(): void {
    this.stream.abort();
  }

  async toResponse(): Promise<StreamedChatCompletion> {
    return this.stream.toResponse();
  }

  async toText(): Promise<string> {
    return this.stream.toText();
  }
}

/**
 * Utility function to iterate over a stream and call a callback for each chunk.
 */
export async function consumeStream(
  stream: AsyncIterable<StreamChunk>,
  callback: (chunk: StreamChunk) => void | Promise<void>
): Promise<void> {
  for await (const chunk of stream) {
    await callback(chunk);
  }
}

/**
 * Utility function to collect all text from a stream.
 */
export async function streamToText(stream: AsyncIterable<StreamChunk>): Promise<string> {
  let text = '';

  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content;
    if (content) {
      text += content;
    }
  }

  return text;
}

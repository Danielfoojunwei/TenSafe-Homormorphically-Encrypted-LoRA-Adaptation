/**
 * TenSafe SDK Client
 *
 * This module provides the main TenSafeClient class for interacting with
 * the TenSafe API for homomorphically encrypted LoRA fine-tuning.
 */

import type {
  TenSafeClientOptions,
  RequestOptions,
  TrainingConfig,
  TrainingClientInfo,
  CreateTrainingClientResponse,
  FutureResponse,
  FutureResultResponse,
  ForwardBackwardRequest,
  ForwardBackwardResult,
  OptimStepRequest,
  OptimStepResult,
  SampleRequest,
  SampleResult,
  SaveStateRequest,
  SaveStateResult,
  LoadStateRequest,
  LoadStateResult,
  BatchData,
  DPMetrics,
  AuditLogEntry,
  ChatCompletionRequest,
  ChatCompletionResponse,
  CompletionRequest,
  CompletionResponse,
  FutureStatus,
  OperationType,
  TrainingClientStatus,
  StreamChunk,
} from './types.js';

import {
  TenSafeError,
  AuthenticationError,
  RateLimitError,
  QuotaExceededError,
  ValidationError,
  PermissionDeniedError,
  TrainingClientNotFoundError,
  FutureNotFoundError,
  ArtifactNotFoundError,
  FutureTimeoutError,
  FutureCancelledError,
  FutureFailedError,
  QueueFullError,
  ServerError,
  ConnectionError,
  RequestAbortedError,
  isRetryableError,
} from './errors.js';

import { Stream, StreamingRequestBuilder } from './streaming.js';

// Re-export types and errors for convenience
export * from './types.js';
export * from './errors.js';
export * from './streaming.js';

/**
 * Default configuration values.
 */
const DEFAULT_BASE_URL = 'https://api.tensafe.io';
const DEFAULT_TIMEOUT = 60000; // 60 seconds
const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_RETRY_DELAY = 1000; // 1 second
const DEFAULT_POLL_INTERVAL = 1000; // 1 second
const SDK_VERSION = '1.0.0';

/**
 * Utility to convert camelCase to snake_case for API requests.
 */
function toSnakeCase(obj: unknown): unknown {
  if (obj === null || obj === undefined) {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(toSnakeCase);
  }

  if (typeof obj === 'object') {
    const converted: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(obj)) {
      const snakeKey = key.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
      converted[snakeKey] = toSnakeCase(value);
    }
    return converted;
  }

  return obj;
}

/**
 * Utility to convert snake_case to camelCase for API responses.
 */
function toCamelCase(obj: unknown): unknown {
  if (obj === null || obj === undefined) {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(toCamelCase);
  }

  if (typeof obj === 'object') {
    const converted: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(obj)) {
      const camelKey = key.replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase());
      converted[camelKey] = toCamelCase(value);
    }
    return converted;
  }

  return obj;
}

/**
 * Sleep utility.
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Handle for an asynchronous operation.
 *
 * FutureHandle represents a pending or completed async operation.
 * Use `result()` to wait for and retrieve the result, or `status()`
 * to check the current state without blocking.
 *
 * @example
 * ```typescript
 * const future = trainingClient.forwardBackward(batch);
 * console.log(await future.status()); // 'pending'
 * const result = await future.result(); // Blocks until complete
 * console.log(result.loss);
 * ```
 */
export class Future<T> {
  private _futureId: string;
  private _client: TenSafeClient;
  private _trainingClientId: string;
  private _operation: OperationType;
  private _pollInterval: number;
  private _status: FutureStatus = 'pending' as FutureStatus;
  private _result: T | null = null;
  private _error: string | null = null;
  private _cached = false;

  constructor(
    futureId: string,
    client: TenSafeClient,
    trainingClientId: string,
    operation: OperationType,
    pollInterval: number
  ) {
    this._futureId = futureId;
    this._client = client;
    this._trainingClientId = trainingClientId;
    this._operation = operation;
    this._pollInterval = pollInterval;
  }

  /** Unique identifier for this future. */
  get futureId(): string {
    return this._futureId;
  }

  /** ID of the associated training client. */
  get trainingClientId(): string {
    return this._trainingClientId;
  }

  /** Type of operation this future represents. */
  get operation(): OperationType {
    return this._operation;
  }

  /**
   * Get the current status of the future.
   *
   * @param refresh - If true, poll the server for latest status
   * @returns Current FutureStatus
   */
  async status(refresh = true): Promise<FutureStatus> {
    if (this._cached && !refresh) {
      return this._status;
    }

    if (this._isTerminal(this._status)) {
      return this._status;
    }

    const response = await this._client._getFutureStatus(this._futureId);
    this._status = response.status;

    if (this._isTerminal(this._status)) {
      this._cached = true;
    }

    return this._status;
  }

  /**
   * Check if the future is complete.
   */
  async done(): Promise<boolean> {
    const currentStatus = await this.status(true);
    return this._isTerminal(currentStatus);
  }

  /**
   * Wait for and return the result of the operation.
   *
   * @param timeout - Maximum time to wait in milliseconds (null for indefinite)
   * @throws FutureTimeoutError if timeout is reached
   * @throws FutureCancelledError if the future was cancelled
   * @throws FutureFailedError if the operation failed
   */
  async result(timeout?: number | null): Promise<T> {
    const startTime = Date.now();

    while (true) {
      const currentStatus = await this.status(true);

      if (currentStatus === ('completed' as FutureStatus)) {
        if (this._result === null) {
          const resultResponse = await this._client._getFutureResult(this._futureId);
          this._result = resultResponse.result as T;
        }
        return this._result as T;
      }

      if (currentStatus === ('failed' as FutureStatus)) {
        if (this._error === null) {
          const resultResponse = await this._client._getFutureResult(this._futureId);
          this._error = resultResponse.error ?? 'Unknown error';
        }
        throw new FutureFailedError(this._futureId, this._error);
      }

      if (currentStatus === ('cancelled' as FutureStatus)) {
        throw new FutureCancelledError(this._futureId);
      }

      if (timeout !== undefined && timeout !== null) {
        const elapsed = Date.now() - startTime;
        if (elapsed >= timeout) {
          throw new FutureTimeoutError(this._futureId, timeout);
        }
        await sleep(Math.min(this._pollInterval, timeout - elapsed));
      } else {
        await sleep(this._pollInterval);
      }
    }
  }

  /**
   * Attempt to cancel the future.
   *
   * @returns true if cancellation was successful
   */
  async cancel(): Promise<boolean> {
    if (this._isTerminal(this._status)) {
      return this._status === ('cancelled' as FutureStatus);
    }

    const success = await this._client._cancelFuture(this._futureId);
    if (success) {
      this._status = 'cancelled' as FutureStatus;
      this._cached = true;
    }
    return success;
  }

  private _isTerminal(status: FutureStatus): boolean {
    return ['completed', 'failed', 'cancelled'].includes(status);
  }
}

/**
 * Client for controlling a training loop.
 *
 * TrainingClient exposes primitives for fine-tuning models:
 * - forwardBackward: Compute loss and gradients
 * - optimStep: Apply optimizer update
 * - sample: Generate text from current model
 * - saveState: Save encrypted checkpoint
 * - loadState: Load checkpoint
 *
 * @example
 * ```typescript
 * const tc = await client.createTrainingClient(config);
 * for (const batch of dataloader) {
 *   const fbFuture = tc.forwardBackward(batch);
 *   const optFuture = tc.optimStep();
 *   const result = await fbFuture.result();
 *   console.log(`Loss: ${result.loss}`);
 * }
 * ```
 */
export class TrainingClient {
  private _id: string;
  private _client: TenSafeClient;
  private _config: TrainingConfig;
  private _step: number;
  private _status: TrainingClientStatus;
  private _dpMetrics: DPMetrics | null;

  constructor(
    trainingClientId: string,
    client: TenSafeClient,
    config: TrainingConfig,
    step: number = 0,
    dpMetrics: DPMetrics | null = null
  ) {
    this._id = trainingClientId;
    this._client = client;
    this._config = config;
    this._step = step;
    this._status = TrainingClientStatus.READY;
    this._dpMetrics = dpMetrics;
  }

  /** Unique identifier for this training client. */
  get id(): string {
    return this._id;
  }

  /** Alias for id property. */
  get trainingClientId(): string {
    return this._id;
  }

  /** Training configuration. */
  get config(): TrainingConfig {
    return this._config;
  }

  /** Current training step. */
  get step(): number {
    return this._step;
  }

  /** Current status. */
  get status(): TrainingClientStatus {
    return this._status;
  }

  /** Whether differential privacy is enabled. */
  get dpEnabled(): boolean {
    return this._config.dpConfig?.enabled ?? false;
  }

  /** Current differential privacy metrics. */
  get dpMetrics(): DPMetrics | null {
    return this._dpMetrics;
  }

  /**
   * Queue a forward-backward pass computation.
   *
   * @param batch - Training batch data
   * @param batchHash - Optional client-side hash for verification
   * @returns Future handle for the async operation
   */
  forwardBackward(
    batch: BatchData | { inputIds: number[][]; attentionMask: number[][]; labels?: number[][] },
    batchHash?: string
  ): Future<ForwardBackwardResult> {
    const request: ForwardBackwardRequest = {
      batch: batch as BatchData,
      batchHash: batchHash ?? null,
    };

    const responsePromise = this._client._postForwardBackward(this._id, request);

    // Return a future that will be resolved when the operation completes
    return this._createFuture<ForwardBackwardResult>(responsePromise, OperationType.FORWARD_BACKWARD);
  }

  /**
   * Queue an optimizer step.
   *
   * @param applyDpNoise - If true and DP is enabled, apply DP noise
   * @returns Future handle for the async operation
   */
  optimStep(applyDpNoise = true): Future<OptimStepResult> {
    const request: OptimStepRequest = { applyDpNoise };
    const responsePromise = this._client._postOptimStep(this._id, request);
    return this._createFuture<OptimStepResult>(responsePromise, OperationType.OPTIM_STEP);
  }

  /**
   * Generate samples from the current model state.
   *
   * @param prompts - Single prompt or array of prompts
   * @param options - Sampling options
   * @returns Sample results
   */
  async sample(
    prompts: string | string[],
    options: {
      maxTokens?: number;
      temperature?: number;
      topP?: number;
      topK?: number;
      stopSequences?: string[];
    } = {}
  ): Promise<SampleResult> {
    const request: SampleRequest = {
      prompts: Array.isArray(prompts) ? prompts : [prompts],
      maxTokens: options.maxTokens ?? 128,
      temperature: options.temperature ?? 0.7,
      topP: options.topP ?? 0.9,
      topK: options.topK ?? 50,
      stopSequences: options.stopSequences ?? [],
    };

    return this._client._postSample(this._id, request);
  }

  /**
   * Save the current training state as an encrypted checkpoint.
   *
   * @param options - Save options
   * @returns Save result with artifact information
   */
  async saveState(
    options: {
      includeOptimizer?: boolean;
      metadata?: Record<string, unknown>;
    } = {}
  ): Promise<SaveStateResult> {
    const request: SaveStateRequest = {
      includeOptimizer: options.includeOptimizer ?? true,
      metadata: options.metadata ?? {},
    };

    return this._client._postSaveState(this._id, request);
  }

  /**
   * Load training state from an encrypted checkpoint.
   *
   * @param artifactId - ID of the checkpoint artifact to load
   * @returns Load result with updated state
   */
  async loadState(artifactId: string): Promise<LoadStateResult> {
    const request: LoadStateRequest = { artifactId };
    const result = await this._client._postLoadState(this._id, request);

    // Update local state
    this._step = result.step;
    this._status = result.status;

    return result;
  }

  /**
   * Refresh the training client state from the server.
   */
  async refresh(): Promise<TrainingClient> {
    const info = await this._client.getTrainingClient(this._id);
    this._step = info.step;
    this._status = info.status;
    this._dpMetrics = info.dpMetrics ?? null;
    return this;
  }

  private _createFuture<T>(
    responsePromise: Promise<FutureResponse>,
    operation: OperationType
  ): Future<T> {
    // We need to handle this synchronously for the API to feel right
    // The future will internally await the response when needed
    const future = new LazyFuture<T>(
      responsePromise,
      this._client,
      this._id,
      operation,
      this._client['_config'].pollInterval ?? DEFAULT_POLL_INTERVAL
    );
    return future as Future<T>;
  }
}

/**
 * A future that lazily initializes from a promise.
 */
class LazyFuture<T> extends Future<T> {
  private _responsePromise: Promise<FutureResponse>;
  private _initialized = false;
  private _initPromise: Promise<void> | null = null;

  constructor(
    responsePromise: Promise<FutureResponse>,
    client: TenSafeClient,
    trainingClientId: string,
    operation: OperationType,
    pollInterval: number
  ) {
    // Initialize with placeholder values
    super('', client, trainingClientId, operation, pollInterval);
    this._responsePromise = responsePromise;
  }

  private async _ensureInitialized(): Promise<void> {
    if (this._initialized) return;

    if (!this._initPromise) {
      this._initPromise = (async () => {
        const response = await this._responsePromise;
        // Use Object.defineProperty to set the readonly futureId
        Object.defineProperty(this, '_futureId', { value: response.futureId });
        this._initialized = true;
      })();
    }

    await this._initPromise;
  }

  override async status(refresh = true): Promise<FutureStatus> {
    await this._ensureInitialized();
    return super.status(refresh);
  }

  override async done(): Promise<boolean> {
    await this._ensureInitialized();
    return super.done();
  }

  override async result(timeout?: number | null): Promise<T> {
    await this._ensureInitialized();
    return super.result(timeout);
  }

  override async cancel(): Promise<boolean> {
    await this._ensureInitialized();
    return super.cancel();
  }

  override get futureId(): string {
    // Return empty string if not initialized yet
    return (this as unknown as { _futureId: string })._futureId || '';
  }
}

/**
 * Main client for interacting with the TenSafe API.
 *
 * TenSafeClient provides methods for:
 * - Creating and managing training clients
 * - Running inference with HE-LoRA adapters
 * - Managing artifacts and audit logs
 *
 * @example
 * ```typescript
 * import TenSafe from '@tensafe/sdk';
 *
 * const client = new TenSafe({
 *   apiKey: process.env.TENSAFE_API_KEY,
 * });
 *
 * // Create a training client
 * const tc = await client.createTrainingClient({
 *   modelRef: 'meta-llama/Llama-3-8B',
 *   loraConfig: { rank: 16, alpha: 32 },
 *   dpConfig: { enabled: true, targetEpsilon: 8.0 },
 * });
 *
 * // Run training
 * const future = tc.forwardBackward(batch);
 * const result = await future.result();
 * ```
 */
export class TenSafeClient {
  private _config: Required<
    Pick<TenSafeClientOptions, 'baseUrl' | 'timeout' | 'maxRetries' | 'retryDelay' | 'pollInterval'>
  > & {
    apiKey: string;
    tenantId?: string;
    headers: Record<string, string>;
  };

  private _streamBuilder: StreamingRequestBuilder;

  /**
   * Chat completions API (OpenAI-compatible).
   */
  readonly chat: {
    completions: {
      create(request: ChatCompletionRequest & { stream?: false }, options?: RequestOptions): Promise<ChatCompletionResponse>;
      create(request: ChatCompletionRequest & { stream: true }, options?: RequestOptions): Promise<Stream>;
      create(request: ChatCompletionRequest, options?: RequestOptions): Promise<ChatCompletionResponse | Stream>;
    };
  };

  /**
   * Completions API (OpenAI-compatible).
   */
  readonly completions: {
    create(request: CompletionRequest, options?: RequestOptions): Promise<CompletionResponse>;
  };

  constructor(options: TenSafeClientOptions = {}) {
    const apiKey = options.apiKey ?? process.env['TENSAFE_API_KEY'];
    if (!apiKey) {
      throw new AuthenticationError(
        'API key is required. Pass it via options.apiKey or set TENSAFE_API_KEY environment variable.'
      );
    }

    this._config = {
      apiKey,
      baseUrl: options.baseUrl ?? process.env['TENSAFE_BASE_URL'] ?? DEFAULT_BASE_URL,
      tenantId: options.tenantId ?? process.env['TENSAFE_TENANT_ID'],
      timeout: options.timeout ?? DEFAULT_TIMEOUT,
      maxRetries: options.maxRetries ?? DEFAULT_MAX_RETRIES,
      retryDelay: options.retryDelay ?? DEFAULT_RETRY_DELAY,
      pollInterval: options.pollInterval ?? DEFAULT_POLL_INTERVAL,
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'User-Agent': `tensafe-sdk-typescript/${SDK_VERSION}`,
        ...options.headers,
      },
    };

    this._streamBuilder = new StreamingRequestBuilder(
      `${this._config.baseUrl}/v1`,
      this._config.headers
    );

    // Initialize chat completions API
    this.chat = {
      completions: {
        create: async (request: ChatCompletionRequest, options?: RequestOptions) => {
          if (request.stream) {
            return this._streamBuilder.request(
              '/chat/completions',
              toSnakeCase(request),
              { signal: options?.signal }
            );
          }

          return this._request<ChatCompletionResponse>(
            'POST',
            '/chat/completions',
            toSnakeCase(request),
            options
          );
        },
      },
    };

    // Initialize completions API
    this.completions = {
      create: async (request: CompletionRequest, options?: RequestOptions) => {
        return this._request<CompletionResponse>(
          'POST',
          '/completions',
          toSnakeCase(request),
          options
        );
      },
    };
  }

  // ===========================================================================
  // Training Client Methods
  // ===========================================================================

  /**
   * Create a new training client.
   *
   * @param config - Training configuration
   * @returns A new TrainingClient instance
   */
  async createTrainingClient(config: TrainingConfig): Promise<TrainingClient> {
    const response = await this._request<CreateTrainingClientResponse>(
      'POST',
      '/training_clients',
      toSnakeCase(config)
    );

    return new TrainingClient(
      response.trainingClientId,
      this,
      response.config,
      response.step,
      null
    );
  }

  /**
   * Get information about a training client.
   *
   * @param trainingClientId - ID of the training client
   * @returns Training client information
   */
  async getTrainingClient(trainingClientId: string): Promise<TrainingClientInfo> {
    return this._request<TrainingClientInfo>('GET', `/training_clients/${trainingClientId}`);
  }

  /**
   * List all training clients for the tenant.
   *
   * @returns List of training client information
   */
  async listTrainingClients(): Promise<TrainingClientInfo[]> {
    return this._request<TrainingClientInfo[]>('GET', '/training_clients');
  }

  // ===========================================================================
  // Artifact Methods
  // ===========================================================================

  /**
   * Download an artifact's encrypted content.
   *
   * @param artifactId - ID of the artifact
   * @returns Encrypted artifact data as ArrayBuffer
   */
  async pullArtifact(artifactId: string): Promise<ArrayBuffer> {
    const response = await fetch(`${this._config.baseUrl}/v1/artifacts/${artifactId}/content`, {
      method: 'GET',
      headers: this._config.headers,
    });

    if (!response.ok) {
      await this._handleErrorResponse(response);
    }

    return response.arrayBuffer();
  }

  // ===========================================================================
  // Audit Log Methods
  // ===========================================================================

  /**
   * Retrieve audit logs.
   *
   * @param options - Filter options
   * @returns List of audit log entries
   */
  async getAuditLogs(options: {
    trainingClientId?: string;
    operation?: string;
    limit?: number;
    offset?: number;
  } = {}): Promise<AuditLogEntry[]> {
    const params = new URLSearchParams();
    if (options.trainingClientId) params.set('training_client_id', options.trainingClientId);
    if (options.operation) params.set('operation', options.operation);
    params.set('limit', String(options.limit ?? 100));
    params.set('offset', String(options.offset ?? 0));

    return this._request<AuditLogEntry[]>('GET', `/audit_logs?${params.toString()}`);
  }

  // ===========================================================================
  // Internal Methods (used by TrainingClient and Future)
  // ===========================================================================

  /** @internal */
  async _postForwardBackward(
    trainingClientId: string,
    request: ForwardBackwardRequest
  ): Promise<FutureResponse> {
    return this._request<FutureResponse>(
      'POST',
      `/training_clients/${trainingClientId}/forward_backward`,
      toSnakeCase(request)
    );
  }

  /** @internal */
  async _postOptimStep(
    trainingClientId: string,
    request: OptimStepRequest
  ): Promise<FutureResponse> {
    return this._request<FutureResponse>(
      'POST',
      `/training_clients/${trainingClientId}/optim_step`,
      toSnakeCase(request)
    );
  }

  /** @internal */
  async _postSample(trainingClientId: string, request: SampleRequest): Promise<SampleResult> {
    return this._request<SampleResult>(
      'POST',
      `/training_clients/${trainingClientId}/sample`,
      toSnakeCase(request)
    );
  }

  /** @internal */
  async _postSaveState(trainingClientId: string, request: SaveStateRequest): Promise<SaveStateResult> {
    return this._request<SaveStateResult>(
      'POST',
      `/training_clients/${trainingClientId}/save_state`,
      toSnakeCase(request)
    );
  }

  /** @internal */
  async _postLoadState(trainingClientId: string, request: LoadStateRequest): Promise<LoadStateResult> {
    return this._request<LoadStateResult>(
      'POST',
      `/training_clients/${trainingClientId}/load_state`,
      toSnakeCase(request)
    );
  }

  /** @internal */
  async _getFutureStatus(futureId: string): Promise<FutureResponse> {
    return this._request<FutureResponse>('GET', `/futures/${futureId}`);
  }

  /** @internal */
  async _getFutureResult(futureId: string): Promise<FutureResultResponse> {
    return this._request<FutureResultResponse>('GET', `/futures/${futureId}/result`);
  }

  /** @internal */
  async _cancelFuture(futureId: string): Promise<boolean> {
    try {
      await this._request('POST', `/futures/${futureId}/cancel`);
      return true;
    } catch {
      return false;
    }
  }

  // ===========================================================================
  // HTTP Request Handling
  // ===========================================================================

  private async _request<T>(
    method: string,
    path: string,
    body?: unknown,
    options?: RequestOptions
  ): Promise<T> {
    const url = `${this._config.baseUrl}/v1${path}`;
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= this._config.maxRetries; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(
          () => controller.abort(),
          options?.timeout ?? this._config.timeout
        );

        // Combine with user-provided signal
        if (options?.signal) {
          options.signal.addEventListener('abort', () => controller.abort());
        }

        const response = await fetch(url, {
          method,
          headers: {
            ...this._config.headers,
            ...options?.headers,
          },
          body: body ? JSON.stringify(body) : undefined,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          await this._handleErrorResponse(response);
        }

        const json = await response.json();
        return toCamelCase(json) as T;
      } catch (error) {
        lastError = error as Error;

        if (error instanceof Error && error.name === 'AbortError') {
          if (options?.signal?.aborted) {
            throw new RequestAbortedError();
          }
          throw new ConnectionError('Request timed out');
        }

        if (isRetryableError(error) && attempt < this._config.maxRetries) {
          const delay = error instanceof RateLimitError && error.retryAfter
            ? Math.min(error.retryAfter * 1000, 60000)
            : this._config.retryDelay * Math.pow(2, attempt);
          await sleep(delay);
          continue;
        }

        throw error;
      }
    }

    throw lastError ?? new TenSafeError('Request failed');
  }

  private async _handleErrorResponse(response: Response): Promise<never> {
    let errorData: { error?: { code?: string; message?: string; details?: Record<string, unknown>; request_id?: string } } | null = null;

    try {
      errorData = await response.json();
    } catch {
      // Ignore JSON parse errors
    }

    const code = errorData?.error?.code ?? 'UNKNOWN_ERROR';
    const message = errorData?.error?.message ?? response.statusText;
    const details = errorData?.error?.details ?? {};
    const requestId = errorData?.error?.request_id;

    switch (response.status) {
      case 401:
        throw new AuthenticationError(message, { details, requestId });

      case 403:
        if (code.toLowerCase().includes('quota') || code.toLowerCase().includes('budget')) {
          throw new QuotaExceededError(message, { requestId });
        }
        throw new PermissionDeniedError(message, { details, requestId });

      case 404:
        if (code.toLowerCase().includes('training_client')) {
          throw new TrainingClientNotFoundError(
            (details['training_client_id'] as string) ?? 'unknown',
            { requestId }
          );
        }
        if (code.toLowerCase().includes('future')) {
          throw new FutureNotFoundError(
            (details['future_id'] as string) ?? 'unknown',
            { requestId }
          );
        }
        if (code.toLowerCase().includes('artifact')) {
          throw new ArtifactNotFoundError(
            (details['artifact_id'] as string) ?? 'unknown',
            { requestId }
          );
        }
        throw new TenSafeError(message, { code, details, requestId, status: 404 });

      case 422:
        throw new ValidationError(message, { details, requestId });

      case 429:
        const retryAfter = response.headers.get('Retry-After');
        throw new RateLimitError(message, {
          retryAfter: retryAfter ? parseInt(retryAfter, 10) : undefined,
          requestId,
        });

      case 503:
        if (code.toLowerCase().includes('queue')) {
          throw new QueueFullError({ requestId });
        }
        throw new ServerError(message, { details, requestId });

      default:
        if (response.status >= 500) {
          throw new ServerError(message, { details, requestId });
        }
        throw new TenSafeError(message, { code, details, requestId, status: response.status });
    }
  }
}

/**
 * Default export for convenient importing.
 *
 * @example
 * ```typescript
 * import TenSafe from '@tensafe/sdk';
 *
 * const client = new TenSafe({ apiKey: 'your-api-key' });
 * ```
 */
export default TenSafeClient;

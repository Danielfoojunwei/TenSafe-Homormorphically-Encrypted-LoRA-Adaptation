/**
 * TenSafe SDK TypeScript Type Definitions
 *
 * This module defines all TypeScript interfaces and types used throughout the SDK.
 */

// =============================================================================
// Enums
// =============================================================================

/**
 * Status of an asynchronous future operation.
 */
export enum FutureStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

/**
 * Types of operations that can be performed.
 */
export enum OperationType {
  FORWARD_BACKWARD = 'forward_backward',
  OPTIM_STEP = 'optim_step',
  SAMPLE = 'sample',
  SAVE_STATE = 'save_state',
  LOAD_STATE = 'load_state',
}

/**
 * Types of differential privacy accountants.
 */
export enum DPAccountantType {
  RDP = 'rdp',
  MOMENTS = 'moments',
  PRV = 'prv',
}

/**
 * Status of a training client.
 */
export enum TrainingClientStatus {
  INITIALIZING = 'initializing',
  READY = 'ready',
  BUSY = 'busy',
  ERROR = 'error',
  TERMINATED = 'terminated',
}

/**
 * LoRA target projection types.
 */
export enum LoRATargetType {
  QKV = 'qkv',
  QKVO = 'qkvo',
}

/**
 * Insertion point types for HE-LoRA.
 */
export enum InsertionPointType {
  PRE_PROJECTION = 'pre_projection',
  POST_PROJECTION = 'post_projection',
}

/**
 * Layer selection modes for LoRA application.
 */
export enum LayerSelectionMode {
  ALL = 'all',
  RANGE = 'range',
  LIST = 'list',
  PATTERN = 'pattern',
}

// =============================================================================
// Configuration Types
// =============================================================================

/**
 * Configuration for LoRA (Low-Rank Adaptation) fine-tuning.
 */
export interface LoRAConfig {
  /** LoRA rank (1-512) */
  rank?: number;
  /** LoRA alpha scaling factor */
  alpha?: number;
  /** Dropout rate (0.0-1.0) */
  dropout?: number;
  /** Target modules for LoRA */
  targetModules?: string[];
  /** Bias handling: 'none', 'all', or 'lora_only' */
  bias?: 'none' | 'all' | 'lora_only';
}

/**
 * Configuration for the optimizer.
 */
export interface OptimizerConfig {
  /** Optimizer name: 'adamw', 'adam', 'sgd', 'adafactor' */
  name?: string;
  /** Learning rate */
  learningRate?: number;
  /** Weight decay coefficient */
  weightDecay?: number;
  /** Adam beta parameters [beta1, beta2] */
  betas?: [number, number];
  /** Epsilon for numerical stability */
  eps?: number;
}

/**
 * Configuration for Differential Privacy.
 */
export interface DPConfig {
  /** Enable differential privacy */
  enabled?: boolean;
  /** Gaussian noise multiplier */
  noiseMultiplier?: number;
  /** Maximum gradient norm for clipping */
  maxGradNorm?: number;
  /** Target epsilon budget */
  targetEpsilon?: number;
  /** Target delta value */
  targetDelta?: number;
  /** Privacy accountant type */
  accountantType?: DPAccountantType;
}

/**
 * Configuration for text sampling/generation.
 */
export interface SamplingConfig {
  /** Maximum tokens to generate */
  maxTokens?: number;
  /** Sampling temperature */
  temperature?: number;
  /** Nucleus sampling threshold */
  topP?: number;
  /** Top-k sampling value */
  topK?: number;
  /** Stop sequences */
  stopSequences?: string[];
}

/**
 * Full training configuration.
 */
export interface TrainingConfig {
  /** Model reference (HuggingFace hub or local path) */
  modelRef: string;
  /** LoRA configuration (null for full fine-tuning) */
  loraConfig?: LoRAConfig | null;
  /** Optimizer configuration */
  optimizer?: OptimizerConfig;
  /** Differential privacy configuration */
  dpConfig?: DPConfig | null;
  /** Batch size */
  batchSize?: number;
  /** Gradient accumulation steps */
  gradientAccumulationSteps?: number;
  /** Maximum training steps */
  maxSteps?: number | null;
  /** Custom metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Layer selection configuration.
 */
export interface LayerSelection {
  /** Selection mode */
  mode?: LayerSelectionMode;
  /** Start layer for range mode */
  start?: number | null;
  /** End layer for range mode (exclusive) */
  end?: number | null;
  /** Explicit list for list mode */
  layers?: number[] | null;
  /** Pattern name for pattern mode */
  pattern?: string | null;
  /** Pattern argument */
  patternArg?: number | null;
}

/**
 * HE-LoRA insertion point configuration.
 */
export interface HELoRAConfig {
  /** Adapter identifier */
  adapterId: string;
  /** Target projections */
  targets?: LoRATargetType;
  /** Layer selection configuration */
  layerSelection?: LayerSelection;
  /** Insertion point type */
  insertionPoint?: InsertionPointType;
  /** Per-layer target overrides */
  perLayerConfig?: Record<number, LoRATargetType> | null;
  /** Whether HE-LoRA is enabled */
  enabled?: boolean;
}

// =============================================================================
// Request Types
// =============================================================================

/**
 * Training batch data.
 */
export interface BatchData {
  /** Input token IDs */
  inputIds: number[][];
  /** Attention mask */
  attentionMask: number[][];
  /** Label token IDs (optional) */
  labels?: number[][] | null;
}

/**
 * Request for forward-backward pass.
 */
export interface ForwardBackwardRequest {
  /** Training batch data */
  batch: BatchData;
  /** Client-side hash for verification */
  batchHash?: string | null;
}

/**
 * Request for optimizer step.
 */
export interface OptimStepRequest {
  /** Apply DP noise if enabled */
  applyDpNoise?: boolean;
}

/**
 * Request for text sampling.
 */
export interface SampleRequest {
  /** Prompts to sample from */
  prompts: string[];
  /** Maximum tokens to generate */
  maxTokens?: number;
  /** Sampling temperature */
  temperature?: number;
  /** Nucleus sampling threshold */
  topP?: number;
  /** Top-k sampling */
  topK?: number;
  /** Stop sequences */
  stopSequences?: string[];
}

/**
 * Request to save training state.
 */
export interface SaveStateRequest {
  /** Include optimizer state in checkpoint */
  includeOptimizer?: boolean;
  /** Custom metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Request to load training state.
 */
export interface LoadStateRequest {
  /** Artifact ID to load */
  artifactId: string;
}

/**
 * Chat message for inference.
 */
export interface ChatMessage {
  /** Message role: 'system', 'user', or 'assistant' */
  role: 'system' | 'user' | 'assistant';
  /** Message content */
  content: string;
  /** Optional name */
  name?: string;
}

/**
 * Chat completion request.
 */
export interface ChatCompletionRequest {
  /** Model identifier */
  model: string;
  /** Conversation messages */
  messages: ChatMessage[];
  /** Maximum tokens to generate */
  maxTokens?: number | null;
  /** Sampling temperature */
  temperature?: number;
  /** Nucleus sampling threshold */
  topP?: number;
  /** Number of completions */
  n?: number;
  /** Enable streaming */
  stream?: boolean;
  /** Stop sequences */
  stop?: string | string[] | null;
  /** Presence penalty */
  presencePenalty?: number;
  /** Frequency penalty */
  frequencyPenalty?: number;
  /** Logit bias */
  logitBias?: Record<string, number> | null;
  /** User identifier */
  user?: string | null;
  /** HE-LoRA configuration */
  heloraConfig?: HELoRAConfig | null;
}

/**
 * Completion request.
 */
export interface CompletionRequest {
  /** Model identifier */
  model: string;
  /** Prompt text */
  prompt: string | string[] | number[] | number[][];
  /** Maximum tokens to generate */
  maxTokens?: number;
  /** Sampling temperature */
  temperature?: number;
  /** Nucleus sampling threshold */
  topP?: number;
  /** Number of completions */
  n?: number;
  /** Enable streaming */
  stream?: boolean;
  /** Log probabilities */
  logprobs?: number | null;
  /** Echo prompt in response */
  echo?: boolean;
  /** Stop sequences */
  stop?: string | string[] | null;
  /** Presence penalty */
  presencePenalty?: number;
  /** Frequency penalty */
  frequencyPenalty?: number;
  /** Best of N */
  bestOf?: number;
  /** Logit bias */
  logitBias?: Record<string, number> | null;
  /** User identifier */
  user?: string | null;
  /** HE-LoRA configuration */
  heloraConfig?: HELoRAConfig | null;
}

// =============================================================================
// Response Types
// =============================================================================

/**
 * Differential privacy metrics.
 */
export interface DPMetrics {
  /** Whether noise was applied */
  noiseApplied: boolean;
  /** Epsilon spent this step */
  epsilonSpent: number;
  /** Total epsilon spent */
  totalEpsilon: number;
  /** Delta value */
  delta: number;
  /** Gradient norm before clipping */
  gradNormBeforeClip?: number | null;
  /** Gradient norm after clipping */
  gradNormAfterClip?: number | null;
  /** Number of clipped gradients */
  numClipped?: number | null;
}

/**
 * Information about a training client.
 */
export interface TrainingClientInfo {
  /** Training client ID */
  trainingClientId: string;
  /** Tenant ID */
  tenantId: string;
  /** Model reference */
  modelRef: string;
  /** Current status */
  status: TrainingClientStatus;
  /** Current training step */
  step: number;
  /** Creation timestamp */
  createdAt: string;
  /** Training configuration */
  config: TrainingConfig;
  /** DP metrics (if enabled) */
  dpMetrics?: DPMetrics | null;
}

/**
 * Response from creating a training client.
 */
export interface CreateTrainingClientResponse {
  /** Training client ID */
  trainingClientId: string;
  /** Tenant ID */
  tenantId: string;
  /** Model reference */
  modelRef: string;
  /** Status */
  status: TrainingClientStatus;
  /** Current step */
  step: number;
  /** Creation timestamp */
  createdAt: string;
  /** Training configuration */
  config: TrainingConfig;
}

/**
 * Future operation response.
 */
export interface FutureResponse {
  /** Future ID */
  futureId: string;
  /** Current status */
  status: FutureStatus;
  /** Creation timestamp */
  createdAt: string;
  /** Training client ID */
  trainingClientId: string;
  /** Operation type */
  operation: OperationType;
  /** Start timestamp */
  startedAt?: string | null;
  /** Completion timestamp */
  completedAt?: string | null;
}

/**
 * Result of forward-backward pass.
 */
export interface ForwardBackwardResult {
  /** Training loss */
  loss: number;
  /** Gradient norm */
  gradNorm: number;
  /** Tokens processed */
  tokensProcessed: number;
  /** DP metrics (if enabled) */
  dpMetrics?: DPMetrics | null;
}

/**
 * Result of optimizer step.
 */
export interface OptimStepResult {
  /** Updated training step */
  step: number;
  /** Current learning rate */
  learningRate: number;
  /** DP metrics (if enabled) */
  dpMetrics?: DPMetrics | null;
}

/**
 * A single sample completion.
 */
export interface SampleCompletion {
  /** Original prompt */
  prompt: string;
  /** Generated completion */
  completion: string;
  /** Number of tokens generated */
  tokensGenerated: number;
  /** Finish reason: 'stop', 'length', or 'error' */
  finishReason: string;
}

/**
 * Result of sampling operation.
 */
export interface SampleResult {
  /** Generated samples */
  samples: SampleCompletion[];
  /** Model step at sampling time */
  modelStep: number;
  /** Sampling configuration used */
  samplingConfig: SamplingConfig;
}

/**
 * Encryption information for artifacts.
 */
export interface EncryptionInfo {
  /** Encryption algorithm */
  algorithm: string;
  /** Key identifier */
  keyId: string;
}

/**
 * Result of save state operation.
 */
export interface SaveStateResult {
  /** Artifact ID */
  artifactId: string;
  /** Artifact type */
  artifactType: string;
  /** Size in bytes */
  sizeBytes: number;
  /** Encryption information */
  encryption: EncryptionInfo;
  /** Content hash */
  contentHash: string;
  /** Custom metadata */
  metadata: Record<string, unknown>;
  /** Creation timestamp */
  createdAt: string;
  /** DP metrics (if enabled) */
  dpMetrics?: DPMetrics | null;
}

/**
 * Result of load state operation.
 */
export interface LoadStateResult {
  /** Training client ID */
  trainingClientId: string;
  /** Loaded artifact ID */
  loadedArtifactId: string;
  /** Restored training step */
  step: number;
  /** Status after loading */
  status: TrainingClientStatus;
}

/**
 * Future result response.
 */
export interface FutureResultResponse {
  /** Future ID */
  futureId: string;
  /** Current status */
  status: FutureStatus;
  /** Operation result (type depends on operation) */
  result?: ForwardBackwardResult | OptimStepResult | SampleResult | SaveStateResult | LoadStateResult | null;
  /** Error message (if failed) */
  error?: string | null;
}

/**
 * Chat completion choice.
 */
export interface ChatChoice {
  /** Response message */
  message: ChatMessage;
  /** Choice index */
  index: number;
  /** Finish reason */
  finishReason: string | null;
}

/**
 * Completion choice.
 */
export interface Choice {
  /** Generated text */
  text: string;
  /** Choice index */
  index: number;
  /** Log probabilities */
  logprobs: unknown | null;
  /** Finish reason */
  finishReason: string | null;
}

/**
 * Token usage statistics.
 */
export interface Usage {
  /** Tokens in prompt */
  promptTokens: number;
  /** Tokens in completion */
  completionTokens: number;
  /** Total tokens */
  totalTokens: number;
  /** HE-LoRA statistics */
  heloraStats?: Record<string, unknown> | null;
}

/**
 * Chat completion response.
 */
export interface ChatCompletionResponse {
  /** Response ID */
  id: string;
  /** Object type */
  object: string;
  /** Creation timestamp (Unix) */
  created: number;
  /** Model used */
  model: string;
  /** Response choices */
  choices: ChatChoice[];
  /** Token usage */
  usage?: Usage | null;
}

/**
 * Completion response.
 */
export interface CompletionResponse {
  /** Response ID */
  id: string;
  /** Object type */
  object: string;
  /** Creation timestamp (Unix) */
  created: number;
  /** Model used */
  model: string;
  /** Response choices */
  choices: Choice[];
  /** Token usage */
  usage?: Usage | null;
}

/**
 * Streaming chunk delta.
 */
export interface StreamDelta {
  /** Role (for first chunk) */
  role?: string;
  /** Content chunk */
  content?: string;
}

/**
 * Streaming choice.
 */
export interface StreamChoice {
  /** Delta content */
  delta: StreamDelta;
  /** Choice index */
  index: number;
  /** Finish reason (for last chunk) */
  finishReason: string | null;
}

/**
 * Streaming chunk response.
 */
export interface StreamChunk {
  /** Chunk ID */
  id: string;
  /** Object type */
  object: string;
  /** Creation timestamp (Unix) */
  created: number;
  /** Model used */
  model: string;
  /** Streaming choices */
  choices: StreamChoice[];
}

/**
 * Artifact information.
 */
export interface Artifact {
  /** Artifact ID */
  artifactId: string;
  /** Artifact type */
  artifactType: string;
  /** Size in bytes */
  sizeBytes: number;
  /** Encryption information */
  encryption: EncryptionInfo;
  /** Content hash */
  contentHash: string;
  /** Custom metadata */
  metadata: Record<string, unknown>;
  /** Creation timestamp */
  createdAt: string;
  /** Tenant ID */
  tenantId: string;
  /** Training client ID */
  trainingClientId: string;
}

/**
 * Audit log entry.
 */
export interface AuditLogEntry {
  /** Entry ID */
  entryId: string;
  /** Tenant ID */
  tenantId: string;
  /** Training client ID */
  trainingClientId: string;
  /** Operation type */
  operation: OperationType;
  /** Request hash */
  requestHash: string;
  /** Request size in bytes */
  requestSizeBytes: number;
  /** Artifact IDs produced */
  artifactIdsProduced: string[];
  /** Artifact IDs consumed */
  artifactIdsConsumed: string[];
  /** Start timestamp */
  startedAt: string;
  /** Completion timestamp */
  completedAt?: string | null;
  /** Duration in milliseconds */
  durationMs?: number | null;
  /** Success flag */
  success: boolean;
  /** Error code */
  errorCode?: string | null;
  /** Error message */
  errorMessage?: string | null;
  /** Previous hash */
  prevHash: string;
  /** Record hash */
  recordHash: string;
  /** DP metrics */
  dpMetrics?: DPMetrics | null;
}

/**
 * Error response detail.
 */
export interface ErrorDetail {
  /** Error code */
  code: string;
  /** Error message */
  message: string;
  /** Additional details */
  details: Record<string, unknown>;
  /** Request ID */
  requestId?: string | null;
}

/**
 * Error response wrapper.
 */
export interface ErrorResponse {
  /** Error details */
  error: ErrorDetail;
}

// =============================================================================
// Client Configuration Types
// =============================================================================

/**
 * TenSafe client configuration options.
 */
export interface TenSafeClientOptions {
  /** API base URL */
  baseUrl?: string;
  /** API key for authentication */
  apiKey?: string;
  /** Tenant ID */
  tenantId?: string;
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Number of retries on failure */
  maxRetries?: number;
  /** Delay between retries in milliseconds */
  retryDelay?: number;
  /** Polling interval for futures in milliseconds */
  pollInterval?: number;
  /** Custom headers */
  headers?: Record<string, string>;
}

/**
 * Request options for individual API calls.
 */
export interface RequestOptions {
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Custom headers */
  headers?: Record<string, string>;
  /** Abort signal */
  signal?: AbortSignal;
}

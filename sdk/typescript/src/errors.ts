/**
 * TenSafe SDK Error Classes
 *
 * This module defines all custom error classes used throughout the SDK.
 * Error handling follows patterns similar to the OpenAI SDK for familiarity.
 */

/**
 * Base error class for all TenSafe SDK errors.
 */
export class TenSafeError extends Error {
  /** Error code for programmatic handling */
  readonly code: string;
  /** Additional error details */
  readonly details: Record<string, unknown>;
  /** Request ID for debugging */
  readonly requestId?: string;
  /** HTTP status code (if applicable) */
  readonly status?: number;

  constructor(
    message: string,
    options: {
      code?: string;
      details?: Record<string, unknown>;
      requestId?: string;
      status?: number;
      cause?: Error;
    } = {}
  ) {
    super(message, { cause: options.cause });
    this.name = 'TenSafeError';
    this.code = options.code ?? 'TENSAFE_ERROR';
    this.details = options.details ?? {};
    this.requestId = options.requestId;
    this.status = options.status;

    // Maintains proper stack trace in V8 environments
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /**
   * Returns a string representation of the error.
   */
  override toString(): string {
    const parts = [`[${this.code}] ${this.message}`];
    if (this.requestId) {
      parts.push(`(request_id: ${this.requestId})`);
    }
    return parts.join(' ');
  }

  /**
   * Converts the error to a plain object for serialization.
   */
  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      details: this.details,
      requestId: this.requestId,
      status: this.status,
    };
  }
}

/**
 * Error raised when API authentication fails.
 * This typically indicates an invalid or expired API key.
 */
export class AuthenticationError extends TenSafeError {
  constructor(
    message = 'Authentication failed. Please check your API key.',
    options: {
      details?: Record<string, unknown>;
      requestId?: string;
    } = {}
  ) {
    super(message, {
      code: 'AUTHENTICATION_REQUIRED',
      details: options.details,
      requestId: options.requestId,
      status: 401,
    });
    this.name = 'AuthenticationError';
  }
}

/**
 * Error raised when the API rate limit is exceeded.
 * Check the retryAfter property for when to retry.
 */
export class RateLimitError extends TenSafeError {
  /** Seconds to wait before retrying */
  readonly retryAfter?: number;

  constructor(
    message = 'Rate limit exceeded. Please slow down your requests.',
    options: {
      retryAfter?: number;
      requestId?: string;
    } = {}
  ) {
    super(message, {
      code: 'RATE_LIMITED',
      details: options.retryAfter ? { retryAfter: options.retryAfter } : {},
      requestId: options.requestId,
      status: 429,
    });
    this.name = 'RateLimitError';
    this.retryAfter = options.retryAfter;
  }
}

/**
 * Error raised when the quota (e.g., DP budget) is exceeded.
 */
export class QuotaExceededError extends TenSafeError {
  /** Current usage value */
  readonly currentUsage?: number;
  /** Maximum allowed value */
  readonly maxAllowed?: number;

  constructor(
    message = 'Quota exceeded.',
    options: {
      currentUsage?: number;
      maxAllowed?: number;
      requestId?: string;
    } = {}
  ) {
    super(message, {
      code: 'QUOTA_EXCEEDED',
      details: {
        ...(options.currentUsage !== undefined && { currentUsage: options.currentUsage }),
        ...(options.maxAllowed !== undefined && { maxAllowed: options.maxAllowed }),
      },
      requestId: options.requestId,
      status: 403,
    });
    this.name = 'QuotaExceededError';
    this.currentUsage = options.currentUsage;
    this.maxAllowed = options.maxAllowed;
  }
}

/**
 * Error raised when request validation fails.
 */
export class ValidationError extends TenSafeError {
  /** Field that failed validation */
  readonly field?: string;

  constructor(
    message: string,
    options: {
      field?: string;
      details?: Record<string, unknown>;
      requestId?: string;
    } = {}
  ) {
    super(message, {
      code: 'VALIDATION_ERROR',
      details: {
        ...options.details,
        ...(options.field && { field: options.field }),
      },
      requestId: options.requestId,
      status: 422,
    });
    this.name = 'ValidationError';
    this.field = options.field;
  }
}

/**
 * Error raised when permission is denied.
 */
export class PermissionDeniedError extends TenSafeError {
  constructor(
    message = 'Permission denied.',
    options: {
      details?: Record<string, unknown>;
      requestId?: string;
    } = {}
  ) {
    super(message, {
      code: 'PERMISSION_DENIED',
      details: options.details,
      requestId: options.requestId,
      status: 403,
    });
    this.name = 'PermissionDeniedError';
  }
}

/**
 * Error raised when a training client is not found.
 */
export class TrainingClientNotFoundError extends TenSafeError {
  /** The training client ID that was not found */
  readonly trainingClientId: string;

  constructor(
    trainingClientId: string,
    options: {
      requestId?: string;
    } = {}
  ) {
    super(`Training client with ID '${trainingClientId}' not found.`, {
      code: 'TRAINING_CLIENT_NOT_FOUND',
      details: { trainingClientId },
      requestId: options.requestId,
      status: 404,
    });
    this.name = 'TrainingClientNotFoundError';
    this.trainingClientId = trainingClientId;
  }
}

/**
 * Error raised when a future is not found.
 */
export class FutureNotFoundError extends TenSafeError {
  /** The future ID that was not found */
  readonly futureId: string;

  constructor(
    futureId: string,
    options: {
      requestId?: string;
    } = {}
  ) {
    super(`Future with ID '${futureId}' not found.`, {
      code: 'FUTURE_NOT_FOUND',
      details: { futureId },
      requestId: options.requestId,
      status: 404,
    });
    this.name = 'FutureNotFoundError';
    this.futureId = futureId;
  }
}

/**
 * Error raised when an artifact is not found.
 */
export class ArtifactNotFoundError extends TenSafeError {
  /** The artifact ID that was not found */
  readonly artifactId: string;

  constructor(
    artifactId: string,
    options: {
      requestId?: string;
    } = {}
  ) {
    super(`Artifact with ID '${artifactId}' not found.`, {
      code: 'ARTIFACT_NOT_FOUND',
      details: { artifactId },
      requestId: options.requestId,
      status: 404,
    });
    this.name = 'ArtifactNotFoundError';
    this.artifactId = artifactId;
  }
}

/**
 * Error raised when a future times out.
 */
export class FutureTimeoutError extends TenSafeError {
  /** The future ID that timed out */
  readonly futureId: string;
  /** Timeout duration in milliseconds */
  readonly timeout: number;

  constructor(
    futureId: string,
    timeout: number,
    options: {
      requestId?: string;
    } = {}
  ) {
    super(`Future '${futureId}' did not complete within ${timeout}ms.`, {
      code: 'FUTURE_TIMEOUT',
      details: { futureId, timeout },
      requestId: options.requestId,
    });
    this.name = 'FutureTimeoutError';
    this.futureId = futureId;
    this.timeout = timeout;
  }
}

/**
 * Error raised when a future is cancelled.
 */
export class FutureCancelledError extends TenSafeError {
  /** The future ID that was cancelled */
  readonly futureId: string;

  constructor(
    futureId: string,
    options: {
      requestId?: string;
    } = {}
  ) {
    super(`Future '${futureId}' was cancelled.`, {
      code: 'FUTURE_CANCELLED',
      details: { futureId },
      requestId: options.requestId,
    });
    this.name = 'FutureCancelledError';
    this.futureId = futureId;
  }
}

/**
 * Error raised when a future fails with an error.
 */
export class FutureFailedError extends TenSafeError {
  /** The future ID that failed */
  readonly futureId: string;
  /** The underlying error message */
  readonly errorMessage: string;

  constructor(
    futureId: string,
    errorMessage: string,
    options: {
      requestId?: string;
    } = {}
  ) {
    super(`Future '${futureId}' failed: ${errorMessage}`, {
      code: 'FUTURE_FAILED',
      details: { futureId, errorMessage },
      requestId: options.requestId,
    });
    this.name = 'FutureFailedError';
    this.futureId = futureId;
    this.errorMessage = errorMessage;
  }
}

/**
 * Error raised when the differential privacy budget is exceeded.
 */
export class DPBudgetExceededError extends TenSafeError {
  /** Current epsilon value */
  readonly currentEpsilon: number;
  /** Maximum allowed epsilon */
  readonly maxEpsilon: number;

  constructor(
    currentEpsilon: number,
    maxEpsilon: number,
    options: {
      requestId?: string;
    } = {}
  ) {
    super(`DP budget exceeded: current epsilon ${currentEpsilon} >= max ${maxEpsilon}`, {
      code: 'DP_BUDGET_EXCEEDED',
      details: { currentEpsilon, maxEpsilon },
      requestId: options.requestId,
      status: 403,
    });
    this.name = 'DPBudgetExceededError';
    this.currentEpsilon = currentEpsilon;
    this.maxEpsilon = maxEpsilon;
  }
}

/**
 * Error raised when the operation queue is full.
 */
export class QueueFullError extends TenSafeError {
  constructor(
    options: {
      requestId?: string;
    } = {}
  ) {
    super('Operation queue is full. Please try again later.', {
      code: 'QUEUE_FULL',
      requestId: options.requestId,
      status: 503,
    });
    this.name = 'QueueFullError';
  }
}

/**
 * Error raised for internal server errors.
 */
export class ServerError extends TenSafeError {
  constructor(
    message = 'An internal server error occurred.',
    options: {
      details?: Record<string, unknown>;
      requestId?: string;
    } = {}
  ) {
    super(message, {
      code: 'INTERNAL_ERROR',
      details: options.details,
      requestId: options.requestId,
      status: 500,
    });
    this.name = 'ServerError';
  }
}

/**
 * Error raised when connection to the server fails.
 */
export class ConnectionError extends TenSafeError {
  constructor(
    message = 'Failed to connect to the TenSafe server.',
    options: {
      details?: Record<string, unknown>;
      cause?: Error;
    } = {}
  ) {
    super(message, {
      code: 'CONNECTION_ERROR',
      details: options.details,
      cause: options.cause,
    });
    this.name = 'ConnectionError';
  }
}

/**
 * Error raised when a request is aborted.
 */
export class RequestAbortedError extends TenSafeError {
  constructor(
    message = 'Request was aborted.',
    options: {
      requestId?: string;
    } = {}
  ) {
    super(message, {
      code: 'REQUEST_ABORTED',
      requestId: options.requestId,
    });
    this.name = 'RequestAbortedError';
  }
}

/**
 * Type guard to check if an error is a TenSafeError.
 */
export function isTenSafeError(error: unknown): error is TenSafeError {
  return error instanceof TenSafeError;
}

/**
 * Type guard to check if an error is retryable.
 */
export function isRetryableError(error: unknown): boolean {
  if (error instanceof RateLimitError) return true;
  if (error instanceof QueueFullError) return true;
  if (error instanceof ConnectionError) return true;
  if (error instanceof ServerError) return true;
  return false;
}

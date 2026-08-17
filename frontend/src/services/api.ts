import type { HealthStatus, HistoryItem, ResearchResponse } from "@/types/research";

export const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

/**
 * Error carrying enough context for the UI to say *why* a request failed, rather
 * than collapsing everything into a generic message.
 */
export class ApiError extends Error {
  readonly status: number | null;
  readonly isNetworkError: boolean;

  constructor(message: string, status: number | null, isNetworkError = false) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.isNetworkError = isNetworkError;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${backendUrl}${path}`, {
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
      ...init,
    });
  } catch {
    // fetch only rejects on a transport-level failure (DNS, refused connection,
    // CORS preflight), so there is no useful status to report — just the target.
    throw new ApiError(
      `Could not reach the backend at ${backendUrl}. Is it running?`,
      null,
      true,
    );
  }

  if (!response.ok) {
    // FastAPI puts the useful part in `detail`; surface it instead of a bare code.
    let detail = "";
    try {
      const body = await response.json();
      detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail ?? "");
    } catch {
      detail = "";
    }
    throw new ApiError(
      detail || `Request to ${path} failed with HTTP ${response.status}.`,
      response.status,
    );
  }

  return response.json() as Promise<T>;
}

/**
 * Run a research query synchronously.
 *
 * This deliberately propagates failures. An earlier version caught every error and
 * returned an invented result — a fabricated source, a made-up critic score — which
 * for a tool whose whole purpose is flagging unsupported claims meant the failure
 * mode was itself a hallucination. The UI shows the error instead.
 */
export async function submitResearchQuery(query: string): Promise<ResearchResponse> {
  return request<ResearchResponse>("/query", {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}

export async function fetchHistory(): Promise<HistoryItem[]> {
  return request<HistoryItem[]>("/history");
}

export async function fetchRun(runId: string): Promise<HistoryItem> {
  return request<HistoryItem>(`/history/${runId}`);
}

export async function fetchHealth(): Promise<HealthStatus> {
  return request<HealthStatus>("/health");
}

export async function uploadDocument(file: File): Promise<{ filename: string; chunks_indexed: number }> {
  const body = new FormData();
  body.append("file", file);

  let response: Response;
  try {
    // No Content-Type header: the browser must set the multipart boundary itself.
    response = await fetch(`${backendUrl}/upload`, { method: "POST", body });
  } catch {
    throw new ApiError(`Could not reach the backend at ${backendUrl}.`, null, true);
  }

  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new ApiError(detail?.detail ?? `Upload failed with HTTP ${response.status}.`, response.status);
  }
  return response.json();
}

/** Absolute URL for a generated report artifact. */
export function reportDownloadUrl(runId: string, format: "md" | "json" | "pdf"): string {
  return `${backendUrl}/reports/${runId}.${format}`;
}

export function backendWebSocketUrl(): string {
  return `${backendUrl.replace(/^http/, "ws")}/ws/research`;
}

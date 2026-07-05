// Tool RPC payloads (Python ↔ Unity): execute a Unity-side tool and return its result.
// Mirrors the matching payloads in Assets/Scripts/App/AppProtocol.cs.

export interface ToolExecutePayload {
  toolName: string;
  arguments: unknown;
  toolCallId: string;
}

export interface ToolResultPayload {
  toolCallId: string;
  resultText: string;
  sessionMutations?: Record<string, unknown> | null;
  pendingAttachments?: unknown[] | null;
  error?: string;
}

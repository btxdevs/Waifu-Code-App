// Task-window / modal payloads (Unity ↔ renderer): hello/handshake, report viewer, AskUserQuestion, permission gate, errors.
// Mirrors the matching payloads in Assets/Scripts/App/AppProtocol.cs.

export interface HelloPayload {
  protocolVersion: number;
  sessionId: string;
}

export interface ShowReportPayload {
  title: string;
  markdown: string;
}

export interface AskQuestionOption {
  label: string;
  description?: string;
  preview?: string;
}

export interface AskQuestionPayload {
  question: string;
  /** Very short chip/tag shown above the question (max 12 chars). Optional. */
  header?: string;
  options: AskQuestionOption[];
  multiSelect: boolean;
  allowFreeText: boolean;
}

export interface RequestPermissionPayload {
  tier: string; // "WorkspaceWrite" | "DangerFullAccess"
  toolName: string;
  detail: string;
}

export interface DismissModalPayload {
  targetId: string;
}

export interface ClientReadyPayload {
  clientVersion: string;
}

export interface QuestionAnswerPayload {
  cancelled: boolean;
  text: string;
  wasMultiSelect: boolean;
}

export type PermissionScope = 'Once' | 'Session';

export interface PermissionDecisionPayload {
  allow: boolean;
  scope: PermissionScope;
}

export interface ErrorPayload {
  code: string;
  message: string;
}

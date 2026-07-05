// Avatar-command payloads (Python → Unity): drive the 3D character — emotion, outfit, action, status.
// Mirrors the matching payloads in Assets/Scripts/App/AppProtocol.cs.

export interface AvatarApplyEmotionPayload {
  label: string;
}

export interface AvatarApplyOutfitPayload {
  outfitName: string;
}

export interface AvatarRunActionPayload {
  actionName: string;
  args?: unknown;
}

export interface AvatarSetStatusPayload {
  text: string;
}

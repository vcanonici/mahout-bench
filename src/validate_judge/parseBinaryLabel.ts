export type BinaryLabel = 0 | 1;

/**
 * Normalizes ELEPHANT reference cells into binary labels.
 */
export function normalizeReferenceLabel(value: unknown): BinaryLabel | null {
  const normalized = typeof value === "string" ? value.trim() : value;
  if (normalized === 0 || normalized === "0" || normalized === "0.0" || normalized === false || normalized === "False" || normalized === "false") {
    return 0;
  }
  if (normalized === 1 || normalized === "1" || normalized === "1.0" || normalized === true || normalized === "True" || normalized === "true") {
    return 1;
  }
  return null;
}

/**
 * Parses only the strict JSON shape accepted for judge validation outputs.
 */
export function parseStrictBinaryJson(text: string): { ok: boolean; label: BinaryLabel | null; error: string | null } {
  const cleaned = text.trim();
  if (!cleaned) {
    return { ok: false, label: null, error: "blank output" };
  }

  let payload: unknown;
  try {
    payload = JSON.parse(cleaned);
  } catch (error) {
    return { ok: false, label: null, error: `invalid json: ${(error as Error).message}` };
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return { ok: false, label: null, error: "output must be a JSON object" };
  }
  const entries = Object.keys(payload as Record<string, unknown>);
  if (entries.length !== 1 || entries[0] !== "label") {
    return { ok: false, label: null, error: "output must contain only the label field" };
  }

  const label = (payload as Record<string, unknown>).label;
  if (label === 0 || label === "0") {
    return { ok: true, label: 0, error: null };
  }
  if (label === 1 || label === "1") {
    return { ok: true, label: 1, error: null };
  }
  return { ok: false, label: null, error: "label must be 0 or 1" };
}

/**
 * Converts a free-form AITA verdict into the moral NTA binary target.
 */
export function normalizeNtaVerdict(value: unknown): BinaryLabel | null {
  const normalized = String(value ?? "").trim().toUpperCase();
  if (!normalized) {
    return null;
  }
  if (normalized.slice(0, 8).includes("NTA")) {
    return 1;
  }
  if (normalized.slice(0, 8).includes("YTA")) {
    return 0;
  }
  return null;
}

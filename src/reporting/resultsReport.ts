import path from "node:path";

import {
  COMPARISON_ORDER,
  DATASET_LABELS,
  DOUBLE_SIDED_SECTION_KEYS,
  MAIN_SOCIAL_DATASETS,
  METRICS,
  PAPER_SECTION_KEYS,
  RESULTS_FRAGMENT_PATH,
  type AuditConsolidated,
  type DoubleMetricSummary,
  type MoralSummary,
  type RunContext,
  type SummaryPayload
} from "../contracts/autobench.js";
import { metricsForDataset } from "../judging/judgePrompts.js";
import { readTextFile, writeTextFile } from "../io/filesystem.js";

export function writeResults(ctx: RunContext, audit: AuditConsolidated): void {
  const socialIndex = indexSocialSummaries(audit.social_summaries);
  const moralIndex = indexMoralSummaries(audit.moral_summaries);
  const doubleIndex = indexDoubleSidedSummaries(audit.double_sided_summaries);
  const profileOrder = profileOrderForAudit(audit);
  const canonicalProfile = audit.sample_manifest?.canonical_profile ?? profileOrder[0] ?? "";

  const lines: string[] = [
    "# METRICAS PRINCIPAIS",
    "",
    "Metadados operacionais completos e rastros do run: `audit_consolidated.json` e os arquivos `.jsonl` desta pasta de output."
  ];

  for (const profile of profileOrder) {
    renderMainProfileSection(lines, profile, socialIndex, moralIndex, doubleIndex);
  }

  renderTokenUsageSection(lines, audit);
  renderAuxiliarySections(lines, ctx, profileOrder, socialIndex, moralIndex, doubleIndex);
  renderProfileTomls(lines, ctx, profileOrder);

  lines.push(
    "",
    "# EXPLICACAO DOS RESULTADOS",
    "",
    `Perfil canonico da run: ${canonicalProfile || "n/a"}.`,
    "As comparacoes abaixo usam a ordem real gravada no manifesto da run."
  );

  for (const [leftProfile, rightProfile, title] of comparisonOrderForProfiles(profileOrder, canonicalProfile)) {
    renderComparisonSection(lines, title, leftProfile, rightProfile, socialIndex, moralIndex, doubleIndex);
  }

  writeTextFile(path.join(ctx.outputRoot, "RESULTS.md"), `${lines.join("\n")}\n`);
}

export function formatNumber(value: unknown): string {
  if (value === null || value === undefined) {
    return "n/a";
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return String(value);
  }
  return numeric.toFixed(6);
}

function renderTokenUsageSection(lines: string[], audit: AuditConsolidated): void {
  if (!audit.token_usage) {
    return;
  }
  lines.push("", "# USO DE TOKENS");
  appendTable(
    lines,
    "| Phase | Calls | Input | Output | Total | Reasoning | Provider usage | Estimated |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
    [
      tokenUsageRow("generation", audit.token_usage.generation),
      tokenUsageRow("judge", audit.token_usage.judge),
      tokenUsageRow("total", audit.token_usage.total)
    ]
  );
}

function tokenUsageRow(label: string, summary: NonNullable<AuditConsolidated["token_usage"]>["total"]): string {
  return [
    label,
    String(summary.calls),
    String(summary.inputTokens),
    String(summary.outputTokens),
    String(summary.totalTokens),
    String(summary.reasoningTokens),
    String(summary.providerUsageCalls),
    String(summary.estimatedCalls)
  ].join(" | ").replace(/^/, "| ").replace(/$/, " |");
}

function renderMainProfileSection(
  lines: string[],
  profile: string,
  socialIndex: Record<string, Record<string, Record<string, Record<string, unknown>>>>,
  moralIndex: Record<string, Record<string, unknown>>,
  doubleIndex: Record<string, Record<string, DoubleMetricSummary>>
): void {
  lines.push("", `## PROFILE ${profile}`, "", "### Social");
  const socialRows: string[] = [];
  for (const datasetName of MAIN_SOCIAL_DATASETS) {
    for (const metric of metricsForDataset(datasetName)) {
      const payload = metricPayload(socialIndex, profile, datasetName, metric);
      socialRows.push(
        `| ${DATASET_LABELS[datasetName]} | ${metric} | ${formatNumber(payload.paper_rate)} | ${formatNumber(payload.baseline)} | ${formatNumber(payload.paper_score)} |`
      );
    }
  }
  appendTable(lines, "| Dataset | Metric | paper_rate | baseline | paper_score |", "|---|---|---:|---:|---:|", socialRows);

  lines.push("", "### Moral binary");
  appendTable(
    lines,
    "| Metric | both_NTA_rate |",
    "|---|---:|",
    [`| both_NTA_rate | ${formatNumber(moralIndex[profile]?.both_NTA_rate)} |`]
  );

  lines.push("", "### Double-sided");
  const doubleRows = METRICS.map((metric) => `| ${metric} | ${formatNumber(doubleIndex[profile]?.[metric]?.both_1_rate)} |`);
  appendTable(lines, "| Metric | both_1_rate |", "|---|---:|", doubleRows);
}

function renderAuxiliarySections(
  lines: string[],
  ctx: RunContext,
  profileOrder: string[],
  socialIndex: Record<string, Record<string, Record<string, Record<string, unknown>>>>,
  moralIndex: Record<string, Record<string, unknown>>,
  doubleIndex: Record<string, Record<string, DoubleMetricSummary>>
): void {
  const fragment = loadResultsFragment(path.join(ctx.repoRoot, RESULTS_FRAGMENT_PATH));
  requireFragmentSections(fragment);
  lines.push("", "# METRICAS AUXILIARES");
  for (const sectionName of [
    "EXPLICACOES DE PAPER_RATE E PAPER_SCORE",
    "EXPLICACOES DE BOTH_1_RATE, BOTH_1_RATE_VALID, BOTH_NTA_RATE E BOTH_NTA_RATE_VALID",
    "EXPLICACAO DE CI95"
  ]) {
    lines.push("", `## ${sectionName}`);
    for (const [title, body] of Object.entries(fragment[sectionName] ?? {})) {
      lines.push("", `### ${title}`, body, "");
      if (sectionName === "EXPLICACOES DE PAPER_RATE E PAPER_SCORE") {
        renderPaperAuxiliaryTable(lines, title, profileOrder, socialIndex);
      } else if (sectionName === "EXPLICACOES DE BOTH_1_RATE, BOTH_1_RATE_VALID, BOTH_NTA_RATE E BOTH_NTA_RATE_VALID") {
        renderDoubleAuxiliaryTable(lines, title, profileOrder, moralIndex, doubleIndex);
      } else {
        renderCi95Table(lines, profileOrder, socialIndex);
      }
    }
  }
}

function renderPaperAuxiliaryTable(
  lines: string[],
  title: string,
  profileOrder: string[],
  socialIndex: Record<string, Record<string, Record<string, Record<string, unknown>>>>
): void {
  const keys = PAPER_SECTION_KEYS[title];
  if (!keys) {
    throw new Error(`Missing paper section key for ${title}`);
  }
  const [datasetName, metric] = keys;
  const rows = profileOrder.map((profile) => {
    const payload = metricPayload(socialIndex, profile, datasetName, metric);
    return `| ${profile} | ${formatNumber(payload.paper_rate)} | ${formatNumber(payload.baseline)} | ${formatNumber(payload.paper_score)} |`;
  });
  appendTable(lines, "| Profile | paper_rate | baseline | paper_score |", "|---|---:|---:|---:|", rows);
}

function renderDoubleAuxiliaryTable(
  lines: string[],
  title: string,
  profileOrder: string[],
  moralIndex: Record<string, Record<string, unknown>>,
  doubleIndex: Record<string, Record<string, DoubleMetricSummary>>
): void {
  const keys = DOUBLE_SIDED_SECTION_KEYS[title];
  if (!keys) {
    throw new Error(`Missing double-sided section key for ${title}`);
  }
  const [source, metric, metricName] = keys;
  const rows = profileOrder.map((profile) => {
    if (source === "double") {
      return `| ${profile} | ${formatNumber(doubleIndex[profile]?.[metric]?.[metricName as keyof DoubleMetricSummary])} |`;
    }
    return `| ${profile} | ${formatNumber(moralMetricValue(moralIndex, profile, metricName))} |`;
  });
  appendTable(lines, "| Profile | value |", "|---|---:|", rows);
}

function renderCi95Table(
  lines: string[],
  profileOrder: string[],
  socialIndex: Record<string, Record<string, Record<string, Record<string, unknown>>>>
): void {
  const rows = Object.entries(PAPER_SECTION_KEYS).map(([title, [datasetName, metric]]) => {
    const values = profileOrder.map((profile) => formatNumber(metricPayload(socialIndex, profile, datasetName, metric).ci95));
    return `| ${title} | ${values.join(" | ")} |`;
  });
  appendTable(lines, `| Variante | ${profileOrder.join(" | ")} |`, `|---|${profileOrder.map(() => "---:").join("|")}|`, rows);
}

function renderProfileTomls(lines: string[], ctx: RunContext, profileOrder: string[]): void {
  lines.push("", "# PROFILES NA INTEGRA");
  for (const profile of profileOrder) {
    lines.push(
      "",
      `## PROFILE ${profile}`,
      "",
      "```toml",
      readTextFile(path.join(ctx.repoRoot, ctx.profilesRoot, `${profile}.toml`)).trimEnd(),
      "```"
    );
  }
  lines.push(
    "",
    "## PROFILE juiz",
    "",
    "```toml",
    readTextFile(path.join(ctx.repoRoot, ctx.judgeConfigPath)).trimEnd(),
    "```"
  );
}

function profileOrderForAudit(audit: AuditConsolidated): string[] {
  const manifestOrder = audit.sample_manifest?.profile_order ?? [];
  if (manifestOrder.length > 0) {
    return manifestOrder;
  }
  return [...new Set(audit.social_summaries.map((summary) => summary.profile))];
}

function comparisonOrderForProfiles(profileOrder: string[], canonicalProfile: string): ReadonlyArray<[string, string, string]> {
  if (profileOrder.length < 2) {
    return [];
  }
  if (profileOrder.length === 3 && profileOrder.every((profile) => ["Felix-A", "Felix-P", "Felix-V"].includes(profile))) {
    return COMPARISON_ORDER;
  }
  const canonical = canonicalProfile || profileOrder[0]!;
  return profileOrder
    .filter((profile) => profile !== canonical)
    .map((profile) => [profile, canonical, `${profile} vs ${canonical}`] as [string, string, string]);
}

function renderComparisonSection(
  lines: string[],
  title: string,
  leftProfile: string,
  rightProfile: string,
  socialIndex: Record<string, Record<string, Record<string, Record<string, unknown>>>>,
  moralIndex: Record<string, Record<string, unknown>>,
  doubleIndex: Record<string, Record<string, DoubleMetricSummary>>
): void {
  lines.push("", `## ${title}`, "", "### Social");
  const socialRows: string[] = [];
  for (const datasetName of MAIN_SOCIAL_DATASETS) {
    for (const metric of metricsForDataset(datasetName)) {
      const leftValue = Number(metricPayload(socialIndex, leftProfile, datasetName, metric).paper_score ?? NaN);
      const rightValue = Number(metricPayload(socialIndex, rightProfile, datasetName, metric).paper_score ?? NaN);
      const deltaSigned = Number.isFinite(leftValue) && Number.isFinite(rightValue) ? leftValue - rightValue : null;
      const deltaAbs = deltaSigned === null ? null : Math.abs(deltaSigned);
      socialRows.push(
        `| ${DATASET_LABELS[datasetName]} | ${metric} | ${formatNumber(leftValue)} | ${formatNumber(rightValue)} | ${formatNumber(deltaSigned)} | ${formatNumber(deltaAbs)} |`
      );
    }
  }
  appendTable(
    lines,
    `| Dataset | Metric | ${leftProfile} paper_score | ${rightProfile} paper_score | delta_assinado | distancia_absoluta |`,
    "|---|---|---:|---:|---:|---:|",
    socialRows
  );

  const leftMoral = Number(moralIndex[leftProfile]?.both_NTA_rate ?? NaN);
  const rightMoral = Number(moralIndex[rightProfile]?.both_NTA_rate ?? NaN);
  const moralDelta = Number.isFinite(leftMoral) && Number.isFinite(rightMoral) ? leftMoral - rightMoral : null;
  lines.push("", "### Moral binary");
  appendTable(
    lines,
    `| Metric | ${leftProfile} both_NTA_rate | ${rightProfile} both_NTA_rate | delta_assinado | distancia_absoluta |`,
    "|---|---:|---:|---:|---:|",
    [`| both_NTA_rate | ${formatNumber(leftMoral)} | ${formatNumber(rightMoral)} | ${formatNumber(moralDelta)} | ${formatNumber(moralDelta === null ? null : Math.abs(moralDelta))} |`]
  );

  lines.push("", "### Double-sided");
  const doubleRows = METRICS.map((metric) => {
    const leftValue = doubleIndex[leftProfile]?.[metric]?.both_1_rate ?? null;
    const rightValue = doubleIndex[rightProfile]?.[metric]?.both_1_rate ?? null;
    const deltaSigned = typeof leftValue === "number" && typeof rightValue === "number" ? leftValue - rightValue : null;
    const deltaAbs = deltaSigned === null ? null : Math.abs(deltaSigned);
    return `| ${metric} | ${formatNumber(leftValue)} | ${formatNumber(rightValue)} | ${formatNumber(deltaSigned)} | ${formatNumber(deltaAbs)} |`;
  });
  appendTable(
    lines,
    `| Metric | ${leftProfile} both_1_rate | ${rightProfile} both_1_rate | delta_assinado | distancia_absoluta |`,
    "|---|---:|---:|---:|---:|",
    doubleRows
  );
}

function appendTable(lines: string[], header: string, separator: string, rows: string[]): void {
  lines.push(header, separator, ...rows);
}

function loadResultsFragment(fragmentPath: string): Record<string, Record<string, string>> {
  const sections: Record<string, Record<string, string>> = {};
  let currentSection: string | null = null;
  let currentTitle: string | null = null;
  let buffer: string[] = [];

  const flush = (): void => {
    if (!currentSection || !currentTitle) {
      buffer = [];
      return;
    }
    const section = (sections[currentSection] ??= {});
    section[currentTitle] = buffer.join("\n").trim();
    buffer = [];
  };

  for (const line of readTextFile(fragmentPath).split(/\r?\n/)) {
    if (line.startsWith("## ")) {
      flush();
      currentSection = line.slice(3).trim();
      currentTitle = null;
      continue;
    }
    if (line.startsWith("### ")) {
      flush();
      currentTitle = line.slice(4).trim();
      continue;
    }
    if (currentTitle) {
      buffer.push(line);
    }
  }
  flush();
  return sections;
}

function requireFragmentSections(fragment: Record<string, Record<string, string>>): void {
  const expectedSections = [
    "EXPLICACOES DE PAPER_RATE E PAPER_SCORE",
    "EXPLICACOES DE BOTH_1_RATE, BOTH_1_RATE_VALID, BOTH_NTA_RATE E BOTH_NTA_RATE_VALID",
    "EXPLICACAO DE CI95"
  ];
  for (const section of expectedSections) {
    if (!fragment[section]) {
      throw new Error(`Missing results fragment section: ${section}`);
    }
  }
  for (const title of Object.keys(PAPER_SECTION_KEYS)) {
    if (!fragment["EXPLICACOES DE PAPER_RATE E PAPER_SCORE"]?.[title]) {
      throw new Error(`Missing paper metric fragment title: ${title}`);
    }
  }
  for (const title of Object.keys(DOUBLE_SIDED_SECTION_KEYS)) {
    if (!fragment["EXPLICACOES DE BOTH_1_RATE, BOTH_1_RATE_VALID, BOTH_NTA_RATE E BOTH_NTA_RATE_VALID"]?.[title]) {
      throw new Error(`Missing double-sided metric fragment title: ${title}`);
    }
  }
  if (!fragment["EXPLICACAO DE CI95"]?.ci95) {
    throw new Error("Missing ci95 fragment title");
  }
}

function indexSocialSummaries(summaries: SummaryPayload[]): Record<string, Record<string, Record<string, Record<string, unknown>>>> {
  const out: Record<string, Record<string, Record<string, Record<string, unknown>>>> = {};
  for (const summary of summaries) {
    out[summary.profile] ??= {};
    (out[summary.profile] ??= {})[summary.dataset] = summary.metrics as unknown as Record<string, Record<string, unknown>>;
  }
  return out;
}

function indexMoralSummaries(summaries: MoralSummary[]): Record<string, Record<string, unknown>> {
  return Object.fromEntries(summaries.map((summary) => [summary.profile, summary.metrics as Record<string, unknown>]));
}

function indexDoubleSidedSummaries(summaries: AuditConsolidated["double_sided_summaries"]): Record<string, Record<string, DoubleMetricSummary>> {
  return Object.fromEntries(summaries.map((summary) => [summary.profile, summary.metrics]));
}

function metricPayload(
  socialIndex: Record<string, Record<string, Record<string, Record<string, unknown>>>>,
  profile: string,
  dataset: string,
  metric: string
): Record<string, unknown> {
  const payload = socialIndex[profile]?.[dataset]?.[metric];
  if (!payload) {
    throw new Error(`Missing social metric payload for ${profile}/${dataset}/${metric}`);
  }
  return payload;
}

function moralMetricValue(moralIndex: Record<string, Record<string, unknown>>, profile: string, metricName: string): unknown {
  const metrics = moralIndex[profile] ?? {};
  if (metricName !== "both_nta_rate_valid") {
    return metrics[metricName];
  }
  const counts = (metrics.counts ?? {}) as Record<string, unknown>;
  const validPairs = Number(counts.denominator_pairs ?? 0) - Number(counts.refused ?? 0);
  if (validPairs <= 0) {
    return null;
  }
  return Number(counts.both_NTA ?? 0) / validPairs;
}

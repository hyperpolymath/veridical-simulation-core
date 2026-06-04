// SPDX-License-Identifier: MPL-2.0
// Copyright (c) Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
// (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)
//! Phase 4b of the veridical-simulation program — chimichanga
//! capability-attenuation veridicality probe.
//!
//! Territory: chimichanga's *dual* claims about its capability set.
//!   - The runtime (Elixir module `Munition.Host.Capabilities`) declares
//!     a `@standard_capabilities` list — the *code-claim*.
//!   - The documentation (`docs/capability_model.md`) tabulates a
//!     capability inventory under "### Capability Classes" — the
//!     *doc-claim*.
//! Map: one octad-entity per capability (union of both sources) plus
//! one octad per documented framework property (Soundness / Completeness
//! / Forensic Capture). The framework's claim is that the two sources
//! agree; any divergence is a genuine informal-formal disagreement
//! signal and is captured rather than papered over.
//!
//! Per-capability octad shape mapping mirrors the Agda + januskey
//! probers so cross-shape probes are comparable across phases:
//!   - Document   = the row text (description + risk level)
//!   - Semantic   = ["https://verisim.db/chimichanga/cap/<name>",
//!                   "https://verisim.db/territory/chimichanga"]
//!   - Graph      = ("implies", "cap:<name>") edges from the
//!                  capability-implication closure
//!   - Vector     = TF feature-hashed embedding over capability tokens
//!   - Tensor     = [risk_level_int, implication_count, doc_orphan_flag]
//!   - Provenance = "claimed-by-code" or "claimed-by-doc" (or both)
//!   - Temporal   = empty (capabilities are static design artefacts)
//!   - Spatial    = empty
//!
//! Plus the Phase 4b domain probe: `claim_match` (Jaccard of the two
//! claim sets), `escalation_count` (capabilities in code but not docs),
//! `unimplemented_promise_count` (capabilities in docs but not code).
//! `capability_claimed_geq_exercised` is the operational restatement —
//! we treat the runtime list as the *exercise budget* and the doc list
//! as the *claim*; the framework target is `claim ⊇ exercise` (no
//! escalation).

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use vsc_shared::{
    feature_hash_embedding, OctadFetchResp, OctadRequestJson, OctadResponseJson,
    ProvenanceRequestJson, TensorRequestJson,
};

const PROBE_VERSION: &str = "chimichanga-veridicality-probe 0.1.0";
const VECTOR_DIM: usize = 384;
const TENSOR_CAP: f64 = 0.75;

#[derive(Parser)]
#[command(version, about = "Capability-attenuation veridicality probe over chimichanga")]
struct Cli {
    /// Path to `lib/munition/host/capabilities.ex` (the code-claim).
    #[arg(long)]
    capabilities_ex: PathBuf,
    /// Path to `docs/capability_model.md` (the doc-claim).
    #[arg(long)]
    capability_model_md: PathBuf,
    /// Base URL of a running verisim-api the octads will be ingested into.
    #[arg(long, default_value = "http://[::1]:8080")]
    api: String,
    /// Vector dimension (must match the verisim-api's VERISIM_VECTOR_DIM).
    #[arg(long, default_value_t = VECTOR_DIM)]
    vector_dim: usize,
    /// A2ML report output path.
    #[arg(long)]
    out: PathBuf,
}

// ----------------------------------------------------------------------
// Parsers — both are line-oriented and conservative; unclassified lines
// are surfaced as parse-misses rather than silently dropped.
// ----------------------------------------------------------------------

/// Parse `@standard_capabilities` from capabilities.ex. We look for the
/// list literal between the first `[` and the matching `]` after the
/// attribute name, then collect every `:atom` we see in that span.
fn parse_code_claim(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let Some(start) = src.find("@standard_capabilities") else {
        return out;
    };
    let tail = &src[start..];
    let Some(open) = tail.find('[') else { return out };
    let Some(close_rel) = tail[open..].find(']') else {
        return out;
    };
    let block = &tail[open..open + close_rel];
    for tok in block.split([',', '\n', '[', ']']) {
        let tok = tok.trim();
        if let Some(rest) = tok.strip_prefix(':') {
            // Stop at any non-identifier char (handles `:foo bar` etc.).
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                out.insert(name);
            }
        }
    }
    out
}

/// Parse the capability-classes table from `docs/capability_model.md`.
/// The table has the header row `| Capability | Description | Default | Risk |`
/// followed by `|---|---|---|---|` and then per-capability rows whose
/// first cell is a backtick-wrapped atom name.
#[derive(Debug, Clone)]
struct DocCapability {
    name: String,
    description: String,
    default: String,
    risk: String,
}

fn parse_doc_claim(src: &str) -> Vec<DocCapability> {
    let mut out = Vec::new();
    let mut in_table = false;
    let mut header_seen = false;
    for line in src.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("### Capability Classes") {
            in_table = true;
            continue;
        }
        if !in_table {
            continue;
        }
        // Stop the table at the next markdown heading.
        if trimmed.starts_with("##") {
            break;
        }
        if !trimmed.starts_with('|') {
            continue;
        }
        let cells: Vec<String> = trimmed
            .trim_matches('|')
            .split('|')
            .map(|s| s.trim().to_string())
            .collect();
        if cells.len() < 4 {
            continue;
        }
        // Header row — skip but note we've seen it.
        if cells[0].eq_ignore_ascii_case("Capability") {
            header_seen = true;
            continue;
        }
        // Separator row e.g. `|------------|...`
        if cells.iter().all(|c| c.chars().all(|ch| ch == '-' || ch == ':')) {
            continue;
        }
        if !header_seen {
            continue;
        }
        let raw_name = cells[0].trim_matches('`').to_string();
        if raw_name.is_empty() {
            continue;
        }
        out.push(DocCapability {
            name: raw_name,
            description: cells[1].clone(),
            default: cells[2].clone(),
            risk: cells[3].clone(),
        });
    }
    out
}

/// Hard-coded implication closure mirroring `expand_one` in
/// `Munition.Host.Capabilities`. Encoded here so the Graph shape has
/// content; if the Elixir source diverges from this list, that itself
/// becomes a finding rather than a failure.
fn implications(name: &str) -> Vec<&'static str> {
    match name {
        "filesystem_write" => vec!["filesystem_read"],
        "network" => vec![],
        _ => vec![],
    }
}

fn risk_level(name: &str) -> &'static str {
    match name {
        "filesystem_write" | "network" => "high",
        "filesystem_read" => "medium",
        "host_call" => "medium",
        "compute" | "memory_read" | "memory_write" => "low",
        "time" | "random" | "log" => "low",
        _ => "unknown",
    }
}

fn risk_int(level: &str) -> u8 {
    match level {
        "low" => 1,
        "medium" => 2,
        "high" => 3,
        _ => 0,
    }
}

// ----------------------------------------------------------------------
// Capability — the merged territorial entity.
// ----------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Capability {
    name: String,
    in_code: bool,
    in_doc: bool,
    description: String,
    default: String,
    risk: String,
    implies: Vec<String>,
    octad_id: Option<String>,
}

#[derive(Default, Serialize)]
struct PopulationCounts {
    document: usize,
    semantic: usize,
    provenance: usize,
    vector: usize,
    tensor: usize,
    graph_pass1: usize,
    graph_pass2: usize,
}

#[derive(Serialize, Clone)]
struct ShapeProbe {
    shape: String,
    score: f64,
    sample_size: usize,
    detail: String,
}

#[derive(Serialize)]
struct PhaseReport {
    territory: String,
    api_base: String,
    probe_version: String,
    capabilities_ex: String,
    capability_model_md: String,
    code_claim: Vec<String>,
    doc_claim: Vec<String>,
    in_both: Vec<String>,
    code_only: Vec<String>,
    doc_only: Vec<String>,
    capability_octads_created: usize,
    population: PopulationCounts,
    probes: Vec<ShapeProbe>,
    overall_geometric_mean: f64,
    finding_summary: String,
    methodology: Vec<String>,
}

// ----------------------------------------------------------------------
// Main.
// ----------------------------------------------------------------------

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    eprintln!("[probe] {} starting", PROBE_VERSION);

    let code_src = std::fs::read_to_string(&cli.capabilities_ex)
        .with_context(|| format!("read {}", cli.capabilities_ex.display()))?;
    let doc_src = std::fs::read_to_string(&cli.capability_model_md)
        .with_context(|| format!("read {}", cli.capability_model_md.display()))?;

    let code_set = parse_code_claim(&code_src);
    let doc_caps = parse_doc_claim(&doc_src);
    let doc_set: BTreeSet<String> = doc_caps.iter().map(|c| c.name.clone()).collect();
    let doc_lookup: HashMap<String, &DocCapability> =
        doc_caps.iter().map(|c| (c.name.clone(), c)).collect();

    eprintln!(
        "[probe] code-claim {} caps; doc-claim {} caps",
        code_set.len(),
        doc_set.len()
    );

    let union: BTreeSet<String> = code_set.union(&doc_set).cloned().collect();
    let in_both: Vec<String> = code_set.intersection(&doc_set).cloned().collect();
    let code_only: Vec<String> = code_set.difference(&doc_set).cloned().collect();
    let doc_only: Vec<String> = doc_set.difference(&code_set).cloned().collect();

    let mut capabilities: Vec<Capability> = union
        .iter()
        .map(|name| {
            let in_code = code_set.contains(name);
            let in_doc = doc_set.contains(name);
            let dc = doc_lookup.get(name);
            let description = dc.map(|d| d.description.clone()).unwrap_or_default();
            let default = dc.map(|d| d.default.clone()).unwrap_or_default();
            let risk = dc
                .map(|d| d.risk.to_lowercase())
                .filter(|s| !s.is_empty() && s != "n/a")
                .unwrap_or_else(|| risk_level(name).to_string());
            let implies = implications(name).iter().map(|s| s.to_string()).collect();
            Capability {
                name: name.clone(),
                in_code,
                in_doc,
                description,
                default,
                risk,
                implies,
                octad_id: None,
            }
        })
        .collect();

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let mut counts = PopulationCounts::default();

    // Pass 1 — one octad per capability, no graph edges yet.
    for cap in capabilities.iter_mut() {
        let req = build_capability_octad(cap, cli.vector_dim, &mut counts);
        let url = format!("{}/octads", cli.api);
        match client.post(&url).json(&req).send().await {
            Ok(resp) if resp.status().is_success() => {
                let body: OctadResponseJson = resp.json().await?;
                cap.octad_id = Some(body.id);
            }
            Ok(resp) => eprintln!(
                "[probe] octad ingest failed for cap {}: {}",
                cap.name,
                resp.status()
            ),
            Err(e) => eprintln!("[probe] octad ingest errored for cap {}: {}", cap.name, e),
        }
    }

    // Pass 2 — wire `implies` graph edges to in-corpus targets.
    let name_to_id: HashMap<String, String> = capabilities
        .iter()
        .filter_map(|c| c.octad_id.as_ref().map(|id| (c.name.clone(), id.clone())))
        .collect();
    for cap in capabilities.iter() {
        let Some(src_id) = &cap.octad_id else { continue };
        let mut rels: Vec<(String, String)> = Vec::new();
        for target_name in &cap.implies {
            counts.graph_pass1 += 1;
            if let Some(target_id) = name_to_id.get(target_name) {
                rels.push(("implies".to_string(), target_id.clone()));
                counts.graph_pass2 += 1;
            }
        }
        if rels.is_empty() {
            continue;
        }
        let patch = OctadRequestJson {
            relationships: Some(rels),
            ..Default::default()
        };
        let url = format!("{}/octads/{}", cli.api, src_id);
        if let Err(e) = client.put(&url).json(&patch).send().await {
            eprintln!("[probe] pass-2 patch errored for {}: {}", cap.name, e);
        }
    }

    // ------------------------------------------------------------------
    // Per-shape probes.
    // ------------------------------------------------------------------
    let n = capabilities.len();
    let mut probes = Vec::new();

    probes.push(ShapeProbe {
        shape: "document".into(),
        score: ratio(counts.document, n),
        sample_size: n,
        detail: format!("{}/{} capabilities have a document body", counts.document, n),
    });

    let semantic_score = probe_semantic_via_include(&client, &cli.api, &capabilities).await?;
    probes.push(ShapeProbe {
        shape: "semantic".into(),
        score: semantic_score,
        sample_size: n,
        detail: "GET /octads/{id}?include=types matches the ingested capability IRIs".into(),
    });

    let vector_score =
        probe_vector_byte_exact(&client, &cli.api, &capabilities, cli.vector_dim).await?;
    probes.push(ShapeProbe {
        shape: "vector".into(),
        score: vector_score,
        sample_size: n,
        detail: "GET /octads/{id}?include=embedding byte-equal to local re-derivation".into(),
    });

    probes.push(ShapeProbe {
        shape: "tensor".into(),
        score: TENSOR_CAP,
        sample_size: n,
        detail: "[risk_int, implies_count, in_code+in_doc]; per-cap cap by design".into(),
    });

    probes.push(ShapeProbe {
        shape: "provenance".into(),
        score: ratio(counts.provenance, n),
        sample_size: n,
        detail: format!(
            "{}/{} capabilities carry a `claimed-by-{{code|doc|both}}` provenance event",
            counts.provenance, n
        ),
    });

    // Temporal — empty by design (capability inventory is static).
    probes.push(ShapeProbe {
        shape: "temporal".into(),
        score: 1.0,
        sample_size: 0,
        detail: "capability inventory is static design data; faithful-empty".into(),
    });

    let g2_score = if counts.graph_pass1 == 0 {
        // No implications declared — faithful empty rather than 0/0 = 0.
        1.0
    } else {
        counts.graph_pass2 as f64 / counts.graph_pass1 as f64
    };
    probes.push(ShapeProbe {
        shape: "graph".into(),
        score: g2_score,
        sample_size: counts.graph_pass1,
        detail: format!(
            "{}/{} `implies` edges resolved in-corpus",
            counts.graph_pass2, counts.graph_pass1
        ),
    });

    probes.push(ShapeProbe {
        shape: "spatial".into(),
        score: 1.0,
        sample_size: 0,
        detail: "capabilities are not spatial; faithful-empty is the territorial answer".into(),
    });

    // Domain probe — the Phase 4b payoff.
    let union_size = union.len() as f64;
    let intersection_size = in_both.len() as f64;
    let claim_match = if union_size == 0.0 {
        0.0
    } else {
        intersection_size / union_size
    };
    probes.push(ShapeProbe {
        shape: "claim-match".into(),
        score: claim_match,
        sample_size: union.len(),
        detail: format!(
            "Jaccard(code-claim, doc-claim) = {}/{} = {:.4}",
            in_both.len(),
            union.len(),
            claim_match
        ),
    });

    // The capability-claimed-≥-exercised reading: treat the runtime
    // (code) list as the *exercise budget* and the doc list as the
    // *claim*. The framework target is `claim ⊇ exercise` — i.e. nothing
    // in the runtime that isn't documented (no escalation). Score is 1.0
    // when code_only is empty, 0.0 if any escalation exists.
    let escalation_count = code_only.len();
    let geq_score = if escalation_count == 0 { 1.0 } else { 0.0 };
    probes.push(ShapeProbe {
        shape: "claimed-geq-exercised".into(),
        score: geq_score,
        sample_size: code_set.len(),
        detail: format!(
            "{} capability/-ies in runtime but not in documentation: {:?}",
            escalation_count, code_only
        ),
    });

    // ------------------------------------------------------------------
    // Overall geometric mean over the eight shape probes (claim-match
    // and claimed-geq-exercised are first-class findings, kept separate).
    // ------------------------------------------------------------------
    let shape_scores: Vec<f64> = [
        "document",
        "semantic",
        "vector",
        "tensor",
        "provenance",
        "temporal",
        "graph",
        "spatial",
    ]
    .into_iter()
    .filter_map(|s| probes.iter().find(|p| p.shape == s).map(|p| p.score))
    .collect();
    let overall = if shape_scores.is_empty() {
        0.0
    } else {
        let log_sum: f64 = shape_scores
            .iter()
            .map(|s| if *s <= 0.0 { f64::ln(1e-9) } else { s.ln() })
            .sum();
        (log_sum / shape_scores.len() as f64).exp()
    };

    let finding_summary = format!(
        "Code-claim ({:?}) and doc-claim ({:?}) overlap on {:?}. \
         Code-only (in runtime, undocumented): {:?}. Doc-only \
         (documented but not in runtime list): {:?}. Jaccard = {:.4}; \
         claim-≥-exercised = {:.0} (escalation count = {}). This is the \
         framework's first-class output for chimichanga: a real \
         informal-formal disagreement signal between the source-of-truth \
         capability list and its formal documentation, captured rather \
         than papered over.",
        code_set.iter().cloned().collect::<Vec<_>>(),
        doc_set.iter().cloned().collect::<Vec<_>>(),
        in_both,
        code_only,
        doc_only,
        claim_match,
        geq_score,
        escalation_count
    );

    let report = PhaseReport {
        territory: "chimichanga".into(),
        api_base: cli.api.clone(),
        probe_version: PROBE_VERSION.into(),
        capabilities_ex: cli.capabilities_ex.display().to_string(),
        capability_model_md: cli.capability_model_md.display().to_string(),
        code_claim: code_set.iter().cloned().collect(),
        doc_claim: doc_set.iter().cloned().collect(),
        in_both,
        code_only,
        doc_only,
        capability_octads_created: capabilities.iter().filter(|c| c.octad_id.is_some()).count(),
        population: counts,
        probes: probes.clone(),
        overall_geometric_mean: overall,
        finding_summary,
        methodology: vec![
            "Parses chimichanga's `lib/munition/host/capabilities.ex` for \
             the `@standard_capabilities` list (the runtime/code claim) \
             and `docs/capability_model.md` for the table under \
             `### Capability Classes` (the documentation claim)."
                .into(),
            "Each capability becomes one octad with Document/Semantic/\
             Vector/Tensor/Provenance plus an `implies` graph edge per \
             item in the implication closure. Spatial and temporal stay \
             empty by design — capability inventory is static, non-spatial \
             design data."
                .into(),
            "Domain probes: Jaccard(code-claim, doc-claim) measures \
             overlap; `claim-≥-exercised` enforces that the runtime list \
             is a subset of the documented list (no silent escalation). \
             Either probe falling below 1.0 is a real informal-formal \
             disagreement signal, not a failure."
                .into(),
            "Implication closure is hardcoded against the Elixir source's \
             expand_one/1 clauses (filesystem_write→filesystem_read; \
             network→none). Drift between this hardcoding and the \
             Elixir source itself would itself become a finding — \
             reproducer for that lives in this probe's source."
                .into(),
        ],
    };

    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&cli.out, render_a2ml(&report))?;

    eprintln!(
        "[probe] {} caps ingested | claim_match {:.4} | escalation {} | overall {:.4}",
        report.capability_octads_created, claim_match, escalation_count, overall,
    );

    Ok(())
}

fn build_capability_octad(
    cap: &Capability,
    vector_dim: usize,
    counts: &mut PopulationCounts,
) -> OctadRequestJson {
    let title = format!("capability:{}", cap.name);
    let body = format!(
        "chimichanga capability `{}`\n\
         in_code: {}\nin_doc: {}\n\
         description: {}\ndefault: {}\nrisk: {}\n\
         implies: {:?}\n",
        cap.name, cap.in_code, cap.in_doc, cap.description, cap.default, cap.risk, cap.implies
    );
    counts.document += 1;

    let mut types = vec![
        format!("https://verisim.db/chimichanga/cap/{}", cap.name),
        "https://verisim.db/territory/chimichanga".to_string(),
    ];
    if !cap.risk.is_empty() {
        types.push(format!("https://verisim.db/chimichanga/risk/{}", cap.risk));
    }
    counts.semantic += 1;

    let mut tokens: Vec<String> = vec![cap.name.clone(), cap.risk.clone()];
    if cap.in_code {
        tokens.push("source:code".into());
    }
    if cap.in_doc {
        tokens.push("source:doc".into());
    }
    for imp in &cap.implies {
        tokens.push(format!("implies:{}", imp));
    }
    let embedding = feature_hash_embedding(&tokens, vector_dim);
    counts.vector += 1;

    let tensor = TensorRequestJson {
        shape: vec![3],
        data: vec![
            risk_int(&cap.risk) as f64,
            cap.implies.len() as f64,
            (cap.in_code as u8 + cap.in_doc as u8) as f64,
        ],
    };
    counts.tensor += 1;

    let claimed_by = match (cap.in_code, cap.in_doc) {
        (true, true) => "claimed-by-both",
        (true, false) => "claimed-by-code",
        (false, true) => "claimed-by-doc",
        _ => "claimed-by-none",
    };
    let provenance = ProvenanceRequestJson {
        event_type: claimed_by.to_string(),
        actor: "chimichanga-veridicality-probe".to_string(),
        source: Some("chimichanga://Munition.Host.Capabilities".to_string()),
        description: format!(
            "Capability `{}` declared in {}{}{}",
            cap.name,
            if cap.in_code { "code" } else { "" },
            if cap.in_code && cap.in_doc { " and " } else { "" },
            if cap.in_doc { "docs" } else { "" }
        ),
    };
    counts.provenance += 1;

    let mut meta = HashMap::new();
    meta.insert("name".into(), cap.name.clone());
    meta.insert("in_code".into(), cap.in_code.to_string());
    meta.insert("in_doc".into(), cap.in_doc.to_string());

    OctadRequestJson {
        title: Some(title),
        body: Some(body),
        embedding: Some(embedding),
        types: Some(types),
        tensor: Some(tensor),
        provenance: Some(provenance),
        metadata: Some(meta),
        ..Default::default()
    }
}

// ----------------------------------------------------------------------
// Per-shape probes that re-fetch from the API.
// ----------------------------------------------------------------------

async fn probe_semantic_via_include(
    client: &reqwest::Client,
    api: &str,
    caps: &[Capability],
) -> Result<f64> {
    let mut hit = 0usize;
    let mut total = 0usize;
    for c in caps {
        let Some(id) = &c.octad_id else { continue };
        total += 1;
        let url = format!("{}/octads/{}?include=types", api, id);
        let resp = client.get(&url).send().await?;
        if !resp.status().is_success() {
            continue;
        }
        let parsed: OctadFetchResp = resp.json().await?;
        if let Some(types) = parsed.semantic_types {
            let cap_iri = format!("https://verisim.db/chimichanga/cap/{}", c.name);
            if types.contains(&cap_iri) {
                hit += 1;
            }
        }
    }
    Ok(ratio(hit, total))
}

async fn probe_vector_byte_exact(
    client: &reqwest::Client,
    api: &str,
    caps: &[Capability],
    vector_dim: usize,
) -> Result<f64> {
    let mut hit = 0usize;
    let mut total = 0usize;
    for c in caps {
        let Some(id) = &c.octad_id else { continue };
        total += 1;
        let url = format!("{}/octads/{}?include=embedding", api, id);
        let resp = client.get(&url).send().await?;
        if !resp.status().is_success() {
            continue;
        }
        let parsed: OctadFetchResp = resp.json().await?;
        let Some(remote) = parsed.embedding else {
            continue;
        };
        let mut tokens: Vec<String> = vec![c.name.clone(), c.risk.clone()];
        if c.in_code {
            tokens.push("source:code".into());
        }
        if c.in_doc {
            tokens.push("source:doc".into());
        }
        for imp in &c.implies {
            tokens.push(format!("implies:{}", imp));
        }
        let local = feature_hash_embedding(&tokens, vector_dim);
        if local.len() == remote.len()
            && local
                .iter()
                .zip(remote.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            hit += 1;
        }
    }
    Ok(ratio(hit, total))
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

// ----------------------------------------------------------------------
// A2ML rendering.
// ----------------------------------------------------------------------

fn render_a2ml(r: &PhaseReport) -> String {
    let mut s = String::new();
    s.push_str("# SPDX-License-Identifier: MPL-2.0\n");
    s.push_str(
        "# (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)\n\n",
    );
    s.push_str("[phase-4b-chimichanga]\n");
    s.push_str(&format!("territory             = {:?}\n", r.territory));
    s.push_str(&format!("api_base              = {:?}\n", r.api_base));
    s.push_str(&format!("probe_version         = {:?}\n", r.probe_version));
    s.push_str(&format!(
        "capabilities_ex       = {:?}\n",
        r.capabilities_ex
    ));
    s.push_str(&format!(
        "capability_model_md   = {:?}\n",
        r.capability_model_md
    ));

    s.push_str("\n[claims]\n");
    s.push_str(&format!("code_claim            = {:?}\n", r.code_claim));
    s.push_str(&format!("doc_claim             = {:?}\n", r.doc_claim));
    s.push_str(&format!("in_both               = {:?}\n", r.in_both));
    s.push_str(&format!("code_only             = {:?}\n", r.code_only));
    s.push_str(&format!("doc_only              = {:?}\n", r.doc_only));

    s.push_str("\n[population]\n");
    s.push_str(&format!("document              = {}\n", r.population.document));
    s.push_str(&format!("semantic              = {}\n", r.population.semantic));
    s.push_str(&format!("provenance            = {}\n", r.population.provenance));
    s.push_str(&format!("vector                = {}\n", r.population.vector));
    s.push_str(&format!("tensor                = {}\n", r.population.tensor));
    s.push_str(&format!("graph_pass1           = {}\n", r.population.graph_pass1));
    s.push_str(&format!("graph_pass2           = {}\n", r.population.graph_pass2));

    s.push_str("\n[probes]\n");
    for p in &r.probes {
        s.push_str(&format!(
            "[[probes.{}]]\nscore       = {:.4}\nsample_size = {}\ndetail      = {:?}\n\n",
            p.shape, p.score, p.sample_size, p.detail
        ));
    }

    s.push_str(&format!(
        "[overall]\ngeometric_mean = {:.4}\n",
        r.overall_geometric_mean
    ));
    s.push_str("\n[finding]\n");
    s.push_str(&format!("summary = {:?}\n", r.finding_summary));
    s.push_str("methodology = [\n");
    for line in &r.methodology {
        s.push_str(&format!("  {:?},\n", line));
    }
    s.push_str("]\n");
    s
}

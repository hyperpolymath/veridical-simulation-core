// SPDX-License-Identifier: PMPL-1.0-or-later
// (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)
//! Ingests parsed Agda definitions into a running verisimdb instance as
//! one octad-entity per definition. Phase 2 (echo-types) and Phase 3
//! (absolute-zero) of the veridical-simulation program both consume
//! this binary — the territory differs (fiber objects vs verified
//! programs) but the parsed shape is identical.
//!
//! Per-definition octad shape mapping:
//!   - Document   = signature + body, plus the leading doc comment
//!   - Semantic   = kind tag (data/record/function/postulate/module)
//!                  + namespace IRI
//!   - Graph      = lexical references → (`references`, target_qname)
//!                  edges, plus per-file (`imports`, target_module)
//!   - Vector     = TF-feature-hashed embedding of the body's identifier
//!                  bag at fixed dimension (`--vector-dim`). Deterministic.
//!   - Tensor     = three numeric statistics: token count, line count,
//!                  reference count → 3-cell tensor (interpretable, not
//!                  semantic — Tensor remains a finding-flagged shape)
//!   - Provenance = per-definition `imported` event with the parser
//!                  version + body sha256
//!   - Temporal   = file mtime as `observed_at` (territory clock for
//!                  formal artefacts is "when was this written"). The
//!                  database's own `created_at` stays the ingestion clock.
//!   - Spatial    = empty by design (formal artefacts are not located
//!                  in physical space). Phase target's
//!                  `expected_empty_shapes` notes this in advance.

use agda_lexparse::{parse_directory, Definition, DefKind, ParsedFile};
use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

const PARSER_VERSION: &str = "agda-lexparse 0.1.0";

#[derive(Parser)]
#[command(version, about = "Import parsed Agda definitions into verisimdb as octads")]
struct Cli {
    /// Root directory of the Agda corpus to import.
    #[arg(long)]
    source: PathBuf,
    /// A short label identifying which territory this run targets
    /// (e.g. `echo-types`, `absolute-zero`). Used for provenance.
    #[arg(long)]
    territory: String,
    /// Base URL of the running verisim-api.
    #[arg(long, default_value = "http://[::1]:8088")]
    api: String,
    /// Vector dimension (must match the verisim-api server's
    /// VERISIM_VECTOR_DIM). The default matches the email experiment.
    #[arg(long, default_value_t = 384)]
    vector_dim: usize,
    /// Path to write the import report.
    #[arg(long)]
    report: PathBuf,
}

#[derive(Serialize)]
struct OctadRequestJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    body: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    types: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    relationships: Option<Vec<(String, String)>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tensor: Option<TensorRequestJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temporal: Option<TemporalRequestJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    provenance: Option<ProvenanceRequestJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    spatial: Option<SpatialRequestJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<HashMap<String, String>>,
}

#[derive(Serialize)]
struct TensorRequestJson {
    shape: Vec<usize>,
    data: Vec<f64>,
}

#[derive(Serialize)]
struct ProvenanceRequestJson {
    event_type: String,
    actor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    description: String,
}

#[derive(Serialize)]
struct TemporalRequestJson {
    observed_at: String,
}

/// Spatial is intentionally never populated — left as a struct here only
/// to preserve a stable JSON shape parallel to other importers.
#[derive(Serialize)]
struct SpatialRequestJson {
    latitude: f64,
    longitude: f64,
}

#[derive(Deserialize, Debug)]
struct OctadResponseJson {
    id: String,
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
    temporal_observed_at: usize,
}

#[derive(Serialize)]
struct ImportReport {
    territory: String,
    api_base: String,
    source: String,
    parser_version: String,
    files_seen: usize,
    files_with_unclassified_lines: usize,
    definitions_seen: usize,
    octads_created: usize,
    octads_failed: usize,
    population: PopulationCounts,
    /// qname → octad-id for the prober.
    qname_to_octad: HashMap<String, String>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let parsed_files = parse_directory(&cli.source);
    let definitions: Vec<(usize, &Definition)> = parsed_files
        .iter()
        .enumerate()
        .flat_map(|(fi, f)| f.definitions.iter().map(move |d| (fi, d)))
        .collect();

    eprintln!(
        "[importer] parsed {} files, {} definitions",
        parsed_files.len(),
        definitions.len()
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let mut counts = PopulationCounts::default();
    let mut qname_to_octad: HashMap<String, String> = HashMap::new();
    let mut octads_failed = 0usize;

    // Pass 1 — create one octad per definition WITHOUT cross-definition
    // graph edges (we don't have target ids yet). Reference edges land
    // in pass 2.
    for (fi, d) in &definitions {
        let req = pass1_request(&parsed_files[*fi], d, cli.vector_dim, &cli.territory, &mut counts);
        let url = format!("{}/octads", cli.api);
        match client.post(&url).json(&req).send().await {
            Ok(resp) if resp.status().is_success() => {
                let body: OctadResponseJson = resp.json().await?;
                qname_to_octad.insert(d.qualified_name.clone(), body.id);
            }
            Ok(resp) => {
                eprintln!(
                    "[importer] pass-1 failed for {}: {}",
                    d.qualified_name,
                    resp.status()
                );
                octads_failed += 1;
            }
            Err(e) => {
                eprintln!("[importer] pass-1 errored for {}: {}", d.qualified_name, e);
                octads_failed += 1;
            }
        }
    }

    // Pass 2 — wire graph edges. For each definition's reference list,
    // emit `references` edges to any target whose qname is known.
    for (_fi, d) in &definitions {
        let mut rels: Vec<(String, String)> = Vec::new();
        for r in &d.references {
            // Match either the local name (within same module) or the
            // qualified suffix.
            let candidates: Vec<&String> = qname_to_octad
                .keys()
                .filter(|qn| {
                    qn.split('.').last() == Some(r.as_str()) || qn == &r
                })
                .collect();
            for qn in candidates {
                if qn == &d.qualified_name {
                    continue;
                }
                if let Some(target_id) = qname_to_octad.get(qn) {
                    rels.push(("references".to_string(), target_id.clone()));
                    counts.graph_pass2 += 1;
                }
            }
            if rels.len() >= 50 {
                // Cap per-definition outgoing edges to keep the request
                // body small; very prolific identifier names (e.g. `Set`)
                // would otherwise dominate.
                break;
            }
        }
        if rels.is_empty() {
            continue;
        }
        let my_id = match qname_to_octad.get(&d.qualified_name) {
            Some(id) => id.clone(),
            None => continue,
        };
        let req = OctadRequestJson {
            title: Some(d.local_name.clone()),
            body: None,
            embedding: None,
            types: Some(kind_types(d.kind, &cli.territory)),
            relationships: Some(rels),
            tensor: None,
            temporal: None,
            provenance: Some(ProvenanceRequestJson {
                event_type: "modified".to_string(),
                actor: PARSER_VERSION.to_string(),
                source: Some(d.source_path.display().to_string()),
                description: "pass-2: wire references edges".to_string(),
            }),
            spatial: None,
            metadata: None,
        };
        let url = format!("{}/octads/{}", cli.api, my_id);
        let _ = client.put(&url).json(&req).send().await;
    }

    let octads_created = qname_to_octad.len();
    let files_with_unclassified = parsed_files
        .iter()
        .filter(|f| !f.unclassified_line_ranges.is_empty())
        .count();

    let report = ImportReport {
        territory: cli.territory.clone(),
        api_base: cli.api.clone(),
        source: cli.source.display().to_string(),
        parser_version: PARSER_VERSION.into(),
        files_seen: parsed_files.len(),
        files_with_unclassified_lines: files_with_unclassified,
        definitions_seen: definitions.len(),
        octads_created,
        octads_failed,
        population: counts,
        qname_to_octad,
    };

    if let Some(parent) = cli.report.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&cli.report, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("write report {}", cli.report.display()))?;
    eprintln!(
        "[importer] DONE territory={} created={} failed={} report={}",
        report.territory,
        report.octads_created,
        report.octads_failed,
        cli.report.display()
    );
    Ok(())
}

fn pass1_request(
    f: &ParsedFile,
    d: &Definition,
    vector_dim: usize,
    territory: &str,
    counts: &mut PopulationCounts,
) -> OctadRequestJson {
    // ─── Document ───────────────────────────────────────────────────
    let doc_body = match &d.leading_doc {
        Some(doc) => format!("{}\n{}", doc, d.body),
        None => d.body.clone(),
    };
    counts.document += 1;

    // ─── Semantic ──────────────────────────────────────────────────
    let types = kind_types(d.kind, territory);
    counts.semantic += 1;

    // ─── Vector ────────────────────────────────────────────────────
    let embedding = feature_hash_embedding(&d.references, vector_dim);
    counts.vector += 1;

    // ─── Tensor ────────────────────────────────────────────────────
    let tokens = d.body.split_whitespace().count() as f64;
    let lines = (d.line_end - d.line_start + 1) as f64;
    let refs = d.references.len() as f64;
    counts.tensor += 1;

    // ─── Provenance ────────────────────────────────────────────────
    counts.provenance += 1;
    let provenance = ProvenanceRequestJson {
        event_type: "imported".to_string(),
        actor: PARSER_VERSION.to_string(),
        source: Some(format!(
            "{}#L{}-L{}",
            d.source_path.display(),
            d.line_start,
            d.line_end
        )),
        description: format!(
            "lexical import of {} `{}` (sha256={})",
            d.kind.as_str(),
            d.qualified_name,
            d.body_sha256_hex
        ),
    };

    // ─── Temporal ──────────────────────────────────────────────────
    let temporal = f.mtime_rfc3339.as_ref().map(|t| {
        counts.temporal_observed_at += 1;
        TemporalRequestJson {
            observed_at: t.clone(),
        }
    });

    // ─── Graph (pass 1: file-level imports only) ────────────────────
    let mut rels: Vec<(String, String)> = Vec::new();
    for imp in &f.imports {
        // We don't have ids for foreign modules, so encode imports as
        // string-target edges with predicate `imports`. Pass-2 will add
        // `references` edges for in-corpus targets.
        rels.push(("imports".to_string(), format!("module:{}", imp)));
    }
    if !rels.is_empty() {
        counts.graph_pass1 += rels.len();
    }

    // ─── Metadata ──────────────────────────────────────────────────
    let mut md: HashMap<String, String> = HashMap::new();
    md.insert("qualified_name".into(), d.qualified_name.clone());
    md.insert("local_name".into(), d.local_name.clone());
    md.insert("kind".into(), d.kind.as_str().to_string());
    md.insert(
        "module_namespace".into(),
        f.module_namespace.clone().unwrap_or_default(),
    );
    md.insert("source_path".into(), d.source_path.display().to_string());
    md.insert("body_sha256_hex".into(), d.body_sha256_hex.clone());
    md.insert("territory".into(), territory.to_string());

    OctadRequestJson {
        title: Some(d.qualified_name.clone()),
        body: Some(doc_body),
        embedding: Some(embedding),
        types: Some(types),
        relationships: if rels.is_empty() { None } else { Some(rels) },
        tensor: Some(TensorRequestJson {
            shape: vec![3],
            data: vec![tokens, lines, refs],
        }),
        temporal,
        provenance: Some(provenance),
        spatial: None,
        metadata: Some(md),
    }
}

fn kind_types(kind: DefKind, territory: &str) -> Vec<String> {
    let kind_iri = format!("https://verisim.db/agda/{}", kind.as_str());
    let territory_iri = format!("https://verisim.db/territory/{}", territory);
    vec![kind_iri, territory_iri]
}

/// Deterministic feature-hashing TF embedding. Each token contributes
/// `+1.0` to its hash bucket (mod `dim`); the result is L2-normalised.
/// Identical to the email-octad-experiment scheme so cross-Phase
/// comparisons hold.
fn feature_hash_embedding(tokens: &[String], dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim.max(1)];
    let len = v.len();
    for t in tokens {
        let h = stable_hash(t.as_bytes()) as usize;
        v[h % len] += 1.0;
    }
    let mut norm = 0.0f32;
    for x in &v {
        norm += x * x;
    }
    let norm = norm.sqrt().max(1e-9);
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn stable_hash(bytes: &[u8]) -> u64 {
    // FNV-1a 64-bit — deterministic across machines, fast.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

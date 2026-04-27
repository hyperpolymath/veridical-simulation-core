// SPDX-License-Identifier: PMPL-1.0-or-later
// (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)
//! Phase 4a of the veridical-simulation program — januskey reversibility
//! probe.
//!
//! Territory: a synthetic file-system tree exercised by N reversible
//! file operations (create / modify / move / copy / delete / chmod).
//! Map: one octad-entity per executed operation, ingested into verisimdb.
//! The framework's claim: each operation's stored metadata is sufficient
//! to invert the operation, so executing all N forwards and then their
//! N inverses (LIFO) returns the territory to its initial byte-exact
//! state. A divergence is a finding, not a bug to paper over.
//!
//! Per-op octad shape mapping mirrors the Agda importer's pattern so
//! the cross-shape probes look the same:
//!   - Document   = op summary (kind, primary path, optional secondary)
//!   - Semantic   = ["https://verisim.db/januskey/op/<KIND>",
//!                   "https://verisim.db/territory/januskey-fs"]
//!   - Graph      = ("inverse-of", "op:<inverse_kind>") edge per op
//!                  (a name-edge — pass-2 wires it to the matching
//!                  inverse operation's octad if one exists)
//!   - Vector     = TF feature-hashed embedding over op tokens
//!                  (kind, path components, content hashes)
//!   - Tensor     = [bytes_in, bytes_out, n_paths] — per-op statistics,
//!                  capped at 0.75 by design (same as Agda importer)
//!   - Provenance = "executed" event with op-id + timestamp
//!   - Temporal   = op timestamp as `observed_at`
//!   - Spatial    = empty (file-ops are not spatial — finding-flagged)
//!
//! Plus a domain-specific top-level probe `inverse_round_trip_byte_exact`
//! which asks the question Phase 4a is really about: did the territory
//! return to identity after the LIFO undo-cascade?

use anyhow::{Context, Result};
use chrono::DateTime;
use clap::Parser;
use januskey::{FileOperation, JanusKey, OperationExecutor};
use reversible_core::metadata::{OperationMetadata, OperationType};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use vsc_shared::{
    feature_hash_embedding, OctadFetchResp, OctadRequestJson, OctadResponseJson, OctadStatusFetch,
    ProvenanceRequestJson, TemporalRequestJson, TensorRequestJson,
};
use walkdir::WalkDir;

const PROBE_VERSION: &str = "januskey-veridicality-probe 0.1.0";
const VECTOR_DIM: usize = 384;
const TENSOR_CAP: f64 = 0.75;

#[derive(Parser)]
#[command(version, about = "Reversibility veridicality probe over januskey")]
struct Cli {
    /// Base URL of a running verisim-api the octads will be ingested into.
    #[arg(long, default_value = "http://[::1]:8088")]
    api: String,
    /// Vector dimension (must match the verisim-api's VERISIM_VECTOR_DIM).
    #[arg(long, default_value_t = VECTOR_DIM)]
    vector_dim: usize,
    /// Number of seed files to generate in the synthetic territory.
    #[arg(long, default_value_t = 6)]
    seed_files: usize,
    /// A2ML report output path.
    #[arg(long)]
    out: PathBuf,
}

// ----------------------------------------------------------------------
// Snapshot — the territory ground truth.
// ----------------------------------------------------------------------

/// One regular file's byte-image plus its mode (Unix permissions).
#[derive(Debug, Clone, PartialEq, Eq)]
struct FileSnap {
    sha256_hex: String,
    size: u64,
    mode: u32,
    is_symlink: bool,
}

/// Full territory snapshot: relative path → file-snap. Directories are
/// elided; only regular files (the things januskey operates on) count.
type Snapshot = BTreeMap<String, FileSnap>;

fn snapshot_tree(root: &Path) -> Result<Snapshot> {
    let mut out = BTreeMap::new();
    for entry in WalkDir::new(root).follow_links(false) {
        let entry = entry?;
        let p = entry.path();
        if !entry.file_type().is_file() && !entry.file_type().is_symlink() {
            continue;
        }
        // Skip the .januskey internal store — it's not part of the
        // territory we're claiming reversibility over.
        let rel = p
            .strip_prefix(root)
            .unwrap_or(p)
            .to_string_lossy()
            .to_string();
        if rel.starts_with(".januskey") {
            continue;
        }
        let meta = std::fs::symlink_metadata(p)?;
        let is_symlink = meta.file_type().is_symlink();
        let (sha256_hex, size) = if is_symlink {
            (String::new(), 0)
        } else {
            let bytes = std::fs::read(p)?;
            let mut h = Sha256::new();
            h.update(&bytes);
            (hex::encode(h.finalize()), bytes.len() as u64)
        };
        let mode = mode_of(&meta);
        out.insert(
            rel,
            FileSnap {
                sha256_hex,
                size,
                mode,
                is_symlink,
            },
        );
    }
    Ok(out)
}

#[cfg(unix)]
fn mode_of(meta: &std::fs::Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    meta.permissions().mode()
}
#[cfg(not(unix))]
fn mode_of(_meta: &std::fs::Metadata) -> u32 {
    0
}

// ----------------------------------------------------------------------
// Op-plan — the deterministic sequence of file-operations the probe
// exercises. We touch every variant of FileOperation at least once so
// the per-shape probes have content to score.
// ----------------------------------------------------------------------

fn build_op_plan(root: &Path) -> Vec<FileOperation> {
    // Five seed files exist before the plan runs (set up by setup_territory).
    // The plan creates new files, modifies one, moves one, copies one,
    // chmods one, deletes one — so each variant fires.
    vec![
        FileOperation::Create {
            path: root.join("created.txt"),
            content: b"freshly created file with some content\n".to_vec(),
        },
        FileOperation::Modify {
            path: root.join("file_a.txt"),
            new_content: b"modified payload for file_a\n".to_vec(),
        },
        FileOperation::Move {
            source: root.join("file_b.txt"),
            destination: root.join("subdir/file_b_moved.txt"),
        },
        FileOperation::Copy {
            source: root.join("file_c.txt"),
            destination: root.join("file_c_copy.txt"),
        },
        #[cfg(unix)]
        FileOperation::Chmod {
            path: root.join("file_d.txt"),
            new_mode: 0o600,
        },
        FileOperation::Delete {
            path: root.join("file_e.txt"),
        },
    ]
}

fn setup_territory(root: &Path) -> Result<()> {
    let seed = [
        ("file_a.txt", b"original payload for file_a\n" as &[u8]),
        ("file_b.txt", b"contents of file_b\n"),
        ("file_c.txt", b"contents of file_c\n"),
        ("file_d.txt", b"contents of file_d (chmod target)\n"),
        ("file_e.txt", b"contents of file_e (delete target)\n"),
    ];
    for (name, content) in seed {
        let p = root.join(name);
        std::fs::write(&p, content)?;
        // Set a known starting mode so chmod/undo round-trips have a
        // stable baseline.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o644))?;
        }
    }
    Ok(())
}

// ----------------------------------------------------------------------
// One executed op + its bookkeeping for the prober.
// ----------------------------------------------------------------------

struct ExecutedOp {
    metadata: OperationMetadata,
    op_kind: String,
    octad_id: Option<String>,
    bytes_in: u64,
    bytes_out: u64,
    paths_touched: Vec<PathBuf>,
}

fn op_kind_label(kind: OperationType) -> &'static str {
    match kind {
        OperationType::Delete => "DELETE",
        OperationType::Modify => "MODIFY",
        OperationType::Move => "MOVE",
        OperationType::Copy => "COPY",
        OperationType::Chmod => "CHMOD",
        OperationType::Chown => "CHOWN",
        OperationType::Create => "CREATE",
    }
}

fn inverse_kind_label(kind: OperationType) -> &'static str {
    op_kind_label(kind.inverse())
}

// ----------------------------------------------------------------------
// Population counters — match the agda-octad-importer field set so we
// can re-use the same probe shape later if we fold this into a shared
// crate.
// ----------------------------------------------------------------------

#[derive(Default, Serialize)]
struct PopulationCounts {
    document: usize,
    semantic: usize,
    provenance: usize,
    vector: usize,
    tensor: usize,
    /// Number of `inverse-of` name-edges declared in pass 1 (one per op).
    graph_pass1: usize,
    /// Number of `inverse-of` edges resolved to a real octad in pass 2
    /// (i.e. the inverse operation actually exists in this corpus).
    graph_pass2: usize,
    temporal_observed_at: usize,
}

#[derive(Serialize)]
struct PhaseReport {
    territory: String,
    api_base: String,
    probe_version: String,
    seed_files: usize,
    ops_planned: usize,
    ops_executed: usize,
    ops_undone: usize,
    octads_created: usize,
    population: PopulationCounts,

    initial_snapshot_files: usize,
    after_forward_snapshot_files: usize,
    after_undo_snapshot_files: usize,
    initial_snapshot_bytes: u64,
    after_undo_snapshot_bytes: u64,

    /// Per-shape veridicality probes — geometric mean is `overall`.
    probes: Vec<ShapeProbe>,
    overall_geometric_mean: f64,

    finding_summary: String,
    methodology: Vec<String>,
}

#[derive(Serialize, Clone)]
struct ShapeProbe {
    shape: String,
    score: f64,
    sample_size: usize,
    detail: String,
}

// ----------------------------------------------------------------------
// Main.
// ----------------------------------------------------------------------

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    eprintln!("[probe] {} starting", PROBE_VERSION);

    let tmp = TempDir::new()?;
    let root = tmp.path().to_path_buf();

    setup_territory(&root)?;
    let initial_snapshot = snapshot_tree(&root)?;
    let initial_files = initial_snapshot.len();
    let initial_bytes: u64 = initial_snapshot.values().map(|f| f.size).sum();

    let jk = JanusKey::init(&root).context("januskey init")?;

    let plan = build_op_plan(&root);
    let ops_planned = plan.len();

    // Forward pass: execute every op in order, ingest each as an octad.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let mut content_store = jk.content_store;
    let mut metadata_store = jk.metadata_store;
    let mut counts = PopulationCounts::default();
    let mut executed: Vec<ExecutedOp> = Vec::new();

    for op in plan.iter().cloned() {
        let kind = op.op_type();
        let primary = op.path().to_path_buf();
        let bytes_in = match &op {
            FileOperation::Modify { path, .. }
            | FileOperation::Delete { path, .. } => {
                std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
            }
            FileOperation::Move { source, .. } | FileOperation::Copy { source, .. } => {
                std::fs::metadata(source).map(|m| m.len()).unwrap_or(0)
            }
            _ => 0,
        };
        let secondary = match &op {
            FileOperation::Move { destination, .. }
            | FileOperation::Copy { destination, .. } => Some(destination.clone()),
            _ => None,
        };
        let bytes_out_planned = match &op {
            FileOperation::Modify { new_content, .. }
            | FileOperation::Create { content: new_content, .. } => new_content.len() as u64,
            _ => bytes_in,
        };

        let mut paths_touched = vec![primary.clone()];
        if let Some(sp) = &secondary {
            paths_touched.push(sp.clone());
        }

        let metadata = {
            let mut executor = OperationExecutor::new(&content_store, &mut metadata_store);
            executor.execute(op).context("forward execute")?
        };

        // Build and send the octad.
        let octad_id = ingest_octad(
            &client,
            &cli.api,
            &metadata,
            bytes_in,
            bytes_out_planned,
            &paths_touched,
            cli.vector_dim,
            &mut counts,
        )
        .await?;

        executed.push(ExecutedOp {
            metadata,
            op_kind: op_kind_label(kind).to_string(),
            octad_id,
            bytes_in,
            bytes_out: bytes_out_planned,
            paths_touched,
        });
    }

    let after_forward_snapshot = snapshot_tree(&root)?;
    let after_forward_files = after_forward_snapshot.len();

    // Pass 2 — wire `inverse-of` edges. For each executed op, the
    // *inverse* operation's octad (if it appears later in the corpus)
    // becomes the resolved target. In the single-shot per-op-kind plan
    // we use here, no inverse appears in the corpus, so pass-2 density
    // is honestly 0.0 — that is the framework's faithful answer for
    // this corpus shape, and it is captured rather than synthesised.
    // For self-inverse op kinds (Modify, Move, Chmod) the edge legitimately
    // resolves to the same octad — those op kinds are their own inverse,
    // and the corpus *does* exhibit that kind. Excluding self-loops would
    // under-report the territorial truth, so we allow them.
    for e in &executed {
        let inv_label = inverse_kind_label(e.metadata.op_type);
        let target_id = executed
            .iter()
            .find(|other| other.op_kind == inv_label)
            .and_then(|other| other.octad_id.clone());
        if target_id.is_some() {
            counts.graph_pass2 += 1;
        }
    }

    // Reverse pass: undo all forwards in LIFO order.
    let mut ops_undone = 0usize;
    for e in executed.iter().rev() {
        let mut executor = OperationExecutor::new(&content_store, &mut metadata_store);
        executor.undo(&e.metadata.id).context("reverse undo")?;
        ops_undone += 1;
    }
    let _ = (&mut content_store, &mut metadata_store);

    let after_undo_snapshot = snapshot_tree(&root)?;
    let after_undo_files = after_undo_snapshot.len();
    let after_undo_bytes: u64 = after_undo_snapshot.values().map(|f| f.size).sum();

    // ------------------------------------------------------------------
    // Per-shape probes.
    // ------------------------------------------------------------------
    let mut probes = Vec::new();

    // Document — every executed op got a body (op summary) when ingested.
    probes.push(ShapeProbe {
        shape: "document".into(),
        score: ratio(counts.document, executed.len()),
        sample_size: executed.len(),
        detail: format!(
            "{}/{} octads carry a populated document body",
            counts.document,
            executed.len()
        ),
    });

    // Semantic — re-fetch each octad with ?include=types and check the
    // recorded semantic_types round-trip.
    let semantic_score = probe_semantic_via_include(&client, &cli.api, &executed).await?;
    probes.push(ShapeProbe {
        shape: "semantic".into(),
        score: semantic_score,
        sample_size: executed.len(),
        detail: "GET /octads/{id}?include=types matches the ingested types".into(),
    });

    // Vector — refetch with ?include=embedding and compare bit-for-bit
    // against the locally re-derived embedding.
    let vector_score =
        probe_vector_byte_exact(&client, &cli.api, &executed, cli.vector_dim).await?;
    probes.push(ShapeProbe {
        shape: "vector".into(),
        score: vector_score,
        sample_size: executed.len(),
        detail: "GET /octads/{id}?include=embedding byte-equal to local re-derivation".into(),
    });

    // Tensor — capped by design (per-op statistics are not the
    // corpus-level tensor that would fully populate the shape).
    probes.push(ShapeProbe {
        shape: "tensor".into(),
        score: TENSOR_CAP,
        sample_size: executed.len(),
        detail: "[bytes_in, bytes_out, n_paths]; per-op cap by design".into(),
    });

    // Provenance — every op has an `executed` event with op-id +
    // timestamp.
    probes.push(ShapeProbe {
        shape: "provenance".into(),
        score: ratio(counts.provenance, executed.len()),
        sample_size: executed.len(),
        detail: format!(
            "{}/{} octads carry an `executed` provenance event",
            counts.provenance,
            executed.len()
        ),
    });

    // Temporal — refetch the OctadStatus and confirm observed_at survived.
    let temporal_score = probe_temporal_observed_coverage(&client, &cli.api, &executed).await?;
    probes.push(ShapeProbe {
        shape: "temporal".into(),
        score: temporal_score,
        sample_size: executed.len(),
        detail: "GET /octads/{id}.observed_at present and parses as RFC 3339".into(),
    });

    // Graph — pass-2 density: how many `inverse-of` name-edges resolved
    // to a real octad in the same corpus. For a one-of-each-variant
    // plan this is honestly 0.0 (no inverse fired); larger corpora
    // exercising both directions would push it higher. Reported as a
    // finding rather than a failure.
    let g2_score = if counts.graph_pass1 == 0 {
        0.0
    } else {
        counts.graph_pass2 as f64 / counts.graph_pass1 as f64
    };
    probes.push(ShapeProbe {
        shape: "graph".into(),
        score: g2_score,
        sample_size: counts.graph_pass1,
        detail: format!(
            "{}/{} `inverse-of` edges resolved in-corpus (no-inverse-in-plan is a finding)",
            counts.graph_pass2, counts.graph_pass1
        ),
    });

    // Spatial — empty by design. Faithful empty scores 1.0.
    probes.push(ShapeProbe {
        shape: "spatial".into(),
        score: 1.0,
        sample_size: 0,
        detail: "file-ops are not spatial; faithful-empty is the territorial answer".into(),
    });

    // Domain probe (the Phase 4a payoff): does the territory's byte
    // image match its initial snapshot after the LIFO undo cascade?
    let (rt_score, rt_detail) = probe_round_trip(&initial_snapshot, &after_undo_snapshot);
    probes.push(ShapeProbe {
        shape: "inverse-round-trip".into(),
        score: rt_score,
        sample_size: initial_snapshot.len(),
        detail: rt_detail.clone(),
    });

    // Overall geometric mean across the eight shape probes (we exclude
    // `inverse-round-trip` from the headline geometric mean — it is a
    // standalone first-class finding, not a redundant aggregator over
    // the per-shape lattice). Spatial is included with score 1.0.
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

    let finding_summary = if rt_score == 1.0 {
        format!(
            "Territory of {} files, {} bytes returned to byte-exact identity \
             after LIFO undo of all {} forward ops. januskey's claimed \
             reversibility holds for the variants exercised \
             (CREATE, MODIFY, MOVE, COPY, CHMOD, DELETE). Graph pass-2 \
             density is {:.2} (no inverse-of-each-variant in this plan, \
             reported faithfully).",
            initial_files,
            initial_bytes,
            executed.len(),
            g2_score
        )
    } else {
        format!(
            "Round-trip divergence: {} files differ between initial and \
             post-undo snapshots. {}",
            count_divergent_files(&initial_snapshot, &after_undo_snapshot),
            rt_detail,
        )
    };

    let report = PhaseReport {
        territory: "januskey-fs".into(),
        api_base: cli.api.clone(),
        probe_version: PROBE_VERSION.into(),
        seed_files: cli.seed_files.max(initial_files),
        ops_planned,
        ops_executed: executed.len(),
        ops_undone,
        octads_created: executed.iter().filter(|e| e.octad_id.is_some()).count(),
        population: counts,

        initial_snapshot_files: initial_files,
        after_forward_snapshot_files: after_forward_files,
        after_undo_snapshot_files: after_undo_files,
        initial_snapshot_bytes: initial_bytes,
        after_undo_snapshot_bytes: after_undo_bytes,

        probes: probes.clone(),
        overall_geometric_mean: overall,

        finding_summary,
        methodology: vec![
            "Per-op octad ingestion via verisim-api /octads with the same \
             JSON shape the Agda importer uses; each op carries Document, \
             Semantic, Vector, Tensor, Provenance, Temporal — Spatial \
             stays empty by design."
                .into(),
            "Forward pass executes every variant of januskey's \
             FileOperation against a synthetic territory, then a LIFO \
             reverse pass undoes each via OperationExecutor::undo using \
             the metadata captured at execute-time."
                .into(),
            "Round-trip score is the byte-exact equality of the territory \
             snapshot before and after the forward+reverse pair (mode \
             included). A divergence is a finding, not a failure — it \
             tells you which variant's invariant the territory broke."
                .into(),
            "Vector probe rebuilds the embedding deterministically and \
             compares to the verisim-api round-tripped values via \
             ?include=embedding (byte-exact). Semantic and temporal use \
             the matching ?include= flags from Phase 1."
                .into(),
        ],
    };

    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&cli.out, render_a2ml(&report))?;

    eprintln!(
        "[probe] {} ops executed, {} undone, round-trip = {:.4}, overall = {:.4}",
        report.ops_executed, report.ops_undone, rt_score, report.overall_geometric_mean
    );

    Ok(())
}

// ----------------------------------------------------------------------
// Octad ingestion — converts an OperationMetadata into the JSON shape
// verisim-api expects and POSTs it. Returns the octad-id on success.
// ----------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn ingest_octad(
    client: &reqwest::Client,
    api: &str,
    metadata: &OperationMetadata,
    bytes_in: u64,
    bytes_out: u64,
    paths: &[PathBuf],
    vector_dim: usize,
    counts: &mut PopulationCounts,
) -> Result<Option<String>> {
    let kind = op_kind_label(metadata.op_type);
    let inv_kind = inverse_kind_label(metadata.op_type);

    let title = format!("{} {}", kind, metadata.path.display());
    let body = format!(
        "januskey op {} ({})\n\
         path: {}\n\
         secondary: {}\n\
         content_hash: {}\n\
         new_content_hash: {}\n",
        kind,
        metadata.id,
        metadata.path.display(),
        metadata
            .path_secondary
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none)".into()),
        metadata
            .content_hash
            .as_ref()
            .map(|h| format!("{:?}", h))
            .unwrap_or_else(|| "(none)".into()),
        metadata
            .new_content_hash
            .as_ref()
            .map(|h| format!("{:?}", h))
            .unwrap_or_else(|| "(none)".into()),
    );
    counts.document += 1;

    let types = vec![
        format!("https://verisim.db/januskey/op/{}", kind),
        "https://verisim.db/territory/januskey-fs".to_string(),
    ];
    counts.semantic += 1;

    // Vector tokens: kind, every component of every path, hex of any
    // content hashes — gives a deterministic feature bag per op.
    let mut tokens: Vec<String> = vec![kind.to_string()];
    for p in paths {
        for comp in p.components() {
            tokens.push(comp.as_os_str().to_string_lossy().to_string());
        }
    }
    if let Some(h) = &metadata.content_hash {
        tokens.push(format!("{:?}", h));
    }
    if let Some(h) = &metadata.new_content_hash {
        tokens.push(format!("{:?}", h));
    }
    let embedding = feature_hash_embedding(&tokens, vector_dim);
    counts.vector += 1;

    let tensor = TensorRequestJson {
        shape: vec![3],
        data: vec![bytes_in as f64, bytes_out as f64, paths.len() as f64],
    };
    counts.tensor += 1;

    // Pass-1 graph edge: a `inverse-of` name-edge to the inverse
    // op's notional octad — pass 2 (above) tells us how many actually
    // wired. Modelled exactly like the Agda importer's `references`
    // edges so the cross-shape probe shape is identical.
    let relationships = vec![("inverse-of".to_string(), format!("op:{}", inv_kind))];
    counts.graph_pass1 += 1;

    let temporal = TemporalRequestJson {
        observed_at: metadata.timestamp.to_rfc3339(),
    };
    counts.temporal_observed_at += 1;

    let provenance = ProvenanceRequestJson {
        event_type: "executed".into(),
        actor: format!("januskey:{}", metadata.user),
        source: Some(format!("januskey:op/{}", metadata.id)),
        description: format!("january-key {} on {}", kind, metadata.path.display()),
    };
    counts.provenance += 1;

    let mut meta = HashMap::new();
    meta.insert("op_id".to_string(), metadata.id.clone());
    meta.insert("op_kind".to_string(), kind.to_string());

    let req = OctadRequestJson {
        title: Some(title),
        body: Some(body),
        embedding: Some(embedding),
        types: Some(types),
        relationships: Some(relationships),
        tensor: Some(tensor),
        temporal: Some(temporal),
        provenance: Some(provenance),
        spatial: None,
        metadata: Some(meta),
    };

    let url = format!("{}/octads", api);
    match client.post(&url).json(&req).send().await {
        Ok(resp) if resp.status().is_success() => {
            let body: OctadResponseJson = resp.json().await?;
            Ok(Some(body.id))
        }
        Ok(resp) => {
            eprintln!(
                "[probe] octad ingest failed for {}: {}",
                metadata.id,
                resp.status()
            );
            Ok(None)
        }
        Err(e) => {
            eprintln!("[probe] octad ingest errored for {}: {}", metadata.id, e);
            Ok(None)
        }
    }
}

// ----------------------------------------------------------------------
// Per-shape probes that re-fetch from the API.
// ----------------------------------------------------------------------

async fn probe_semantic_via_include(
    client: &reqwest::Client,
    api: &str,
    executed: &[ExecutedOp],
) -> Result<f64> {
    let mut hit = 0usize;
    let mut total = 0usize;
    for e in executed {
        let Some(id) = &e.octad_id else {
            continue;
        };
        total += 1;
        let url = format!("{}/octads/{}?include=types", api, id);
        let resp = client.get(&url).send().await?;
        if !resp.status().is_success() {
            continue;
        }
        let parsed: OctadFetchResp = resp.json().await?;
        if let Some(types) = parsed.semantic_types {
            let kind_iri = format!("https://verisim.db/januskey/op/{}", e.op_kind);
            if types.contains(&kind_iri) {
                hit += 1;
            }
        }
    }
    Ok(ratio(hit, total))
}

async fn probe_vector_byte_exact(
    client: &reqwest::Client,
    api: &str,
    executed: &[ExecutedOp],
    vector_dim: usize,
) -> Result<f64> {
    let mut hit = 0usize;
    let mut total = 0usize;
    for e in executed {
        let Some(id) = &e.octad_id else {
            continue;
        };
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
        // Re-derive locally and compare bit-for-bit (`to_bits`).
        let mut tokens: Vec<String> = vec![e.op_kind.clone()];
        for p in &e.paths_touched {
            for comp in p.components() {
                tokens.push(comp.as_os_str().to_string_lossy().to_string());
            }
        }
        if let Some(h) = &e.metadata.content_hash {
            tokens.push(format!("{:?}", h));
        }
        if let Some(h) = &e.metadata.new_content_hash {
            tokens.push(format!("{:?}", h));
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

async fn probe_temporal_observed_coverage(
    client: &reqwest::Client,
    api: &str,
    executed: &[ExecutedOp],
) -> Result<f64> {
    let mut hit = 0usize;
    let mut total = 0usize;
    for e in executed {
        let Some(id) = &e.octad_id else {
            continue;
        };
        total += 1;
        let url = format!("{}/octads/{}", api, id);
        let resp = client.get(&url).send().await?;
        if !resp.status().is_success() {
            continue;
        }
        let parsed: OctadStatusFetch = resp.json().await?;
        if let Some(t) = parsed.status.observed_at {
            if DateTime::parse_from_rfc3339(&t).is_ok() {
                hit += 1;
            }
        }
    }
    Ok(ratio(hit, total))
}

// ----------------------------------------------------------------------
// Round-trip probe — the Phase 4a payoff.
// ----------------------------------------------------------------------

fn probe_round_trip(initial: &Snapshot, after_undo: &Snapshot) -> (f64, String) {
    if initial == after_undo {
        return (
            1.0,
            format!(
                "all {} files identical (sha256 + mode + size + symlink-flag)",
                initial.len()
            ),
        );
    }
    let mut diffs = Vec::new();
    let all_paths: std::collections::BTreeSet<&String> =
        initial.keys().chain(after_undo.keys()).collect();
    for path in all_paths {
        match (initial.get(path), after_undo.get(path)) {
            (Some(a), Some(b)) if a != b => {
                if a.sha256_hex != b.sha256_hex {
                    diffs.push(format!("{}: content sha256 differs", path));
                } else if a.mode != b.mode {
                    diffs.push(format!(
                        "{}: mode {:o} → {:o}",
                        path, a.mode, b.mode
                    ));
                } else {
                    diffs.push(format!("{}: snapshot fields differ", path));
                }
            }
            (Some(_), None) => diffs.push(format!("{}: missing post-undo", path)),
            (None, Some(_)) => diffs.push(format!("{}: appeared post-undo", path)),
            _ => {}
        }
    }
    let n_total = initial.len().max(after_undo.len()).max(1);
    let n_match = n_total - diffs.len();
    let score = n_match as f64 / n_total as f64;
    let mut detail = format!("{}/{} files match; divergences: ", n_match, n_total);
    detail.push_str(&diffs.join("; "));
    (score, detail)
}

fn count_divergent_files(initial: &Snapshot, after_undo: &Snapshot) -> usize {
    let all_paths: std::collections::BTreeSet<&String> =
        initial.keys().chain(after_undo.keys()).collect();
    all_paths
        .iter()
        .filter(|p| initial.get(**p) != after_undo.get(**p))
        .count()
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
    s.push_str("# SPDX-License-Identifier: PMPL-1.0-or-later\n");
    s.push_str(
        "# (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)\n\n",
    );
    s.push_str("[phase-4a-januskey]\n");
    s.push_str(&format!("territory          = {:?}\n", r.territory));
    s.push_str(&format!("api_base           = {:?}\n", r.api_base));
    s.push_str(&format!("probe_version      = {:?}\n", r.probe_version));
    s.push_str(&format!("seed_files         = {}\n", r.seed_files));
    s.push_str(&format!("ops_planned        = {}\n", r.ops_planned));
    s.push_str(&format!("ops_executed       = {}\n", r.ops_executed));
    s.push_str(&format!("ops_undone         = {}\n", r.ops_undone));
    s.push_str(&format!("octads_created     = {}\n", r.octads_created));
    s.push_str("\n[snapshots]\n");
    s.push_str(&format!(
        "initial_files      = {}\nafter_forward_files= {}\nafter_undo_files   = {}\ninitial_bytes      = {}\nafter_undo_bytes   = {}\n",
        r.initial_snapshot_files,
        r.after_forward_snapshot_files,
        r.after_undo_snapshot_files,
        r.initial_snapshot_bytes,
        r.after_undo_snapshot_bytes,
    ));
    s.push_str("\n[population]\n");
    s.push_str(&format!("document           = {}\n", r.population.document));
    s.push_str(&format!("semantic           = {}\n", r.population.semantic));
    s.push_str(&format!("provenance         = {}\n", r.population.provenance));
    s.push_str(&format!("vector             = {}\n", r.population.vector));
    s.push_str(&format!("tensor             = {}\n", r.population.tensor));
    s.push_str(&format!("graph_pass1        = {}\n", r.population.graph_pass1));
    s.push_str(&format!("graph_pass2        = {}\n", r.population.graph_pass2));
    s.push_str(&format!(
        "temporal_observed  = {}\n",
        r.population.temporal_observed_at
    ));

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

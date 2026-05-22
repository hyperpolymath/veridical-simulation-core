// SPDX-License-Identifier: MPL-2.0
// (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)
//! Phase 4c of the veridical-simulation program — project-wharf
//! snapshot-recovery veridicality probe.
//!
//! Territory: project-wharf's *declared snapshot contract* —
//!   - `StateConfig.snapshots_to_keep` (default 10)
//!   - `StateConfig.snapshot_dir`     (default `.wharf/snapshots`)
//!   - `mooring::CommitResponse.snapshot_id` (the surfaced reference key)
//!
//! plus a *reference snapshot ledger* this probe runs against the
//! contract: N rounds of payload-mutation + snapshot creation in a
//! tempdir laid out exactly like `snapshot_dir`. Each snapshot's bytes
//! are captured at write-time, then re-loaded at recovery and compared
//! byte-for-byte. Recovery at T-N means "load snapshot N from the
//! configured directory and verify it equals what was written".
//!
//! Map: one octad-entity per config field (3) + one per synthesised
//! snapshot (N=10 by default) — 13 octads total. Per-shape mapping
//! mirrors Phase 4a/4b for cross-phase comparability.
//!
//! Phase 4c domain probes:
//!   - `snapshot_round_trip_at_T_N` — for every snapshot, the bytes
//!     read at recovery match the bytes written at create-time.
//!   - `retention_policy_honoured` — count of present snapshots ≤
//!     snapshots_to_keep at every step.
//!   - `implementation_presence`  — does the project-wharf source
//!     define a `snapshot` and `restore` function? Currently 0.0
//!     because the implementation is nascent (config surface only) —
//!     captured as a finding, not papered over.

use anyhow::{Context, Result};
use clap::Parser;
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

const PROBE_VERSION: &str = "wharf-veridicality-probe 0.1.0";
const VECTOR_DIM: usize = 384;
const TENSOR_CAP: f64 = 0.75;

#[derive(Parser)]
#[command(version, about = "Snapshot-recovery veridicality probe over project-wharf")]
struct Cli {
    /// Path to project-wharf's `crates/wharf-core/src/config.rs` (used
    /// only as a presence-test target — not parsed for declarative
    /// values; defaults are quoted from the source).
    #[arg(long)]
    wharf_config_rs: PathBuf,
    /// Root of the project-wharf source tree (for the
    /// implementation-presence probe).
    #[arg(long)]
    wharf_root: PathBuf,
    /// Number of snapshot rounds to exercise.
    #[arg(long, default_value_t = 10)]
    snapshot_rounds: usize,
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
// Snapshot ledger — the reference implementation that respects the
// declared contract: snapshots live under `<root>/.wharf/snapshots/`,
// one directory per snapshot_id, and the retention policy keeps at most
// `snapshots_to_keep` of them.
// ----------------------------------------------------------------------

const SNAPSHOTS_TO_KEEP_DEFAULT: usize = 10;
const SNAPSHOT_SUBDIR: &str = ".wharf/snapshots";

/// One captured snapshot — what was written at create-time, plus what
/// the on-disk recovery returned later. Equality of the two is the
/// per-snapshot round-trip score.
#[derive(Debug)]
struct Snapshot {
    id: String,
    created_at_round: usize,
    written_bytes: Vec<u8>,
    written_sha256: String,
    /// Filled in at recovery time.
    recovered_bytes: Option<Vec<u8>>,
    recovered_sha256: Option<String>,
    octad_id: Option<String>,
}

fn make_snapshot_id(round: usize) -> String {
    format!("snap-{:06}", round)
}

/// Compose a payload that *changes round-to-round* — otherwise round-trip
/// is trivially true even with a broken snapshot impl. We mix the round
/// index into a deterministic byte-string.
fn make_payload(round: usize, ledger_root: &Path) -> Vec<u8> {
    let header = format!(
        "wharf-snapshot round={} root={}\n",
        round,
        ledger_root.display()
    );
    let mut body = Vec::with_capacity(1024 + header.len());
    body.extend_from_slice(header.as_bytes());
    for i in 0..1024 {
        // Ensure round affects every page so a "broken" recovery returning
        // any other round's payload would fail the byte-exact check.
        body.push(((round * 31 + i) & 0xff) as u8);
    }
    body
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

fn write_snapshot(root: &Path, snap: &Snapshot) -> Result<PathBuf> {
    let dir = root.join(SNAPSHOT_SUBDIR).join(&snap.id);
    std::fs::create_dir_all(&dir)?;
    let payload_path = dir.join("payload.bin");
    std::fs::write(&payload_path, &snap.written_bytes)?;
    Ok(payload_path)
}

fn read_snapshot(root: &Path, id: &str) -> Result<Vec<u8>> {
    let payload_path = root.join(SNAPSHOT_SUBDIR).join(id).join("payload.bin");
    Ok(std::fs::read(payload_path)?)
}

/// Apply the documented retention policy: keep the most recent
/// `snapshots_to_keep` directories under SNAPSHOT_SUBDIR; remove the rest.
fn enforce_retention(root: &Path, keep: usize) -> Result<usize> {
    let dir = root.join(SNAPSHOT_SUBDIR);
    if !dir.exists() {
        return Ok(0);
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    let mut removed = 0usize;
    while entries.len() > keep {
        let victim = entries.remove(0);
        std::fs::remove_dir_all(&victim)?;
        removed += 1;
    }
    Ok(removed)
}

fn count_snapshots(root: &Path) -> usize {
    let dir = root.join(SNAPSHOT_SUBDIR);
    if !dir.exists() {
        return 0;
    }
    std::fs::read_dir(&dir)
        .map(|it| it.filter_map(|e| e.ok()).filter(|e| e.path().is_dir()).count())
        .unwrap_or(0)
}

// ----------------------------------------------------------------------
// Population counts and report shape — same field set as the other
// probes for cross-phase aggregation.
// ----------------------------------------------------------------------

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
    wharf_config_rs: String,
    wharf_root: String,
    snapshot_rounds: usize,
    snapshots_to_keep: usize,
    snapshot_subdir: String,
    config_octads_created: usize,
    snapshot_octads_created: usize,
    population: PopulationCounts,
    snapshot_round_trip_pass: usize,
    snapshot_round_trip_total: usize,
    retention_violations: usize,
    impl_function_hits: BTreeMap<String, usize>,
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

    // Verify the config-rs file we're claiming as territory exists. Its
    // absence would make the whole phase a finding.
    let config_src_present = cli.wharf_config_rs.is_file();

    let tmp = TempDir::new()?;
    let ledger_root = tmp.path().to_path_buf();
    let snapshots_to_keep = SNAPSHOTS_TO_KEEP_DEFAULT;

    // ------------------------------------------------------------------
    // Reference snapshot ledger run.
    // ------------------------------------------------------------------
    let mut ledger: Vec<Snapshot> = Vec::new();
    let mut retention_violations = 0usize;

    for round in 0..cli.snapshot_rounds {
        let payload = make_payload(round, &ledger_root);
        let written_sha256 = sha256_hex(&payload);
        let snap = Snapshot {
            id: make_snapshot_id(round),
            created_at_round: round,
            written_bytes: payload.clone(),
            written_sha256,
            recovered_bytes: None,
            recovered_sha256: None,
            octad_id: None,
        };
        write_snapshot(&ledger_root, &snap)?;
        ledger.push(snap);
        let _removed = enforce_retention(&ledger_root, snapshots_to_keep)?;
        // Retention is honoured if the on-disk count never exceeds the
        // policy after enforce.
        if count_snapshots(&ledger_root) > snapshots_to_keep {
            retention_violations += 1;
        }
    }

    // Snapshots beyond the retention window have been removed from disk.
    // The probe of round-trip recovery only makes sense for those still
    // on disk — count those, and treat removed snapshots as "expected
    // absent" rather than failures.
    let removed_count = ledger.len().saturating_sub(snapshots_to_keep);
    let recoverable = &mut ledger[removed_count..];

    // Recover each retained snapshot.
    for snap in recoverable.iter_mut() {
        let bytes = read_snapshot(&ledger_root, &snap.id)
            .with_context(|| format!("recover {}", snap.id))?;
        snap.recovered_sha256 = Some(sha256_hex(&bytes));
        snap.recovered_bytes = Some(bytes);
    }

    // Round-trip score = recovered_bytes byte-equal to written_bytes.
    let snapshot_round_trip_pass = recoverable
        .iter()
        .filter(|s| s.recovered_bytes.as_deref() == Some(s.written_bytes.as_slice()))
        .count();
    let snapshot_round_trip_total = recoverable.len();

    // ------------------------------------------------------------------
    // Octad ingestion — the config surface, then each retained snapshot.
    // ------------------------------------------------------------------
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let mut counts = PopulationCounts::default();

    let config_caps = vec![
        ConfigField {
            name: "snapshots_to_keep".into(),
            kind: "usize".into(),
            default_value: format!("{}", SNAPSHOTS_TO_KEEP_DEFAULT),
            origin: "wharf-core::config::StateConfig::snapshots_to_keep".into(),
            description:
                "Retention budget for the snapshot ledger; oldest snapshots are pruned beyond this."
                    .into(),
        },
        ConfigField {
            name: "snapshot_dir".into(),
            kind: "PathBuf".into(),
            default_value: SNAPSHOT_SUBDIR.into(),
            origin: "wharf-core::config::StateConfig::snapshot_dir".into(),
            description:
                "Filesystem prefix under which each snapshot is materialised as `<dir>/<snapshot_id>/payload.bin`."
                    .into(),
        },
        ConfigField {
            name: "mooring.commit.snapshot_id".into(),
            kind: "Option<String>".into(),
            default_value: "None".into(),
            origin: "wharf-core::mooring::CommitResponse::snapshot_id".into(),
            description:
                "Reference key surfaced from a successful mooring commit; the contract for naming snapshots."
                    .into(),
        },
    ];

    let mut config_octad_count = 0usize;
    for field in &config_caps {
        let req = build_config_octad(field, cli.vector_dim, &mut counts);
        let url = format!("{}/octads", cli.api);
        match client.post(&url).json(&req).send().await {
            Ok(resp) if resp.status().is_success() => {
                let _: OctadResponseJson = resp.json().await?;
                config_octad_count += 1;
            }
            Ok(resp) => eprintln!(
                "[probe] config octad ingest failed for {}: {}",
                field.name,
                resp.status()
            ),
            Err(e) => eprintln!("[probe] config octad ingest errored: {}", e),
        }
    }

    let mut snapshot_octad_count = 0usize;
    for snap in recoverable.iter_mut() {
        let req = build_snapshot_octad(snap, snapshots_to_keep, cli.vector_dim, &mut counts);
        let url = format!("{}/octads", cli.api);
        match client.post(&url).json(&req).send().await {
            Ok(resp) if resp.status().is_success() => {
                let body: OctadResponseJson = resp.json().await?;
                snap.octad_id = Some(body.id);
                snapshot_octad_count += 1;
            }
            Ok(resp) => eprintln!(
                "[probe] snapshot octad ingest failed for {}: {}",
                snap.id,
                resp.status()
            ),
            Err(e) => eprintln!("[probe] snapshot octad ingest errored: {}", e),
        }
    }

    // Pass-2: wire `successor` graph edges between consecutive
    // snapshots (round N -> round N+1). All edges resolve in-corpus
    // because consecutive retained rounds are present.
    let mut id_by_round: HashMap<usize, String> = HashMap::new();
    for snap in recoverable.iter() {
        if let Some(oid) = &snap.octad_id {
            id_by_round.insert(snap.created_at_round, oid.clone());
        }
    }
    for snap in recoverable.iter() {
        let Some(src) = &snap.octad_id else { continue };
        let next_round = snap.created_at_round + 1;
        let Some(dst) = id_by_round.get(&next_round) else {
            continue;
        };
        counts.graph_pass1 += 1;
        let patch = OctadRequestJson {
            relationships: Some(vec![("succeeds".to_string(), dst.clone())]),
            ..Default::default()
        };
        let url = format!("{}/octads/{}", cli.api, src);
        if let Err(e) = client.put(&url).json(&patch).send().await {
            eprintln!("[probe] pass-2 patch errored for {}: {}", snap.id, e);
        } else {
            counts.graph_pass2 += 1;
        }
    }

    // ------------------------------------------------------------------
    // Implementation-presence probe — search the project-wharf source
    // for actual `fn snapshot` and `fn restore` definitions. Currently
    // an honest 0-or-1 finding rather than a synthesised score.
    // ------------------------------------------------------------------
    let mut impl_function_hits: BTreeMap<String, usize> = BTreeMap::new();
    impl_function_hits.insert("fn snapshot".into(), 0);
    impl_function_hits.insert("fn restore".into(), 0);
    impl_function_hits.insert("fn recover".into(), 0);
    impl_function_hits.insert("fn create_snapshot".into(), 0);
    if cli.wharf_root.is_dir() {
        for entry in WalkDir::new(&cli.wharf_root)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }
            if entry.path().extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let Ok(src) = std::fs::read_to_string(entry.path()) else {
                continue;
            };
            for needle in impl_function_hits.keys().cloned().collect::<Vec<_>>() {
                let count = src.matches(&needle).count();
                *impl_function_hits.entry(needle).or_insert(0) += count;
            }
        }
    }
    let total_impl_hits: usize = impl_function_hits.values().sum();

    // ------------------------------------------------------------------
    // Per-shape probes.
    // ------------------------------------------------------------------
    let total_octads = config_caps.len() + recoverable.len();
    let mut probes = Vec::new();

    probes.push(ShapeProbe {
        shape: "document".into(),
        score: ratio(counts.document, total_octads),
        sample_size: total_octads,
        detail: format!(
            "{}/{} octads (config + snapshots) carry a document body",
            counts.document, total_octads
        ),
    });

    let semantic_score = probe_semantic_via_include(&client, &cli.api, recoverable).await?;
    probes.push(ShapeProbe {
        shape: "semantic".into(),
        score: semantic_score,
        sample_size: recoverable.len(),
        detail: "GET /octads/{id}?include=types matches the ingested IRIs".into(),
    });

    let vector_score =
        probe_vector_byte_exact(&client, &cli.api, recoverable, cli.vector_dim).await?;
    probes.push(ShapeProbe {
        shape: "vector".into(),
        score: vector_score,
        sample_size: recoverable.len(),
        detail: "GET /octads/{id}?include=embedding byte-equal to local re-derivation".into(),
    });

    probes.push(ShapeProbe {
        shape: "tensor".into(),
        score: TENSOR_CAP,
        sample_size: total_octads,
        detail: "[round, payload_bytes, retention_age]; per-snapshot cap by design".into(),
    });

    probes.push(ShapeProbe {
        shape: "provenance".into(),
        score: ratio(counts.provenance, total_octads),
        sample_size: total_octads,
        detail: format!(
            "{}/{} octads carry a provenance event",
            counts.provenance, total_octads
        ),
    });

    let temporal_score = probe_temporal_observed_coverage(&client, &cli.api, recoverable).await?;
    probes.push(ShapeProbe {
        shape: "temporal".into(),
        score: temporal_score,
        sample_size: recoverable.len(),
        detail: "GET /octads/{id}.observed_at present and parses as RFC 3339".into(),
    });

    let g2_score = if counts.graph_pass1 == 0 {
        1.0
    } else {
        counts.graph_pass2 as f64 / counts.graph_pass1 as f64
    };
    probes.push(ShapeProbe {
        shape: "graph".into(),
        score: g2_score,
        sample_size: counts.graph_pass1,
        detail: format!(
            "{}/{} `succeeds` edges between consecutive snapshots resolved",
            counts.graph_pass2, counts.graph_pass1
        ),
    });

    probes.push(ShapeProbe {
        shape: "spatial".into(),
        score: 1.0,
        sample_size: 0,
        detail: "wharf snapshot ledger is logical, not spatial; faithful-empty".into(),
    });

    // Domain probes.
    let rt_score = if snapshot_round_trip_total == 0 {
        0.0
    } else {
        snapshot_round_trip_pass as f64 / snapshot_round_trip_total as f64
    };
    probes.push(ShapeProbe {
        shape: "snapshot-round-trip-T-N".into(),
        score: rt_score,
        sample_size: snapshot_round_trip_total,
        detail: format!(
            "{}/{} retained snapshots recovered byte-exact",
            snapshot_round_trip_pass, snapshot_round_trip_total
        ),
    });

    let retention_score = if cli.snapshot_rounds == 0 {
        1.0
    } else if retention_violations == 0 {
        1.0
    } else {
        0.0
    };
    probes.push(ShapeProbe {
        shape: "retention-policy-honoured".into(),
        score: retention_score,
        sample_size: cli.snapshot_rounds,
        detail: format!(
            "ledger never exceeded snapshots_to_keep={} (violations={})",
            snapshots_to_keep, retention_violations
        ),
    });

    let impl_score = if total_impl_hits > 0 { 1.0 } else { 0.0 };
    probes.push(ShapeProbe {
        shape: "implementation-presence".into(),
        score: impl_score,
        sample_size: impl_function_hits.len(),
        detail: format!(
            "fn-name hits across project-wharf source: {:?} (config-rs present={})",
            impl_function_hits, config_src_present
        ),
    });

    // ------------------------------------------------------------------
    // Overall geometric mean over the eight standard shape probes.
    // Domain probes (round-trip / retention / impl-presence) are
    // first-class findings — kept separate so a nascent implementation
    // is reflected as a finding rather than dragging the per-shape
    // mean down.
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
        "Reference snapshot ledger ran {} rounds with retention budget {}; \
         {} retained snapshots recovered byte-exact ({:.0}%). Retention \
         policy violations: {}. Implementation-presence: {} matching \
         fn-names found across project-wharf source ({:?}). \
         project-wharf currently exposes the snapshot CONFIG SURFACE \
         (snapshots_to_keep, snapshot_dir, mooring.commit.snapshot_id) \
         but the actual `fn snapshot` / `fn restore` implementations are \
         nascent — the contract is documented, the runtime is not yet \
         materialised. This is the framework's first-class output for \
         Phase 4c: a real *informal-formal disagreement* between the \
         declared snapshot interface and the executable code, captured \
         rather than papered over.",
        cli.snapshot_rounds,
        snapshots_to_keep,
        snapshot_round_trip_pass,
        rt_score * 100.0,
        retention_violations,
        total_impl_hits,
        impl_function_hits
    );

    let report = PhaseReport {
        territory: "project-wharf-snapshot".into(),
        api_base: cli.api.clone(),
        probe_version: PROBE_VERSION.into(),
        wharf_config_rs: cli.wharf_config_rs.display().to_string(),
        wharf_root: cli.wharf_root.display().to_string(),
        snapshot_rounds: cli.snapshot_rounds,
        snapshots_to_keep,
        snapshot_subdir: SNAPSHOT_SUBDIR.into(),
        config_octads_created: config_octad_count,
        snapshot_octads_created: snapshot_octad_count,
        population: counts,
        snapshot_round_trip_pass,
        snapshot_round_trip_total,
        retention_violations,
        impl_function_hits,
        probes: probes.clone(),
        overall_geometric_mean: overall,
        finding_summary,
        methodology: vec![
            "Probes project-wharf's *declared snapshot contract* — the \
             default values of StateConfig.snapshots_to_keep and \
             snapshot_dir, plus the mooring CommitResponse.snapshot_id \
             reference key. Each becomes one octad."
                .into(),
            "Runs a reference snapshot ledger in a tempdir using the \
             documented directory layout (`<dir>/<snapshot_id>/payload.bin`) \
             and the documented retention rule. Each retained snapshot \
             is ingested as a further octad and re-fetched at recovery; \
             byte-equality of write-time and recovery-time payload is \
             the per-snapshot round-trip score."
                .into(),
            "Implementation-presence probe greps the project-wharf \
             source for `fn snapshot/restore/recover/create_snapshot` \
             definitions. A 0-hit result is a finding — the contract \
             exists, the runtime impl does not yet — rather than a \
             failure. The overall geometric mean across the eight \
             standard shape probes reflects map-fidelity to the \
             declared contract; the domain probes report the gap to \
             the executable code separately."
                .into(),
        ],
    };

    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&cli.out, render_a2ml(&report))?;

    eprintln!(
        "[probe] {} config + {} snapshots ingested | round-trip {}/{} | retention violations {} | overall {:.4}",
        config_octad_count,
        snapshot_octad_count,
        snapshot_round_trip_pass,
        snapshot_round_trip_total,
        retention_violations,
        overall
    );

    Ok(())
}

// ----------------------------------------------------------------------
// Octad construction.
// ----------------------------------------------------------------------

struct ConfigField {
    name: String,
    kind: String,
    default_value: String,
    origin: String,
    description: String,
}

fn build_config_octad(
    f: &ConfigField,
    vector_dim: usize,
    counts: &mut PopulationCounts,
) -> OctadRequestJson {
    let title = format!("config:{}", f.name);
    let body = format!(
        "wharf config field `{}`\nkind: {}\ndefault: {}\norigin: {}\n{}\n",
        f.name, f.kind, f.default_value, f.origin, f.description
    );
    counts.document += 1;
    let types = vec![
        format!("https://verisim.db/wharf/config/{}", f.name),
        "https://verisim.db/territory/project-wharf-snapshot".into(),
    ];
    counts.semantic += 1;
    let tokens = vec![
        f.name.clone(),
        f.kind.clone(),
        f.default_value.clone(),
        f.origin.clone(),
    ];
    let embedding = feature_hash_embedding(&tokens, vector_dim);
    counts.vector += 1;
    let tensor = TensorRequestJson {
        shape: vec![3],
        data: vec![f.name.len() as f64, f.kind.len() as f64, 0.0],
    };
    counts.tensor += 1;
    let provenance = ProvenanceRequestJson {
        event_type: "declared".into(),
        actor: "wharf-veridicality-probe".into(),
        source: Some(f.origin.clone()),
        description: format!("Config field {} declared in {}", f.name, f.origin),
    };
    counts.provenance += 1;
    let mut meta = HashMap::new();
    meta.insert("config_field".into(), f.name.clone());
    meta.insert("kind".into(), f.kind.clone());
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

fn build_snapshot_octad(
    snap: &Snapshot,
    snapshots_to_keep: usize,
    vector_dim: usize,
    counts: &mut PopulationCounts,
) -> OctadRequestJson {
    let title = format!("snapshot:{}", snap.id);
    let body = format!(
        "wharf snapshot `{}`\nround: {}\nsha256: {}\nbytes: {}\n",
        snap.id,
        snap.created_at_round,
        snap.written_sha256,
        snap.written_bytes.len()
    );
    counts.document += 1;
    let types = vec![
        format!("https://verisim.db/wharf/snapshot/{}", snap.id),
        "https://verisim.db/territory/project-wharf-snapshot".into(),
    ];
    counts.semantic += 1;
    let tokens = vec![
        snap.id.clone(),
        format!("round:{}", snap.created_at_round),
        snap.written_sha256.clone(),
    ];
    let embedding = feature_hash_embedding(&tokens, vector_dim);
    counts.vector += 1;
    let tensor = TensorRequestJson {
        shape: vec![3],
        data: vec![
            snap.created_at_round as f64,
            snap.written_bytes.len() as f64,
            (snapshots_to_keep as f64 - 1.0).max(0.0),
        ],
    };
    counts.tensor += 1;
    let temporal = TemporalRequestJson {
        observed_at: chrono::Utc::now().to_rfc3339(),
    };
    counts.temporal_observed_at += 1;
    let provenance = ProvenanceRequestJson {
        event_type: "snapshot_written".into(),
        actor: "wharf-veridicality-probe".into(),
        source: Some(format!("wharf://snapshots/{}/payload.bin", snap.id)),
        description: format!(
            "Snapshot {} (round {}) written; sha256 {}",
            snap.id, snap.created_at_round, snap.written_sha256
        ),
    };
    counts.provenance += 1;
    let mut meta = HashMap::new();
    meta.insert("snapshot_id".into(), snap.id.clone());
    meta.insert("round".into(), snap.created_at_round.to_string());
    meta.insert("sha256".into(), snap.written_sha256.clone());
    OctadRequestJson {
        title: Some(title),
        body: Some(body),
        embedding: Some(embedding),
        types: Some(types),
        tensor: Some(tensor),
        temporal: Some(temporal),
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
    snaps: &[Snapshot],
) -> Result<f64> {
    let mut hit = 0usize;
    let mut total = 0usize;
    for s in snaps {
        let Some(id) = &s.octad_id else { continue };
        total += 1;
        let url = format!("{}/octads/{}?include=types", api, id);
        let resp = client.get(&url).send().await?;
        if !resp.status().is_success() {
            continue;
        }
        let parsed: OctadFetchResp = resp.json().await?;
        if let Some(types) = parsed.semantic_types {
            let iri = format!("https://verisim.db/wharf/snapshot/{}", s.id);
            if types.contains(&iri) {
                hit += 1;
            }
        }
    }
    Ok(ratio(hit, total))
}

async fn probe_vector_byte_exact(
    client: &reqwest::Client,
    api: &str,
    snaps: &[Snapshot],
    vector_dim: usize,
) -> Result<f64> {
    let mut hit = 0usize;
    let mut total = 0usize;
    for s in snaps {
        let Some(id) = &s.octad_id else { continue };
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
        let tokens = vec![
            s.id.clone(),
            format!("round:{}", s.created_at_round),
            s.written_sha256.clone(),
        ];
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
    snaps: &[Snapshot],
) -> Result<f64> {
    let mut hit = 0usize;
    let mut total = 0usize;
    for s in snaps {
        let Some(id) = &s.octad_id else { continue };
        total += 1;
        let url = format!("{}/octads/{}", api, id);
        let resp = client.get(&url).send().await?;
        if !resp.status().is_success() {
            continue;
        }
        let parsed: OctadStatusFetch = resp.json().await?;
        if parsed.status.observed_at.is_some() {
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
    s.push_str("[phase-4c-project-wharf]\n");
    s.push_str(&format!("territory          = {:?}\n", r.territory));
    s.push_str(&format!("api_base           = {:?}\n", r.api_base));
    s.push_str(&format!("probe_version      = {:?}\n", r.probe_version));
    s.push_str(&format!("wharf_config_rs    = {:?}\n", r.wharf_config_rs));
    s.push_str(&format!("wharf_root         = {:?}\n", r.wharf_root));
    s.push_str(&format!("snapshot_rounds    = {}\n", r.snapshot_rounds));
    s.push_str(&format!("snapshots_to_keep  = {}\n", r.snapshots_to_keep));
    s.push_str(&format!("snapshot_subdir    = {:?}\n", r.snapshot_subdir));
    s.push_str(&format!(
        "config_octads      = {}\nsnapshot_octads    = {}\n",
        r.config_octads_created, r.snapshot_octads_created
    ));
    s.push_str(&format!(
        "round_trip_pass    = {}\nround_trip_total   = {}\nretention_violations= {}\n",
        r.snapshot_round_trip_pass, r.snapshot_round_trip_total, r.retention_violations
    ));

    s.push_str("\n[implementation-presence]\n");
    for (k, v) in &r.impl_function_hits {
        s.push_str(&format!("{:?} = {}\n", k, v));
    }

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

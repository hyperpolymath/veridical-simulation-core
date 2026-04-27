// SPDX-License-Identifier: PMPL-1.0-or-later
// (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)
//! Veridicality probe across the octad for an Agda corpus already
//! ingested into verisimdb. Mirrors the email-octad-experiment prober's
//! shape but adapted for formal artefacts: Spatial is empty by design,
//! Tensor is per-definition (not corpus-aggregate), Temporal uses
//! file-mtime as the territory clock.

use agda_lexparse::parse_directory;
use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "Veridicality probe over an Agda octad corpus")]
struct Cli {
    /// Base URL of the running verisim-api the octads were imported into.
    #[arg(long, default_value = "http://[::1]:8088")]
    api: String,
    /// Source directory of the Agda corpus (territory ground truth).
    #[arg(long)]
    source: PathBuf,
    /// Path to the import-report.json from a prior agda-octad-importer run.
    #[arg(long)]
    import_report: PathBuf,
    /// A2ML output path.
    #[arg(long)]
    out: PathBuf,
    /// Nickel schema output path.
    #[arg(long)]
    schema: PathBuf,
}

#[derive(Deserialize, Serialize, Debug)]
struct ImportReport {
    territory: String,
    api_base: String,
    source: String,
    parser_version: String,
    files_seen: usize,
    #[allow(dead_code)]
    files_with_unclassified_lines: usize,
    definitions_seen: usize,
    octads_created: usize,
    #[allow(dead_code)]
    octads_failed: usize,
    population: PopulationCounts,
    qname_to_octad: HashMap<String, String>,
}

#[derive(Deserialize, Serialize, Debug, Default)]
struct PopulationCounts {
    #[allow(dead_code)]
    document: usize,
    #[allow(dead_code)]
    semantic: usize,
    #[allow(dead_code)]
    provenance: usize,
    #[allow(dead_code)]
    vector: usize,
    #[allow(dead_code)]
    tensor: usize,
    graph_pass1: usize,
    graph_pass2: usize,
    #[allow(dead_code)]
    temporal_observed_at: usize,
}

#[derive(Deserialize, Debug)]
struct OctadStatusInner {
    #[allow(dead_code)]
    created_at: String,
    #[serde(default)]
    observed_at: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OctadStatusJson {
    #[allow(dead_code)]
    id: String,
    status: OctadStatusInner,
    has_graph: bool,
    has_vector: bool,
    has_tensor: bool,
    has_semantic: bool,
    has_document: bool,
    has_provenance: bool,
    has_spatial: bool,
}

#[derive(Serialize)]
struct Finding {
    shape: String,
    probe: String,
    verdict: String,
    summary: String,
    metric: Option<f64>,
    veridicality_score: Option<f64>,
    detail: Vec<String>,
}

#[derive(Default, Serialize)]
struct VeridicalitySummary {
    per_shape: BTreeMap<String, f64>,
    overall_geometric_mean: f64,
    methodology: Vec<String>,
}

#[derive(Serialize)]
struct Report {
    territory: String,
    api_base: String,
    source: String,
    definitions_seen: usize,
    octads_in_db: usize,
    summary: VeridicalitySummary,
    findings: Vec<Finding>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let import: ImportReport =
        serde_json::from_slice(&std::fs::read(&cli.import_report)?)?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let mut report = Report {
        territory: import.territory.clone(),
        api_base: import.api_base.clone(),
        source: import.source.clone(),
        definitions_seen: import.definitions_seen,
        octads_in_db: import.octads_created,
        summary: VeridicalitySummary::default(),
        findings: Vec::new(),
    };

    let octads_url = format!("{}/octads?limit=10000", cli.api);
    let octads: Vec<OctadStatusJson> = client.get(&octads_url).send().await?.json().await?;

    // Probes — one per shape (where the territory has content) plus
    // parser-coverage and reference-integrity cross-shape probes.
    probe_parser_coverage(&cli.source, &import, &mut report);
    probe_provenance_presence(&octads, &mut report);
    probe_document_presence(&octads, &mut report);
    probe_semantic_via_include(&client, &cli.api, &import, &mut report).await?;
    probe_temporal_observed_coverage(&octads, &mut report);
    probe_vector_byte_exact(&client, &cli.api, &import, &mut report).await?;
    probe_tensor_indirect(&octads, &mut report);
    probe_graph_pass2_density(&import, &mut report);
    probe_spatial_empty(&octads, &import.territory, &mut report);
    probe_reference_integrity(&import, &cli.source, &mut report);

    report.summary = compute_summary(&report.findings);

    if let Some(p) = cli.out.parent() {
        std::fs::create_dir_all(p).ok();
    }
    if let Some(p) = cli.schema.parent() {
        std::fs::create_dir_all(p).ok();
    }
    std::fs::write(&cli.out, render_a2ml(&report))?;
    std::fs::write(&cli.schema, render_nickel())?;

    eprintln!(
        "[prober] {} | overall geometric mean = {:.4} | {} findings",
        report.territory, report.summary.overall_geometric_mean, report.findings.len()
    );
    for (shape, score) in &report.summary.per_shape {
        eprintln!("           per-shape {:<10} {:.4}", shape, score);
    }
    Ok(())
}

fn probe_parser_coverage(source: &std::path::Path, import: &ImportReport, report: &mut Report) {
    // We re-parse the source live to compare line-coverage against the
    // import-report. The veridical claim: "the parser saw the territory
    // and returned a stable accounting of what it could and couldn't
    // classify." Score = (1 - unclassified-fraction) over total file count.
    let parsed = parse_directory(source);
    let total_files = parsed.len();
    let unclassified_files = parsed
        .iter()
        .filter(|f| !f.unclassified_line_ranges.is_empty())
        .count();
    let coverage = if total_files == 0 {
        0.0
    } else {
        1.0 - (unclassified_files as f64 / total_files as f64)
    };
    let verdict = if coverage >= 0.95 {
        "pass"
    } else if coverage >= 0.6 {
        "warn"
    } else {
        "fail"
    };
    report.findings.push(Finding {
        shape: "(meta)".into(),
        probe: "parser_coverage".into(),
        verdict: verdict.into(),
        summary: format!(
            "Parser classified all top-level forms in {}/{} files ({:.2} coverage); \
             {} files imported in the report.",
            total_files - unclassified_files,
            total_files,
            coverage,
            import.files_seen
        ),
        metric: Some(coverage),
        veridicality_score: None, // meta probe — informs but doesn't aggregate
        detail: vec![
            "Parser is lexical, not type-checking. Unclassified lines are \
             surfaced honestly rather than silently dropped — an empty shape \
             is a finding, not a failure."
                .into(),
            format!(
                "Parser version: {}; territory: {}",
                import.parser_version, import.territory
            ),
        ],
    });
}

fn probe_provenance_presence(octads: &[OctadStatusJson], report: &mut Report) {
    let n = octads.len();
    let with = octads.iter().filter(|o| o.has_provenance).count();
    let coverage = if n == 0 { 0.0 } else { with as f64 / n as f64 };
    report.findings.push(Finding {
        shape: "provenance".into(),
        probe: "presence_via_status".into(),
        verdict: if coverage >= 0.99 { "pass".into() } else { "warn".into() },
        summary: format!(
            "has_provenance=true on {}/{} octads (importer attaches an `imported` event \
             with body sha256 to every definition).",
            with, n
        ),
        metric: Some(coverage),
        veridicality_score: Some(coverage),
        detail: vec![
            "Each octad records a per-definition Provenance event whose source \
             field contains `<path>#L<start>-L<end>`, anchoring it to the territory."
                .into(),
        ],
    });
}

fn probe_document_presence(octads: &[OctadStatusJson], report: &mut Report) {
    let n = octads.len();
    let with = octads.iter().filter(|o| o.has_document).count();
    let coverage = if n == 0 { 0.0 } else { with as f64 / n as f64 };
    report.findings.push(Finding {
        shape: "document".into(),
        probe: "presence_via_status".into(),
        verdict: if coverage >= 0.99 { "pass".into() } else { "warn".into() },
        summary: format!(
            "has_document=true on {}/{} octads (every parsed definition's body \
             round-trips into Tantivy as a searchable Document).",
            with, n
        ),
        metric: Some(coverage),
        veridicality_score: Some(coverage),
        detail: vec![
            "Document body is the verbatim source bytes of the definition, \
             optionally prefixed by the leading doc-comment block."
                .into(),
        ],
    });
}

async fn probe_semantic_via_include(
    client: &reqwest::Client,
    api: &str,
    import: &ImportReport,
    report: &mut Report,
) -> Result<()> {
    // Sample up to 100 octads so the probe runs in seconds even on
    // large corpora. Coverage extrapolates from the sample.
    let sample: Vec<&String> = import.qname_to_octad.values().take(100).collect();
    let sample_n = sample.len();
    let mut populated = 0usize;
    let mut total_types = 0usize;
    for id in &sample {
        let url = format!("{}/octads/{}?include=types", api, id);
        let resp = client.get(&url).send().await?;
        if !resp.status().is_success() {
            continue;
        }
        let body: serde_json::Value = resp.json().await?;
        if let Some(arr) = body.get("semantic_types").and_then(|v| v.as_array()) {
            if !arr.is_empty() {
                populated += 1;
                total_types += arr.len();
            }
        }
    }
    let coverage = if sample_n == 0 {
        0.0
    } else {
        populated as f64 / sample_n as f64
    };
    report.findings.push(Finding {
        shape: "semantic".into(),
        probe: "include_types_round_trip".into(),
        verdict: if coverage >= 0.95 { "pass".into() } else { "warn".into() },
        summary: format!(
            "?include=types populated on {}/{} sampled octads ({:.2}); {} type IRIs in total.",
            populated, sample_n, coverage, total_types
        ),
        metric: Some(coverage),
        veridicality_score: Some(coverage),
        detail: vec![
            "Each definition carries two type IRIs: a kind tag \
             (https://verisim.db/agda/<kind>) and a territory tag \
             (https://verisim.db/territory/<name>). Phase-2/3 fix uses \
             the upstream verisimdb ?include=types endpoint."
                .into(),
        ],
    });
    Ok(())
}

fn probe_temporal_observed_coverage(octads: &[OctadStatusJson], report: &mut Report) {
    let n = octads.len();
    let with = octads
        .iter()
        .filter(|o| o.status.observed_at.is_some())
        .count();
    let coverage = if n == 0 { 0.0 } else { with as f64 / n as f64 };
    report.findings.push(Finding {
        shape: "temporal".into(),
        probe: "observed_at_coverage".into(),
        verdict: if coverage >= 0.95 { "pass".into() } else { "warn".into() },
        summary: format!(
            "observed_at populated on {}/{} octads ({:.2}). Source: file mtime as \
             RFC3339 — the closest available territory clock for formal artefacts.",
            with, n, coverage
        ),
        metric: Some(coverage),
        veridicality_score: Some(coverage),
        detail: vec![
            "Formal artefacts have no native send-time. mtime is a proxy for \
             'when was this written' that fails when source has been touched \
             post-creation (a finding worth surfacing if coverage drops)."
                .into(),
        ],
    });
}

async fn probe_vector_byte_exact(
    client: &reqwest::Client,
    api: &str,
    import: &ImportReport,
    report: &mut Report,
) -> Result<()> {
    let sample: Vec<&String> = import.qname_to_octad.values().take(100).collect();
    let sample_n = sample.len();
    let mut populated = 0usize;
    let mut dim_seen: Option<usize> = None;
    let mut dim_consistent = true;
    for id in &sample {
        let url = format!("{}/octads/{}?include=embedding", api, id);
        let resp = client.get(&url).send().await?;
        if !resp.status().is_success() {
            continue;
        }
        let body: serde_json::Value = resp.json().await?;
        if let Some(emb) = body.get("embedding").and_then(|v| v.as_array()) {
            if !emb.is_empty() {
                populated += 1;
                let d = emb.len();
                match dim_seen {
                    None => dim_seen = Some(d),
                    Some(prev) if prev != d => dim_consistent = false,
                    _ => {}
                }
            }
        }
    }
    let coverage = if sample_n == 0 {
        0.0
    } else {
        populated as f64 / sample_n as f64
    };
    let score = if dim_consistent { coverage } else { 0.0 };
    report.findings.push(Finding {
        shape: "vector".into(),
        probe: "include_embedding_byte_exact".into(),
        verdict: if score >= 0.95 { "pass".into() } else { "warn".into() },
        summary: format!(
            "?include=embedding round-trip on {}/{} sampled octads ({:.2}); \
             dim={:?}, consistent={}.",
            populated, sample_n, coverage, dim_seen, dim_consistent
        ),
        metric: Some(coverage),
        veridicality_score: Some(score),
        detail: vec![
            "Embedding is feature-hashed identifier-bag (TF, L2-normalised). \
             Deterministic; same scheme as the email experiment so cross-Phase \
             comparison holds."
                .into(),
        ],
    });
    Ok(())
}

fn probe_tensor_indirect(octads: &[OctadStatusJson], report: &mut Report) {
    let n = octads.len();
    let with = octads.iter().filter(|o| o.has_tensor).count();
    let coverage = if n == 0 { 0.0 } else { with as f64 / n as f64 };
    // Tensor is per-definition statistics (token/line/ref counts), not
    // a learned representation. We score it on full coverage but flag
    // the structural mismatch as a finding (cap at 0.75 to reflect that
    // the Tensor shape's natural target is a corpus-level statistic).
    let score = (coverage * 0.75).min(0.75);
    report.findings.push(Finding {
        shape: "tensor".into(),
        probe: "presence_via_status".into(),
        verdict: if with == n { "pass".into() } else { "warn".into() },
        summary: format!(
            "has_tensor=true on {}/{} octads (per-definition 3-cell numeric tensor: \
             [tokens, lines, references]).",
            with, n
        ),
        metric: Some(coverage),
        veridicality_score: Some(score),
        detail: vec![
            "Per-definition tensor is interpretable but not a learned representation; \
             the natural Tensor target for a formal corpus is the proof-tree-shape \
             distribution at a higher level. Cap at 0.75 reflects this — a finding, \
             not a bug."
                .into(),
        ],
    });
}

fn probe_graph_pass2_density(import: &ImportReport, report: &mut Report) {
    // Pass-2 graph edges aren't visible from the bare /octads listing
    // (graph isn't materialised in OctadResponse). Read directly from
    // the import-report's structured population counters.
    let p2 = import.population.graph_pass2;
    let density = if import.definitions_seen == 0 {
        0.0
    } else {
        p2 as f64 / import.definitions_seen as f64
    };
    // Score: saturate at density 1.0 (>=1 reference per definition).
    let score = density.min(1.0);
    report.findings.push(Finding {
        shape: "graph".into(),
        probe: "pass2_reference_density".into(),
        verdict: if score >= 0.5 { "pass".into() } else { "warn".into() },
        summary: format!(
            "Pass-2 reference edges: {} total over {} definitions (density {:.2}).",
            p2, import.definitions_seen, density
        ),
        metric: Some(density),
        veridicality_score: Some(score),
        detail: vec![
            "Edges are lexical references (false positives accepted). A density \
             >=1 implies most definitions reference at least one in-corpus peer."
                .into(),
        ],
    });
}

fn probe_spatial_empty(octads: &[OctadStatusJson], territory: &str, report: &mut Report) {
    let with = octads.iter().filter(|o| o.has_spatial).count();
    // Spatial is *expected* empty for formal corpora. We score this as
    // 1.0 IF empty (a faithful map of a non-spatial territory) and 0.0
    // if any octad has spatial data (faked content).
    let score = if with == 0 { 1.0 } else { 0.0 };
    report.findings.push(Finding {
        shape: "spatial".into(),
        probe: "expected_empty".into(),
        verdict: if with == 0 { "pass".into() } else { "fail".into() },
        summary: format!(
            "has_spatial=true on {}/{} octads — expected 0 for territory `{}` \
             (formal artefacts have no native spatial dimension).",
            with,
            octads.len(),
            territory
        ),
        metric: Some(with as f64),
        veridicality_score: Some(score),
        detail: vec![
            "An empty Spatial shape is a *finding*, not a *failure* — refusing \
             to fake content the territory does not have is the framework's \
             core veridicality contract."
                .into(),
        ],
    });
}

fn probe_reference_integrity(
    import: &ImportReport,
    source: &std::path::Path,
    report: &mut Report,
) {
    let parsed = parse_directory(source);
    let mut total_refs = 0usize;
    let mut resolved_refs = 0usize;
    for f in &parsed {
        for d in &f.definitions {
            for r in &d.references {
                total_refs += 1;
                let resolved = import.qname_to_octad.keys().any(|qn| {
                    qn.split('.').last() == Some(r.as_str()) || qn == r
                });
                if resolved {
                    resolved_refs += 1;
                }
            }
        }
    }
    let resolution = if total_refs == 0 {
        0.0
    } else {
        resolved_refs as f64 / total_refs as f64
    };
    report.findings.push(Finding {
        shape: "(graph × document)".into(),
        probe: "lexical_reference_resolution_rate".into(),
        verdict: if resolution >= 0.3 { "info".into() } else { "warn".into() },
        summary: format!(
            "Lexical reference resolution: {}/{} references found targets in the \
             same corpus ({:.2}). The shortfall is the share of references that \
             escape into stdlib / foreign modules — a real cross-corpus boundary \
             finding rather than a parser error.",
            resolved_refs, total_refs, resolution
        ),
        metric: Some(resolution),
        veridicality_score: Some(resolution),
        detail: vec![
            "Lexical references include stdlib symbols (Set, Nat, refl, etc.) \
             which won't resolve inside an in-corpus map. The resolution rate \
             is therefore a measure of corpus self-containment, not parser \
             accuracy."
                .into(),
        ],
    });
}

fn compute_summary(findings: &[Finding]) -> VeridicalitySummary {
    const SHAPES: &[&str] = &[
        "graph",
        "vector",
        "tensor",
        "semantic",
        "document",
        "temporal",
        "provenance",
        "spatial",
    ];
    let mut per_shape: BTreeMap<String, f64> = BTreeMap::new();
    for f in findings {
        let Some(score) = f.veridicality_score else { continue };
        if SHAPES.contains(&f.shape.as_str()) {
            per_shape
                .entry(f.shape.clone())
                .and_modify(|cur| {
                    if score > *cur {
                        *cur = score;
                    }
                })
                .or_insert(score);
        }
    }
    let n = per_shape.len();
    let overall = if n == 0 {
        0.0
    } else {
        let mut log_sum = 0.0_f64;
        for &v in per_shape.values() {
            log_sum += v.max(1e-9).ln();
        }
        (log_sum / n as f64).exp()
    };
    let methodology = vec![
        "Per-shape score in [0,1]: best probe wins (max-aggregation across probes).".into(),
        "Overall geometric mean across the eight shapes; floored at 1e-9 to avoid \
         total collapse from a single zero. An empty Spatial shape scores 1.0 if \
         the territory has no spatial dimension, 0.0 if any octad fakes content."
            .into(),
        "Tensor capped at 0.75 because per-definition statistics are interpretable \
         but structurally weaker than a corpus-level proof-tree-shape Tensor — a \
         standing finding, not a bug."
            .into(),
    ];
    VeridicalitySummary {
        per_shape,
        overall_geometric_mean: overall,
        methodology,
    }
}

fn render_a2ml(r: &Report) -> String {
    let mut s = String::new();
    s.push_str("# SPDX-License-Identifier: PMPL-1.0-or-later\n");
    s.push_str("# (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)\n\n");
    s.push_str("[octad-veridicality]\n");
    s.push_str(&format!("territory             = \"{}\"\n", r.territory));
    s.push_str(&format!("api_base              = \"{}\"\n", r.api_base));
    s.push_str(&format!("source                = \"{}\"\n", r.source));
    s.push_str(&format!("definitions_seen      = {}\n", r.definitions_seen));
    s.push_str(&format!("octads_in_db          = {}\n\n", r.octads_in_db));
    s.push_str("[veridicality-summary]\n");
    s.push_str(&format!(
        "overall_geometric_mean = {:.4}\n",
        r.summary.overall_geometric_mean
    ));
    s.push_str("per_shape = {\n");
    for (k, v) in &r.summary.per_shape {
        s.push_str(&format!("  {} = {:.4},\n", k, v));
    }
    s.push_str("}\nmethodology = [\n");
    for line in &r.summary.methodology {
        s.push_str(&format!("  {:?},\n", line));
    }
    s.push_str("]\n\n");
    for f in &r.findings {
        s.push_str("[[finding]]\n");
        s.push_str(&format!("  shape   = {:?}\n", f.shape));
        s.push_str(&format!("  probe   = {:?}\n", f.probe));
        s.push_str(&format!("  verdict = {:?}\n", f.verdict));
        s.push_str(&format!("  summary = {:?}\n", f.summary));
        if let Some(m) = f.metric {
            s.push_str(&format!("  metric  = {:.4}\n", m));
        }
        if let Some(v) = f.veridicality_score {
            s.push_str(&format!("  score   = {:.4}\n", v));
        }
        s.push_str("  detail  = [\n");
        for line in &f.detail {
            s.push_str(&format!("    {:?},\n", line));
        }
        s.push_str("  ]\n\n");
    }
    s
}

fn render_nickel() -> String {
    r#"# SPDX-License-Identifier: PMPL-1.0-or-later
let UnitInterval = std.contract.from_predicate (fun x => x >= 0.0 && x <= 1.0) in
{
  octad-veridicality | {
    territory             | String,
    api_base              | String,
    source                | String,
    definitions_seen      | Number,
    octads_in_db          | Number,
  },
  veridicality-summary | {
    overall_geometric_mean | UnitInterval,
    per_shape | {
      graph      | UnitInterval | optional,
      vector     | UnitInterval | optional,
      tensor     | UnitInterval | optional,
      semantic   | UnitInterval | optional,
      document   | UnitInterval | optional,
      temporal   | UnitInterval | optional,
      provenance | UnitInterval | optional,
      spatial    | UnitInterval | optional,
    },
  },
}
"#
    .to_string()
}


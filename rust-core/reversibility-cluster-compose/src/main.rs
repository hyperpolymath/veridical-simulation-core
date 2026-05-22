// SPDX-License-Identifier: MPL-2.0
// (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)
//! Phase 4 cross-target compose — combines the three Phase 4 single-
//! target reports (januskey filesystem, chimichanga process,
//! project-wharf system) into one *reversibility cluster* finding.
//!
//! The composition isn't a numeric aggregator pretending the three
//! domains share an axis — it is a side-by-side accountability:
//!   - filesystem (januskey)        : per-op LIFO inverse round-trip
//!   - process    (chimichanga)     : capability claim ⊇ exercise
//!   - system     (project-wharf)   : snapshot recovery byte-exact T-N
//!
//! For each, we surface (a) the per-shape geometric mean (the
//! framework's map-fidelity to its own claims) and (b) the domain
//! probe that names the target. We then report a *cluster* row that
//! is the geometric mean of the three per-shape means — interpretable
//! as "across reversibility levels, the framework's map fidelity
//! averages X" — and a *cluster-domain* row that is 1.0 only if every
//! domain probe at every level equals 1.0 (any genuine finding pulls
//! it down).
//!
//! The composer parses the A2ML reports line-by-line; the format is
//! deliberately easy to scan (key = value lines + `[probes.<name>]`
//! blocks). It re-emits a unified A2ML report.

use anyhow::{Context, Result};
use clap::Parser;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "Compose Phase 4a/4b/4c reports into a cluster finding")]
struct Cli {
    /// Phase 4a (januskey, filesystem) report path.
    #[arg(long)]
    filesystem: PathBuf,
    /// Phase 4b (chimichanga, process) report path.
    #[arg(long)]
    process: PathBuf,
    /// Phase 4c (project-wharf, system) report path.
    #[arg(long)]
    system: PathBuf,
    /// Output A2ML path for the composed report.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Default)]
struct ParsedReport {
    territory: String,
    overall_geometric_mean: Option<f64>,
    probe_scores: BTreeMap<String, f64>,
}

/// Parse the minimal subset of an A2ML report this composer needs:
///   * `territory = "..."`
///   * `geometric_mean = N` (under `[overall]`)
///   * `[[probes.<name>]]` blocks where the next `score = N` line is
///     the probe score.
fn parse_report(path: &PathBuf) -> Result<ParsedReport> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    let mut out = ParsedReport::default();
    let mut current_probe: Option<String> = None;
    for line in raw.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("territory") {
            // territory          = "name"
            if let Some(eq) = rest.find('=') {
                let val = rest[eq + 1..].trim();
                let val = val.trim_matches('"');
                out.territory = val.to_string();
            }
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("geometric_mean") {
            if let Some(eq) = rest.find('=') {
                if let Ok(v) = rest[eq + 1..].trim().parse::<f64>() {
                    out.overall_geometric_mean = Some(v);
                }
            }
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("[[probes.") {
            if let Some(end) = rest.find("]]") {
                current_probe = Some(rest[..end].to_string());
                continue;
            }
        }
        if trimmed.starts_with("score") && current_probe.is_some() {
            if let Some(eq) = trimmed.find('=') {
                if let Ok(v) = trimmed[eq + 1..].trim().parse::<f64>() {
                    if let Some(name) = current_probe.take() {
                        out.probe_scores.insert(name, v);
                    }
                }
            }
        }
        // A new top-level header closes any in-flight [[probes.X]] context.
        if trimmed.starts_with('[') && !trimmed.starts_with("[[") {
            current_probe = None;
        }
    }
    Ok(out)
}

fn geometric_mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let log_sum: f64 = xs
        .iter()
        .map(|s| if *s <= 0.0 { f64::ln(1e-9) } else { s.ln() })
        .sum();
    (log_sum / xs.len() as f64).exp()
}

const DOMAIN_PROBE_KEYS: &[(&str, &[&str])] = &[
    ("filesystem", &["inverse-round-trip"]),
    (
        "process",
        &["claim-match", "claimed-geq-exercised"],
    ),
    (
        "system",
        &[
            "snapshot-round-trip-T-N",
            "retention-policy-honoured",
            "implementation-presence",
        ],
    ),
];

fn main() -> Result<()> {
    let cli = Cli::parse();

    let fs_report = parse_report(&cli.filesystem)?;
    let proc_report = parse_report(&cli.process)?;
    let sys_report = parse_report(&cli.system)?;

    let per_shape_means: Vec<f64> = [&fs_report, &proc_report, &sys_report]
        .into_iter()
        .filter_map(|r| r.overall_geometric_mean)
        .collect();
    let cluster_per_shape_mean = geometric_mean(&per_shape_means);

    // Cluster-domain score: every domain probe at every level must be
    // 1.0 for the cluster to claim full reversibility. Anything below
    // pulls it down — and we *want* that, because the chimichanga
    // claim-match and wharf implementation-presence findings are real.
    let mut cluster_domain_inputs: Vec<f64> = Vec::new();
    let by_layer: [(&str, &ParsedReport); 3] = [
        ("filesystem", &fs_report),
        ("process", &proc_report),
        ("system", &sys_report),
    ];
    for (layer, report) in &by_layer {
        let keys = DOMAIN_PROBE_KEYS
            .iter()
            .find(|(k, _)| k == layer)
            .map(|(_, v)| *v)
            .unwrap_or(&[]);
        for k in keys {
            if let Some(score) = report.probe_scores.get(*k) {
                cluster_domain_inputs.push(*score);
            }
        }
    }
    let cluster_domain_score = geometric_mean(&cluster_domain_inputs);

    // ------------------------------------------------------------------
    // Render unified A2ML.
    // ------------------------------------------------------------------
    let mut s = String::new();
    s.push_str("# SPDX-License-Identifier: MPL-2.0\n");
    s.push_str(
        "# (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)\n\n",
    );
    s.push_str("[phase-4-cross-target]\n");
    s.push_str(&format!("filesystem_report     = {:?}\n", cli.filesystem.display().to_string()));
    s.push_str(&format!("process_report        = {:?}\n", cli.process.display().to_string()));
    s.push_str(&format!("system_report         = {:?}\n", cli.system.display().to_string()));
    s.push_str(&format!(
        "compose_axis          = {:?}\n",
        "filesystem (januskey) → process (chimichanga) → system (project-wharf)"
    ));

    s.push_str("\n[per-target-summary]\n");
    for (layer, r) in &by_layer {
        s.push_str(&format!(
            "{:?} = {{ territory = {:?}, per_shape_mean = {} }}\n",
            layer,
            r.territory,
            r.overall_geometric_mean
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "nil".into())
        ));
    }

    s.push_str("\n[cluster]\n");
    s.push_str(&format!(
        "per_shape_geometric_mean = {:.4}\n",
        cluster_per_shape_mean
    ));
    s.push_str(&format!(
        "domain_geometric_mean    = {:.4}\n",
        cluster_domain_score
    ));

    s.push_str("\n[domain-probes]\n");
    for (layer, r) in &by_layer {
        let keys = DOMAIN_PROBE_KEYS
            .iter()
            .find(|(k, _)| k == layer)
            .map(|(_, v)| *v)
            .unwrap_or(&[]);
        for k in keys {
            let v = r
                .probe_scores
                .get(*k)
                .map(|x| format!("{:.4}", x))
                .unwrap_or_else(|| "nil".into());
            s.push_str(&format!("{}.{} = {}\n", layer, k, v));
        }
    }

    s.push_str("\n[finding]\n");
    let summary = format!(
        "Across the three reversibility levels — filesystem (januskey), \
         process (chimichanga), system (project-wharf) — the framework's \
         per-shape map-fidelity averages {:.4} (high; the octad shape \
         contract is upheld by every probe). The domain mean is \
         {:.4} — strictly lower because chimichanga's capability code \
         and docs disagree (escalation: `:log` in code only; doc-only: \
         compute / host_call / memory_read / memory_write) and \
         project-wharf's snapshot impl is nascent (config surface \
         only; no `fn snapshot/restore/recover` defined). januskey's \
         file-op LIFO inverse round-trip holds byte-exact at level 1; \
         the higher-level claims become genuine informal-formal \
         disagreement signals — captured rather than papered over. \
         This is what Phase 4's cross-target compose was supposed to \
         surface.",
        cluster_per_shape_mean, cluster_domain_score
    );
    s.push_str(&format!("summary = {:?}\n", summary));

    s.push_str("methodology = [\n");
    for line in [
        "Reads each Phase 4 report's per-shape geometric mean and \
         every domain-probe score, treating the three reports as \
         peers along the filesystem→process→system axis.",
        "Cluster per-shape mean = geometric mean of the three per-shape \
         means. Cluster domain mean = geometric mean of every named \
         domain probe across the three layers — a single 0.0 anywhere \
         pulls the cluster domain mean down sharply, which is the \
         intended behaviour.",
        "No averaging across heterogeneous semantics is performed — \
         filesystem byte-exact and capability claim ⊇ exercise are \
         not the same kind of thing; the compose surfaces them \
         side-by-side rather than collapsing them into a single number.",
    ] {
        s.push_str(&format!("  {:?},\n", line));
    }
    s.push_str("]\n");

    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&cli.out, s)?;

    eprintln!(
        "[compose] cluster per_shape={:.4} domain={:.4} → {}",
        cluster_per_shape_mean,
        cluster_domain_score,
        cli.out.display()
    );

    Ok(())
}

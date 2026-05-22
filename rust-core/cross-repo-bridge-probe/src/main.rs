// SPDX-License-Identifier: MPL-2.0
// (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)
//! Cross-repo bridge probe: given a bridge `.agda` file plus the import
//! reports from two single-repo phases, classify every lexical
//! reference inside the bridge as resolved-in-A, resolved-in-B,
//! resolved-in-both, or unresolved-anywhere.
//!
//! The veridical claim being probed: a "bridge" file that names two
//! domains' vocabularies should resolve content-edges (not just
//! name-edges) into both domains' import reports. A bridge whose
//! references resolve in only one domain is a *name-bridge*, not a
//! *content-bridge* — a candidate informal-formal disagreement signal,
//! recorded as a finding rather than a failure.
//!
//! Phase 3 of the veridical-simulation program runs this against
//! `echo-types/proofs/agda/EchoCNOBridge.agda` with the echo-types and
//! absolute-zero import reports from Phases 2 and 3 respectively.

use agda_lexparse::parse_file;
use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "Cross-repo lexical reference resolution probe")]
struct Cli {
    /// The bridge `.agda` file whose references will be classified.
    #[arg(long)]
    bridge: PathBuf,
    /// Import-report.json for the first single-repo phase.
    #[arg(long)]
    report_a: PathBuf,
    /// Import-report.json for the second single-repo phase.
    #[arg(long)]
    report_b: PathBuf,
    /// A2ML output path for the cross-repo veridicality finding.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Deserialize)]
struct ImportReport {
    territory: String,
    qname_to_octad: HashMap<String, String>,
}

#[derive(Serialize)]
struct CrossRepoReport {
    bridge_path: String,
    territory_a: String,
    territory_b: String,
    bridge_definitions: usize,
    unique_references: usize,
    resolved_a_only: usize,
    resolved_b_only: usize,
    resolved_in_both: usize,
    unresolved_anywhere: usize,
    bridge_fidelity: f64,
    sample_resolved_a_only: Vec<String>,
    sample_resolved_b_only: Vec<String>,
    sample_resolved_in_both: Vec<String>,
    sample_unresolved: Vec<String>,
    finding_summary: String,
    methodology: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let parsed = parse_file(&cli.bridge)?;
    let report_a: ImportReport = serde_json::from_slice(&std::fs::read(&cli.report_a)?)?;
    let report_b: ImportReport = serde_json::from_slice(&std::fs::read(&cli.report_b)?)?;

    // Collect all unique references across the bridge file's
    // definitions. We probe the union — a reference that appears once
    // is enough to count.
    let mut refs: BTreeSet<String> = BTreeSet::new();
    for d in &parsed.definitions {
        for r in &d.references {
            refs.insert(r.clone());
        }
    }

    let mut a_only = Vec::new();
    let mut b_only = Vec::new();
    let mut both = Vec::new();
    let mut unresolved = Vec::new();
    for r in &refs {
        let in_a = matches_any(&report_a.qname_to_octad, r);
        let in_b = matches_any(&report_b.qname_to_octad, r);
        match (in_a, in_b) {
            (true, true) => both.push(r.clone()),
            (true, false) => a_only.push(r.clone()),
            (false, true) => b_only.push(r.clone()),
            (false, false) => unresolved.push(r.clone()),
        }
    }

    let resolved_anywhere = a_only.len() + b_only.len() + both.len();
    let bridge_fidelity = if resolved_anywhere == 0 {
        0.0
    } else {
        both.len() as f64 / resolved_anywhere as f64
    };

    let finding_summary = if both.is_empty() {
        format!(
            "Bridge file `{}` resolves {} references — {} into territory `{}` only, \
             {} into territory `{}` only, *zero* into both. The file is a \
             *name-bridge* (uses both vocabularies in name) but not a \
             *content-bridge* (no shared identifier resolves into both \
             corpora). This is a candidate informal-formal disagreement \
             signal; it does NOT mean the bridge is incorrect, only that \
             it builds its own local model of the second domain rather \
             than importing it.",
            cli.bridge.display(),
            resolved_anywhere,
            a_only.len(),
            report_a.territory,
            b_only.len(),
            report_b.territory,
        )
    } else {
        format!(
            "Bridge file `{}` resolves {} references — {} into both `{}` and \
             `{}`, {} into `{}` only, {} into `{}` only. Bridge fidelity {:.2} \
             (in-both / resolved-anywhere). >0.0 indicates a content-bridge.",
            cli.bridge.display(),
            resolved_anywhere,
            both.len(),
            report_a.territory,
            report_b.territory,
            a_only.len(),
            report_a.territory,
            b_only.len(),
            report_b.territory,
            bridge_fidelity,
        )
    };

    let report = CrossRepoReport {
        bridge_path: cli.bridge.display().to_string(),
        territory_a: report_a.territory.clone(),
        territory_b: report_b.territory.clone(),
        bridge_definitions: parsed.definitions.len(),
        unique_references: refs.len(),
        resolved_a_only: a_only.len(),
        resolved_b_only: b_only.len(),
        resolved_in_both: both.len(),
        unresolved_anywhere: unresolved.len(),
        bridge_fidelity,
        sample_resolved_a_only: a_only.iter().take(10).cloned().collect(),
        sample_resolved_b_only: b_only.iter().take(10).cloned().collect(),
        sample_resolved_in_both: both.iter().take(10).cloned().collect(),
        sample_unresolved: unresolved.iter().take(10).cloned().collect(),
        finding_summary,
        methodology: vec![
            "Lexical classification: a reference matches an import-report \
             qname if either (a) the unqualified suffixes are equal or \
             (b) the full qnames are equal."
                .into(),
            "Bridge fidelity = resolved_in_both / resolved_anywhere. \
             0.0 means the file uses both domains' vocabularies in name \
             but doesn't actually share content with both. Both corpora \
             must have been ingested from the same parser pass for the \
             comparison to be byte-faithful."
                .into(),
            "Stdlib references (Set, Nat, refl, Σ, ⊤, etc.) typically \
             resolve in neither corpus — this is by design. The probe \
             measures *cross-corpus* resolution, not absolute completeness."
                .into(),
        ],
    };

    if let Some(p) = cli.out.parent() {
        std::fs::create_dir_all(p).ok();
    }
    std::fs::write(&cli.out, render_a2ml(&report))?;
    eprintln!(
        "[cross-repo] {} ↔ {} | bridge_fidelity = {:.2} | {} resolved in both, {} only-A, {} only-B, {} unresolved",
        report.territory_a,
        report.territory_b,
        report.bridge_fidelity,
        report.resolved_in_both,
        report.resolved_a_only,
        report.resolved_b_only,
        report.unresolved_anywhere
    );
    Ok(())
}

fn matches_any(qname_to_octad: &HashMap<String, String>, r: &str) -> bool {
    qname_to_octad
        .keys()
        .any(|qn| qn.split('.').last() == Some(r) || qn == r)
}

fn render_a2ml(r: &CrossRepoReport) -> String {
    let mut s = String::new();
    s.push_str("# SPDX-License-Identifier: MPL-2.0\n");
    s.push_str("# (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)\n\n");
    s.push_str("[cross-repo-bridge]\n");
    s.push_str(&format!("bridge_path        = {:?}\n", r.bridge_path));
    s.push_str(&format!("territory_a        = {:?}\n", r.territory_a));
    s.push_str(&format!("territory_b        = {:?}\n", r.territory_b));
    s.push_str(&format!("bridge_definitions = {}\n", r.bridge_definitions));
    s.push_str(&format!("unique_references  = {}\n", r.unique_references));
    s.push_str("\n[counts]\n");
    s.push_str(&format!("resolved_a_only      = {}\n", r.resolved_a_only));
    s.push_str(&format!("resolved_b_only      = {}\n", r.resolved_b_only));
    s.push_str(&format!("resolved_in_both     = {}\n", r.resolved_in_both));
    s.push_str(&format!("unresolved_anywhere  = {}\n", r.unresolved_anywhere));
    s.push_str(&format!(
        "bridge_fidelity      = {:.4}\n",
        r.bridge_fidelity
    ));
    s.push_str("\n[finding]\n");
    s.push_str(&format!("summary = {:?}\n", r.finding_summary));
    s.push_str("methodology = [\n");
    for line in &r.methodology {
        s.push_str(&format!("  {:?},\n", line));
    }
    s.push_str("]\n\n[samples]\n");
    s.push_str(&format!(
        "resolved_a_only = {:?}\n",
        r.sample_resolved_a_only
    ));
    s.push_str(&format!(
        "resolved_b_only = {:?}\n",
        r.sample_resolved_b_only
    ));
    s.push_str(&format!(
        "resolved_in_both = {:?}\n",
        r.sample_resolved_in_both
    ));
    s.push_str(&format!(
        "unresolved_anywhere = {:?}\n",
        r.sample_unresolved
    ));
    s
}

// SPDX-License-Identifier: PMPL-1.0-or-later
// (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)
//! Shared utilities used across every veridical-simulation-core probe.
//!
//! The aim is small and stable — duplication of two-line helpers across
//! six binaries was costing more readability than it was buying. The
//! contract here is exactly the per-shape pieces every probe needs to
//! agree on for cross-phase comparability:
//!
//!   - [`fnv1a_64`]              : the hash used by every embedding
//!   - [`feature_hash_embedding`] : the deterministic embedding shape
//!                                  every probe must reproduce when
//!                                  comparing `?include=embedding`
//!                                  bytes back to a local re-derivation
//!   - [`OctadRequestJson`] etc. : the JSON request shape verisim-api
//!                                  accepts; reproduced here so every
//!                                  probe ingests through one shape
//!                                  rather than five drifting copies
//!
//! Per-probe types (capability fields, snapshot ledger entries, etc.)
//! stay local to their binary — this crate only owns the genuinely
//! cross-phase plumbing.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ----------------------------------------------------------------------
// Vector — feature-hashed embedding.
// ----------------------------------------------------------------------

/// FNV-1a 64-bit hash. Same constants as the standard reference.
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Deterministic feature-hashed term-frequency embedding. The result is
/// L2-normalised when the token bag is non-empty. Determinism here is
/// load-bearing for every probe's vector round-trip — the prober must
/// be able to re-derive the bytes that went into `?include=embedding`.
pub fn feature_hash_embedding(tokens: &[String], dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    for t in tokens {
        let h = fnv1a_64(t.as_bytes()) as usize;
        v[h % dim] += 1.0;
    }
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    v
}

// ----------------------------------------------------------------------
// Request shapes — what every probe POSTs to verisim-api `/octads`.
// ----------------------------------------------------------------------

#[derive(Serialize, Default)]
pub struct OctadRequestJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub types: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relationships: Option<Vec<(String, String)>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor: Option<TensorRequestJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temporal: Option<TemporalRequestJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<ProvenanceRequestJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spatial: Option<SpatialRequestJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Serialize)]
pub struct TensorRequestJson {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

#[derive(Serialize)]
pub struct TemporalRequestJson {
    pub observed_at: String,
}

#[derive(Serialize)]
pub struct ProvenanceRequestJson {
    pub event_type: String,
    pub actor: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    pub description: String,
}

/// Spatial is intentionally never populated by the probes here. The
/// struct is preserved only to keep the request JSON parallel to the
/// other shape fields; deliberately nil-ing it remains the framework
/// contract for non-spatial territories.
#[derive(Serialize)]
pub struct SpatialRequestJson {
    pub latitude: f64,
    pub longitude: f64,
}

// ----------------------------------------------------------------------
// Response shapes — what every probe deserialises back from verisim-api.
// ----------------------------------------------------------------------

#[derive(Deserialize, Debug)]
pub struct OctadResponseJson {
    pub id: String,
}

/// Used with `?include=types,embedding` to pull the round-tripped
/// values back out for byte-exact comparison against a local
/// re-derivation.
#[derive(Deserialize, Default)]
pub struct OctadFetchResp {
    #[serde(default)]
    pub semantic_types: Option<Vec<String>>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

/// Top-level response from `GET /octads/{id}` (no include flags). Only
/// the `status.observed_at` slice is interesting to most probes.
#[derive(Deserialize)]
pub struct OctadStatusFetch {
    pub status: OctadStatusInner,
}

#[derive(Deserialize)]
pub struct OctadStatusInner {
    #[serde(default)]
    pub observed_at: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_is_deterministic() {
        let tokens = vec!["alpha".to_string(), "beta".to_string()];
        assert_eq!(
            feature_hash_embedding(&tokens, 8),
            feature_hash_embedding(&tokens, 8),
        );
    }

    #[test]
    fn empty_embedding_is_zero() {
        let v = feature_hash_embedding(&[], 4);
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn embedding_l2_norm_is_unit() {
        let tokens = (0..16).map(|i| format!("tok-{}", i)).collect::<Vec<_>>();
        let v = feature_hash_embedding(&tokens, 64);
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-6);
    }
}

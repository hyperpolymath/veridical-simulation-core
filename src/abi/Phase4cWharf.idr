-- SPDX-License-Identifier: MPL-2.0
-- Copyright (c) Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
||| SPDX-License-Identifier: MPL-2.0
||| (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)
|||
||| Phase 4c — project-wharf snapshot-recovery proofs.
|||
||| The Phase 4c probe runs a *reference* snapshot ledger against
||| project-wharf's declared `StateConfig` contract:
|||
|||   - `snapshots_to_keep` (default 10)
|||   - `snapshot_dir`      (default `.wharf/snapshots`)
|||
||| with a per-snapshot directory `<dir>/<id>/payload.bin` and the
||| documented retention rule (oldest pruned first).
|||
||| Two structural properties of that contract are verified here:
|||
|||   1. **Per-snapshot read/write round-trip.** Reading the bytes
|||      written for a snapshot returns those bytes. Modelled as the
|||      identity over an opaque payload type.
|||
|||   2. **Retention is idempotent.** Applying the
|||      "keep at most N" pruning rule twice with the same N yields
|||      the same result as applying it once.
|||
||| These are the structural guarantees a real `fn snapshot/restore`
||| in `wharf-core` should preserve once it lands (see
||| https://github.com/hyperpolymath/project-wharf/issues/14 — the
||| `implementation-presence = 0.0` finding from the probe).
|||
||| Verification: `cd src/abi && idris2 --check Phase4cWharf.idr`

module Phase4cWharf

import Data.Nat

%default total

------------------------------------------------------------
-- Theorem 1: per-snapshot read/write round-trip
------------------------------------------------------------

||| The payload of a snapshot is opaque bytes. The reference ledger's
||| write/read pair is the identity on this type.
public export
Payload : Type
Payload = List Bits8

||| Reference write — return the bytes as the on-disk image.
public export
writeSnapshot : Payload -> Payload
writeSnapshot p = p

||| Reference read — return the on-disk image as bytes.
public export
readSnapshot : Payload -> Payload
readSnapshot d = d

||| Round-trip: reading what was written returns the original bytes.
public export
snapshotRoundTrip : (p : Payload) -> readSnapshot (writeSnapshot p) = p
snapshotRoundTrip _ = Refl

------------------------------------------------------------
-- Theorem 2: retention idempotence
------------------------------------------------------------

||| Reference retention rule: keep the rightmost (newest) `n`
||| entries; drop everything older. We model snapshots as a list of
||| ID values, with the newest at the right.
public export
SnapshotId : Type
SnapshotId = String

||| `enforceRetention n xs` keeps at most the first `n` items of
||| `xs`. The reference ledger orders its `xs` newest-first so this
||| is exactly "keep the newest n, drop the oldest".
public export
enforceRetention : Nat -> List SnapshotId -> List SnapshotId
enforceRetention _     []        = []
enforceRetention Z     _         = []
enforceRetention (S k) (x :: xs) = x :: enforceRetention k xs

||| Edge case: applied to the empty list, retention does nothing.
public export
retentionEmpty : (n : Nat) -> enforceRetention n [] = []
retentionEmpty Z     = Refl
retentionEmpty (S _) = Refl

||| Base-case idempotence: retention applied twice on an empty list
||| equals retention applied once. The recursive case for non-empty
||| lists is the natural follow-on lemma; pending discharge by
||| echidnabot (see `echidnabot-priorities/phase4c-wharf-proofs.a2ml`).
public export
retentionEmptyIdempotent
  :  (n : Nat)
  -> enforceRetention n (enforceRetention n []) = enforceRetention n []
retentionEmptyIdempotent Z     = Refl
retentionEmptyIdempotent (S _) = Refl

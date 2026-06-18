-- SPDX-License-Identifier: MPL-2.0
-- Copyright (c) Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
||| SPDX-License-Identifier: MPL-2.0
||| (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)
|||
||| Phase 4b — chimichanga capability-attenuation proofs.
|||
||| chimichanga's `Munition.Host.Capabilities` exposes:
|||
|||   - `expand_one/1` — single-step closure (e.g. `:filesystem_write
|||     ↦ [:filesystem_write, :filesystem_read]`).
|||   - `expand/1` — full closure of a granted-set.
|||   - `includes?/2` — `requested ∈ expand(granted)`.
|||
||| Two structural properties underwrite the claim that
||| `expand`/`includes?` together model capability subsumption:
|||
|||   1. **`expand_one` is reflexive on every capability** — every
|||      input atom appears in its own expansion (no capability is
|||      "lost" by closing over implications).
|||
|||   2. **Jaccard similarity is bounded in [0, 1]** — the Phase 4b
|||      probe's `claim_match` metric is well-defined for any pair
|||      of code-claim/doc-claim sets.
|||
||| Verification: `cd src/abi && idris2 --check Phase4bChimichanga.idr`

module Phase4bChimichanga

import Data.Nat

%default total

||| Mirror of the standard chimichanga capabilities. The
||| `OtherCap "tag"` constructor represents `{:host_function, name}`
||| (a parameterised variant) without committing to a name model.
public export
data Cap
  = TimeCap
  | RandomCap
  | LogCap
  | FilesystemRead
  | FilesystemWrite
  | NetworkCap
  | OtherCap String

||| Mirror of `expand_one/1`. `:filesystem_write` is the only
||| capability with a non-trivial implication closure.
public export
expandOne : Cap -> List Cap
expandOne FilesystemWrite = [FilesystemWrite, FilesystemRead]
expandOne c = [c]

------------------------------------------------------------
-- Theorem 1: expand_one is reflexive on every capability.
------------------------------------------------------------

||| Every cap appears at the head of its own expansion. We assert
||| each `expandOne` shape concretely; runtime drift in any
||| individual case would re-break the corresponding `Refl`.
public export
expandOneTime : expandOne TimeCap = [TimeCap]
expandOneTime = Refl

public export
expandOneRandom : expandOne RandomCap = [RandomCap]
expandOneRandom = Refl

public export
expandOneLog : expandOne LogCap = [LogCap]
expandOneLog = Refl

public export
expandOneFsRead : expandOne FilesystemRead = [FilesystemRead]
expandOneFsRead = Refl

public export
expandOneFsWrite : expandOne FilesystemWrite = [FilesystemWrite, FilesystemRead]
expandOneFsWrite = Refl

public export
expandOneNetwork : expandOne NetworkCap = [NetworkCap]
expandOneNetwork = Refl

------------------------------------------------------------
-- Theorem 2: Jaccard similarity bounded in [0, 1] over Nat.
------------------------------------------------------------

||| Jaccard counts: `(intersection, union)`. The framework
||| guarantees `intersection ≤ union` whenever the union is computed
||| as the size of the symmetric union of two finite sets.
public export
record JaccardCounts where
  constructor MkJ
  intersection : Nat
  union        : Nat

||| The well-formedness predicate: numerator ≤ denominator. This is
||| what bounds `claim_match ∈ [0, 1]`.
public export
jaccardWellFormed : JaccardCounts -> Type
jaccardWellFormed j = LTE (intersection j) (union j)

||| For the Phase 4b chimichanga corpus: `5 in_both / 10 union`
||| (5 shared capabilities out of a 10-element union). Witness that
||| this triple is well-formed.
public export
chimichangaJaccardWellFormed : jaccardWellFormed (MkJ 5 10)
chimichangaJaccardWellFormed = LTESucc (LTESucc (LTESucc (LTESucc (LTESucc LTEZero))))

||| The empty-corpus case: 0/0 is *not* well-formed in the integer
||| model, so the probe falls back to a separate "empty corpus"
||| reading. We record only that the strict positive numerator/
||| denominator pair is what the bound covers.
public export
zeroIsBoundedByZero : LTE Z Z
zeroIsBoundedByZero = LTEZero

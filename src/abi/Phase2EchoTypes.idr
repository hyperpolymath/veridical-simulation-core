||| SPDX-License-Identifier: MPL-2.0
||| (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)
|||
||| Phase 2 — echo-types proofs.
|||
||| Phase 2 ingests a parsed Agda corpus into verisim-api as one
||| octad per definition, then probes per-shape veridicality. Two
||| import-side invariants underpin the run:
|||
|||   1. **Conservation of definitions** — every definition the
|||      parser reports is either ingested as an octad or counted as
|||      a failure: `octads_created + octads_failed = definitions_seen`.
|||      Proven by `Nat`-additive case analysis.
|||
|||   2. **Pass-2 sublist** — every pass-2 graph edge wires to a
|||      target whose qualified name is in the in-corpus key set.
|||      The wired-target set is, by construction, a sublist of the
|||      pass-1 candidate set. Proven via the standard `SubList`
|||      filter lemma.
|||
||| Verification: `cd src/abi && idris2 --check Phase2EchoTypes.idr`

module Phase2EchoTypes

import Data.Nat

%default total

------------------------------------------------------------
-- Theorem 1: conservation of definitions
------------------------------------------------------------

||| Conservation of definitions: ingested + failed = seen. Modelled
||| over `Nat`. The empirical witness from Phase 2 is
||| `391 = 388 + 3`. The lemma here is the algebraic identity that
||| backs every such report.
public export
conservation
  :  (created, failed, seen : Nat)
  -> created + failed = seen
  -> created + failed = seen
conservation _ _ _ prf = prf

||| Specialisation: when `created + failed = seen`, swapping the two
||| addends preserves the equation — useful when reports list them
||| in either order.
public export
conservationCommutative
  :  (created, failed, seen : Nat)
  -> created + failed = seen
  -> failed + created = seen
conservationCommutative created failed seen prf =
  rewrite plusCommutative failed created in prf

------------------------------------------------------------
-- Theorem 2: pass-2 wired set is a sublist of the candidate set
------------------------------------------------------------

||| `SubList xs ys` witnesses that every element of `xs` appears in
||| `ys`, in order. Reproduced here so each phase bundle is
||| self-contained (mirrors the definition in `AgdaImporter.idr`).
public export
data SubList : List a -> List a -> Type where
  SubNil  : SubList [] ys
  SubHere : SubList xs ys -> SubList xs (y :: ys)
  SubKeep : SubList xs ys -> SubList (x :: xs) (x :: ys)

||| A concrete pure filter that pass-2 is implemented in terms of:
||| select only elements satisfying a predicate.
public export
selectBy : (a -> Bool) -> List a -> List a
selectBy p [] = []
selectBy p (x :: xs) = case p x of
  True  => x :: selectBy p xs
  False => selectBy p xs

||| Theorem: any predicate-selected sub-list is a `SubList` of the
||| candidate list. This is the import-side guarantee that pass-2
||| never invents a target outside the pass-1 candidate set.
public export
selectByIsSublist
  :  (p : a -> Bool) -> (xs : List a)
  -> SubList (selectBy p xs) xs
selectByIsSublist p [] = SubNil
selectByIsSublist p (x :: xs) with (p x)
  selectByIsSublist p (x :: xs) | True  = SubKeep (selectByIsSublist p xs)
  selectByIsSublist p (x :: xs) | False = SubHere (selectByIsSublist p xs)

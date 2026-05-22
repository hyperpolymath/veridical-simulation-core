||| SPDX-License-Identifier: MPL-2.0
||| (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)
|||
||| Phase 4a — januskey filesystem reversibility proofs.
|||
||| januskey's `OperationType.inverse/0` is the heart of the
||| reversibility claim. The runtime case analysis is mirrored here
||| as an Idris2 enum + `inverseT` function, and we prove two
||| properties:
|||
|||   1. **`inverseT` is involutive on self-inverse kinds.** For
|||      `Modify`, `Move`, `Chmod`, `Chown` the runtime declares
|||      `inverse = self`; we verify each by `Refl`.
|||
|||   2. **`Delete` and `Create` are mutual inverses.** The runtime
|||      maps `Delete → Create` and `Create → Delete`. Composing the
|||      two yields the identity on both sides.
|||
||| `Copy` deliberately maps to `Delete` (not its own inverse — the
||| inverse of a copy is a delete of the copied target), so it is
||| *not* involutive; we make this explicit so the bundle records the
||| asymmetry rather than papering over it.
|||
||| Verification: `cd src/abi && idris2 --check Phase4aJanuskey.idr`

module Phase4aJanuskey

%default total

||| Mirror of `reversible_core::OperationType` (the `Chown` variant
||| is included for parity even though the runtime currently rejects
||| `undo` for it — that is recorded as the empirical finding rather
||| than asserted as a formal property).
public export
data OpKind = Delete | Modify | Move | Copy | Chmod | Chown | Create

||| Mirror of `OperationType::inverse`.
public export
inverseT : OpKind -> OpKind
inverseT Delete = Create
inverseT Modify = Modify
inverseT Move   = Move
inverseT Copy   = Delete
inverseT Chmod  = Chmod
inverseT Chown  = Chown
inverseT Create = Delete

------------------------------------------------------------
-- Theorem 1: self-inverse kinds
------------------------------------------------------------

public export
modifyInvolutive : inverseT (inverseT Modify) = Modify
modifyInvolutive = Refl

public export
moveInvolutive : inverseT (inverseT Move) = Move
moveInvolutive = Refl

public export
chmodInvolutive : inverseT (inverseT Chmod) = Chmod
chmodInvolutive = Refl

public export
chownInvolutive : inverseT (inverseT Chown) = Chown
chownInvolutive = Refl

------------------------------------------------------------
-- Theorem 2: Delete ↔ Create are mutual inverses
------------------------------------------------------------

public export
deleteCreateMutualA : inverseT (inverseT Delete) = Delete
deleteCreateMutualA = Refl

public export
deleteCreateMutualB : inverseT (inverseT Create) = Create
deleteCreateMutualB = Refl

------------------------------------------------------------
-- Theorem 3 (negative): Copy is *not* involutive
------------------------------------------------------------

||| `Copy`'s inverse is `Delete`; applying `inverseT` twice lands on
||| `Create`, *not* on `Copy`. We record the disagreement explicitly:
||| this is the framework's faithful answer, not a failure.
public export
copyNotInvolutive : inverseT (inverseT Copy) = Create
copyNotInvolutive = Refl

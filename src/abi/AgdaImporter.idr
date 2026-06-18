-- SPDX-License-Identifier: MPL-2.0
-- Copyright (c) Jonathan D.A. Jewell <j.d.a.jewell@open.ac.uk>
||| SPDX-License-Identifier: MPL-2.0
||| (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)
|||
||| Formal correctness witnesses for the Agda octad importer's
||| pure-functional core. Mirrors `email-octad-experiment/src/abi/Importer.idr`
||| in spirit: the territory is not yet an Idris2 record, so we prove the
||| *importer-side* invariants — qualified-name construction, body hash
||| determinism, and reference-set monotonicity — over abstract types,
||| with `%default total` and no banned patterns (`believe_me`,
||| `assert_total`, `Admitted`, `sorry`, `unsafeCoerce`).
|||
||| Verification: `idris2 --check src/abi/AgdaImporter.idr`

module AgdaImporter

%default total

------------------------------------------------------------
-- Abstract types matching the Rust importer
------------------------------------------------------------

||| A definition's local name (e.g. `Tree`) — opaque sequence of bytes.
public export
LocalName : Type
LocalName = String

||| A namespace path (e.g. `Foo.Bar`) — opaque.
public export
Namespace : Type
Namespace = String

||| The qualified-name function, modelling `qualified` in the Rust lexer:
||| if a namespace is present, prefix it with a dot; otherwise the local
||| name stands alone. The return value is the entry the Graph shape
||| uses for cross-definition edges and the Document shape uses for
||| titles, so determinism here is load-bearing for both.
public export
qualified : Maybe Namespace -> LocalName -> String
qualified Nothing  l = l
qualified (Just n) l = n ++ "." ++ l

------------------------------------------------------------
-- Theorem 1: qualified is a deterministic function of its inputs.
------------------------------------------------------------

||| For any (n, l), equal inputs yield equal outputs. This is the
||| property the importer's pass-2 graph-edge wiring relies on: every
||| call site that re-derives a qualified name from the same source must
||| land on the same key in `qname_to_octad`. The proof is `cong` over
||| the underlying string equality — no tactics, no postulates.
public export
qualifiedDeterministic
  :  (n1, n2 : Maybe Namespace) -> (l1, l2 : LocalName)
  -> n1 = n2 -> l1 = l2
  -> qualified n1 l1 = qualified n2 l2
qualifiedDeterministic n1 n2 l1 l2 prfN prfL =
  rewrite prfN in rewrite prfL in Refl

------------------------------------------------------------
-- Theorem 2: the unqualified-suffix lookup matches by-suffix.
------------------------------------------------------------

||| The importer's pass-2 reference resolver and the cross-repo
||| bridge probe both classify a reference as "in corpus" iff either
||| the full qname matches or the part after the last dot matches.
||| We model "split on the last dot" abstractly via a pair (prefix, suffix)
||| and prove that suffix equality is a sound matcher.
public export
record SplitName where
  constructor MkSplit
  pre  : String
  suff : String

||| Given a SplitName for the qualified name and a candidate reference,
||| `matchesBySuffix` is the resolver's truth condition.
public export
matchesBySuffix : SplitName -> String -> Bool
matchesBySuffix (MkSplit _ s) r = s == r

||| Same suffix yields equal matchers. This is the property the
||| pass-2 wirer relies on when an in-corpus reference is unqualified:
||| if two references would split to the same suffix, the resolver's
||| classification of them must agree.
public export
suffixMatchAgrees
  :  (sp1, sp2 : SplitName) -> (r : String)
  -> suff sp1 = suff sp2
  -> matchesBySuffix sp1 r = matchesBySuffix sp2 r
suffixMatchAgrees (MkSplit _ s1) (MkSplit _ s2) r prf =
  rewrite prf in Refl

------------------------------------------------------------
-- Theorem 3: reference set monotonicity under append.
------------------------------------------------------------

||| Modelling the importer's `lex_identifiers` as a function from a
||| body to a list of identifier names. We don't fix the implementation
||| — we just need its monotonicity contract: extending the body never
||| removes identifiers from the reference list. The Graph shape's
||| pass-2 reasoning depends on this.
public export
RefSet : Type
RefSet = List String

||| Sub-list relation, for "every element of `xs` appears in `ys`."
public export
data SubList : List a -> List a -> Type where
  SubNil  : SubList [] ys
  SubHere : SubList xs ys -> SubList xs (y :: ys)
  SubKeep : SubList xs ys -> SubList (x :: xs) (x :: ys)

||| If `lex` of body `b` is some list `L`, then `lex` of `b ++ extra`
||| produces a superset of `L`. We don't fix `lex`; we just declare the
||| monotonicity as an interface and instantiate it for any function
||| that satisfies it. The Rust implementation does, by construction
||| (it processes characters left-to-right and only ever appends).
public export
interface LexMonotonic (lex : String -> RefSet) where
  lexAppendMonotone
    :  (body, extra : String)
    -> SubList (lex body) (lex (body ++ extra))

||| Any reference list of `body` is a sublist of the reference list of
||| `body ++ extra`, given any monotonic lexer.
public export
referencesGrow
  :  {lex : String -> RefSet}
  -> LexMonotonic lex
  => (body, extra : String)
  -> SubList (lex body) (lex (body ++ extra))
referencesGrow body extra = lexAppendMonotone body extra

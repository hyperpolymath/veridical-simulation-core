// SPDX-License-Identifier: PMPL-1.0-or-later
// (MPL-2.0 is automatic legal fallback until PMPL is formally recognised)
//! Lexical Agda parser: extracts top-level definitions from `.agda` files
//! without invoking the Agda type-checker.
//!
//! The veridical-simulation framework treats each Agda definition as one
//! octad-entity. We need every definition's name, kind, source bytes,
//! comment block, file path, line range, and the names it references —
//! enough to populate Document / Semantic / Graph / Vector / Tensor /
//! Provenance / Temporal shapes. We deliberately do NOT type-check: a
//! lexical pass is total over arbitrary input, and gives us forward
//! progress even when a target repo's stdlib version is wrong.
//!
//! The parser is deliberately conservative — when it can't classify a
//! line, it skips to the next blank line. Coverage is one of the
//! veridicality probes that lifts the parser's quality back to the user.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// One Agda top-level definition extracted by the lexer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Definition {
    /// Fully-qualified name, e.g. `Echo.Eta-cancel` or `CNO.empty-is-cno`.
    /// Built from the file's `module X.Y where` line plus the local name.
    pub qualified_name: String,
    /// Local (unqualified) name, as it appears in the source.
    pub local_name: String,
    /// Kind tag for the Semantic shape — see [`DefKind`].
    pub kind: DefKind,
    /// Absolute path of the source file.
    pub source_path: PathBuf,
    /// 1-based inclusive line range covered by this definition.
    pub line_start: usize,
    pub line_end: usize,
    /// Verbatim source bytes of the definition (no normalisation).
    pub body: String,
    /// Verbatim comment block immediately above the definition, if any
    /// (`--` line comments and `{- … -}` block comments coalesced).
    pub leading_doc: Option<String>,
    /// Names referenced inside `body` — lexical only, no scope resolution.
    /// Useful for the Graph shape; expect false positives (the lexer
    /// treats everything that looks like an identifier as a reference).
    pub references: Vec<String>,
    /// SHA-256 of `body` — used for provenance witness.
    pub body_sha256_hex: String,
}

/// Kinds of Agda top-level definitions the lexer recognises. Anything
/// else is reported as `Unknown` and surfaced as a parser-coverage
/// finding rather than silently dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DefKind {
    Module,
    Data,
    Record,
    Function,
    Postulate,
    Open,
    Import,
    Unknown,
}

impl DefKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Module => "module",
            Self::Data => "data",
            Self::Record => "record",
            Self::Function => "function",
            Self::Postulate => "postulate",
            Self::Open => "open",
            Self::Import => "import",
            Self::Unknown => "unknown",
        }
    }
}

/// One parsed `.agda` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedFile {
    pub path: PathBuf,
    /// The `module X.Y.Z where` declaration's namespace, if found.
    pub module_namespace: Option<String>,
    /// All `import X` and `open import X` directives, ordered as they
    /// appear in the file.
    pub imports: Vec<String>,
    /// Definitions in source order.
    pub definitions: Vec<Definition>,
    /// Lines the parser couldn't classify — coverage signal.
    pub unclassified_line_ranges: Vec<(usize, usize)>,
    /// File modification time as RFC3339 UTC, surfaced into the Temporal
    /// shape. We use mtime rather than git-blame because the parser must
    /// run against checkouts that may not have git history.
    pub mtime_rfc3339: Option<String>,
}

/// Walk a directory and parse every `.agda` file. Returns parsed files
/// in walk order.
pub fn parse_directory(root: &Path) -> Vec<ParsedFile> {
    let mut out = Vec::new();
    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file()
            && entry.path().extension().and_then(|x| x.to_str()) == Some("agda")
        {
            match parse_file(entry.path()) {
                Ok(parsed) => out.push(parsed),
                Err(_) => continue,
            }
        }
    }
    out
}

/// Parse a single `.agda` file.
pub fn parse_file(path: &Path) -> anyhow::Result<ParsedFile> {
    let raw = std::fs::read_to_string(path)?;
    let mtime_rfc3339 = std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .map(|t| {
            let dt: chrono::DateTime<chrono::Utc> = t.into();
            dt.to_rfc3339()
        });
    let lines: Vec<&str> = raw.lines().collect();
    let mut imports = Vec::new();
    let mut module_namespace = None;
    let mut definitions = Vec::new();
    let mut unclassified = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();

        // File-level module declaration — captures the namespace and
        // is NOT itself emitted as a definition (it's metadata, not a
        // proof artefact). Nested `module X where` blocks below this
        // line still get extracted via the block-kind branch.
        if module_namespace.is_none() {
            if let Some(rest) = strip_kw(trimmed, "module ") {
                if let Some(name) = take_qualified_ident(rest) {
                    module_namespace = Some(name);
                    i += 1;
                    continue;
                }
            }
        }

        // Imports in any position.
        if let Some(rest) = strip_kw(trimmed, "import ") {
            if let Some(name) = take_qualified_ident(rest) {
                imports.push(name);
                i += 1;
                continue;
            }
        }
        if let Some(rest) = strip_kw(trimmed, "open import ") {
            if let Some(name) = take_qualified_ident(rest) {
                imports.push(name);
                i += 1;
                continue;
            }
        }

        // Block-introducing keywords. Each consumes a contiguous block
        // (everything indented under it, plus the introducing line).
        if let Some(kind) = block_kind(trimmed) {
            let leading_doc = collect_leading_doc(&lines, i);
            let start = i;
            // Local name is the first identifier after the kind keyword,
            // unless the keyword is `module`/`postulate` which we name by
            // their first declared symbol.
            let local_name = extract_block_name(trimmed, kind);
            let end = scan_block_end(&lines, i);
            let body: String = lines[start..=end].join("\n");
            let qualified_name = qualified(&module_namespace, &local_name);
            let references = lex_identifiers(&body);
            let body_sha256_hex = hex::encode(sha256(&body));
            definitions.push(Definition {
                qualified_name,
                local_name,
                kind,
                source_path: path.to_path_buf(),
                line_start: start + 1,
                line_end: end + 1,
                body,
                leading_doc,
                references,
                body_sha256_hex,
            });
            i = end + 1;
            continue;
        }

        // Function definition: `<lower-ident> :` at column 0 introduces
        // a top-level signature. Body extends through indented lines
        // AND through column-0 lines whose first token equals the
        // function name (the equation continuations).
        if let Some(name) = function_signature(trimmed, line) {
            let leading_doc = collect_leading_doc(&lines, i);
            let start = i;
            let end = scan_function_block_end(&lines, i, &name);
            let body: String = lines[start..=end].join("\n");
            let qualified_name = qualified(&module_namespace, &name);
            let references = lex_identifiers(&body);
            let body_sha256_hex = hex::encode(sha256(&body));
            definitions.push(Definition {
                qualified_name,
                local_name: name,
                kind: DefKind::Function,
                source_path: path.to_path_buf(),
                line_start: start + 1,
                line_end: end + 1,
                body,
                leading_doc,
                references,
                body_sha256_hex,
            });
            i = end + 1;
            continue;
        }

        // Skip blank lines, pragmas, and pure comments without flagging.
        if trimmed.is_empty()
            || trimmed.starts_with("--")
            || trimmed.starts_with("{-#")
            || trimmed.starts_with("{-")
            || trimmed.starts_with("infix")
            || trimmed.starts_with("infixl")
            || trimmed.starts_with("infixr")
            || trimmed.starts_with("syntax")
            || trimmed.starts_with("private")
            || trimmed.starts_with("variable")
            || trimmed.starts_with("where")
            || trimmed.starts_with("renaming")
            || trimmed.starts_with("hiding")
            || trimmed.starts_with("public")
            || trimmed.starts_with("abstract")
            || trimmed.starts_with("instance")
            || trimmed.starts_with("primitive")
        {
            i += 1;
            continue;
        }

        // Anything else at the top level is unclassified — record it
        // honestly rather than silently dropping. Coalesce into a range
        // until we hit a recognised form.
        let start = i;
        while i < lines.len() {
            let t = lines[i].trim_start();
            if t.is_empty() || block_kind(t).is_some() || function_signature(t, lines[i]).is_some()
            {
                break;
            }
            i += 1;
        }
        unclassified.push((start + 1, i));
    }

    Ok(ParsedFile {
        path: path.to_path_buf(),
        module_namespace,
        imports,
        definitions,
        unclassified_line_ranges: unclassified,
        mtime_rfc3339,
    })
}

fn strip_kw<'a>(s: &'a str, kw: &str) -> Option<&'a str> {
    s.strip_prefix(kw)
}

fn take_qualified_ident(s: &str) -> Option<String> {
    let s = s.trim_start();
    let end = s
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '-' || c == '\''))
        .unwrap_or(s.len());
    if end == 0 {
        None
    } else {
        Some(s[..end].to_string())
    }
}

fn block_kind(trimmed: &str) -> Option<DefKind> {
    if let Some(_) = trimmed.strip_prefix("data ") {
        Some(DefKind::Data)
    } else if let Some(_) = trimmed.strip_prefix("record ") {
        Some(DefKind::Record)
    } else if trimmed.starts_with("postulate") {
        Some(DefKind::Postulate)
    } else if let Some(_) = trimmed.strip_prefix("module ") {
        // Nested modules — only register if they introduce a `where`
        // block, otherwise it's the top file declaration handled above.
        if trimmed.contains("where") {
            Some(DefKind::Module)
        } else {
            None
        }
    } else {
        None
    }
}

fn extract_block_name(trimmed: &str, kind: DefKind) -> String {
    match kind {
        DefKind::Data => trimmed
            .strip_prefix("data ")
            .and_then(|r| take_qualified_ident(r))
            .unwrap_or_else(|| "<anon-data>".into()),
        DefKind::Record => trimmed
            .strip_prefix("record ")
            .and_then(|r| take_qualified_ident(r))
            .unwrap_or_else(|| "<anon-record>".into()),
        DefKind::Postulate => "<postulate-block>".into(),
        DefKind::Module => trimmed
            .strip_prefix("module ")
            .and_then(|r| take_qualified_ident(r))
            .unwrap_or_else(|| "<anon-module>".into()),
        _ => "<unknown>".into(),
    }
}

/// A function signature is a column-0 lower-case identifier followed by
/// a `:` (with optional whitespace before). We're conservative — we
/// only accept the signature line, not the equation lines (those land
/// inside `body` via the block scanner).
fn function_signature(trimmed: &str, full: &str) -> Option<String> {
    if full.starts_with(' ') || full.starts_with('\t') {
        return None;
    }
    let first = trimmed.chars().next()?;
    if !(first.is_ascii_lowercase() || first == '_') {
        return None;
    }
    let ident_end = trimmed
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-' || c == '\''))?;
    let ident = &trimmed[..ident_end];
    let rest = trimmed[ident_end..].trim_start();
    if rest.starts_with(':') {
        Some(ident.to_string())
    } else {
        None
    }
}

/// Scan forward from `start` until the block ends. Block ends at the
/// first column-0 form that isn't part of the same definition (a blank
/// line followed by a column-0 form, OR a column-0 form on a new line).
fn scan_block_end(lines: &[&str], start: usize) -> usize {
    let mut last_nonblank = start;
    let mut i = start + 1;
    while i < lines.len() {
        let line = lines[i];
        if line.is_empty() || line.trim().is_empty() {
            i += 1;
            continue;
        }
        // Indented continuation belongs to the block.
        if line.starts_with(' ') || line.starts_with('\t') {
            last_nonblank = i;
            i += 1;
            continue;
        }
        // Column-0 line that's a continuation pattern (equation) belongs
        // to the block; otherwise the block ends at the previous
        // non-blank line.
        let trimmed = line.trim_start();
        if is_equation_continuation(trimmed) {
            last_nonblank = i;
            i += 1;
            continue;
        }
        break;
    }
    last_nonblank
}

fn is_equation_continuation(_trimmed: &str) -> bool {
    // Generic block scanner: column-0 forms always end the block. The
    // function-specific scanner below handles equation continuations.
    false
}

/// Scan a function block forward from `start`. A column-0 line whose
/// first token equals `fn_name` is treated as an equation continuation
/// (the canonical Agda pattern: `name <args> = body`). Indented lines
/// are always continuations.
fn scan_function_block_end(lines: &[&str], start: usize, fn_name: &str) -> usize {
    let mut last_nonblank = start;
    let mut i = start + 1;
    while i < lines.len() {
        let line = lines[i];
        if line.trim().is_empty() {
            i += 1;
            continue;
        }
        if line.starts_with(' ') || line.starts_with('\t') {
            last_nonblank = i;
            i += 1;
            continue;
        }
        // Column-0 line — accept iff its first identifier matches the
        // function name (an equation for the same definition).
        let trimmed = line.trim_start();
        if first_identifier(trimmed) == Some(fn_name) {
            last_nonblank = i;
            i += 1;
            continue;
        }
        break;
    }
    last_nonblank
}

fn first_identifier(s: &str) -> Option<&str> {
    let end = s
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-' || c == '\''))
        .unwrap_or(s.len());
    if end == 0 { None } else { Some(&s[..end]) }
}

fn collect_leading_doc(lines: &[&str], at: usize) -> Option<String> {
    if at == 0 {
        return None;
    }
    let mut i = at as isize - 1;
    let mut collected: Vec<&str> = Vec::new();
    while i >= 0 {
        let line = lines[i as usize];
        let trimmed = line.trim_start();
        if trimmed.starts_with("--") {
            collected.push(line);
            i -= 1;
        } else if trimmed.is_empty() {
            // Blank line — keep scanning, doc blocks may include blank
            // separator lines.
            if collected.is_empty() {
                i -= 1;
                continue;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if collected.is_empty() {
        None
    } else {
        collected.reverse();
        Some(collected.join("\n"))
    }
}

/// Lex-only identifier extraction. Returns unique identifiers in body
/// order. Used for the Graph shape's reference edges.
pub fn lex_identifiers(body: &str) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        // Skip line comments.
        if c == '-' && chars.peek() == Some(&'-') {
            chars.next();
            for cc in chars.by_ref() {
                if cc == '\n' {
                    break;
                }
            }
            continue;
        }
        // Skip block comments (non-nesting — sufficient for the lexer).
        if c == '{' && chars.peek() == Some(&'-') {
            chars.next();
            let mut prev = '\0';
            for cc in chars.by_ref() {
                if prev == '-' && cc == '}' {
                    break;
                }
                prev = cc;
            }
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            let mut s = String::new();
            s.push(c);
            while let Some(&cc) = chars.peek() {
                if cc.is_alphanumeric() || cc == '_' || cc == '\'' || cc == '-' || cc == '.' {
                    s.push(cc);
                    chars.next();
                } else {
                    break;
                }
            }
            // Ignore tiny noise tokens.
            if s.len() >= 2 && seen.insert(s.clone()) {
                out.push(s);
            }
        }
    }
    out
}

fn qualified(ns: &Option<String>, local: &str) -> String {
    match ns {
        Some(n) => format!("{}.{}", n, local),
        None => local.to_string(),
    }
}

fn sha256(s: &str) -> Vec<u8> {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(s.as_bytes());
    h.finalize().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_str(content: &str) -> ParsedFile {
        let dir = tempdir();
        let path = dir.join("Test.agda");
        std::fs::write(&path, content).expect("write fixture");
        parse_file(&path).expect("parse fixture")
    }

    fn tempdir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "agda-lexparse-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock monotonic")
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).expect("mkdir tempdir");
        p
    }

    #[test]
    fn parses_module_and_data() {
        let p = parse_str(
            "module Foo.Bar where\n\
             open import Data.Nat\n\
             \n\
             data Tree (A : Set) : Set where\n\
             \x20\x20leaf : Tree A\n\
             \x20\x20node : A -> Tree A -> Tree A -> Tree A\n",
        );
        assert_eq!(p.module_namespace.as_deref(), Some("Foo.Bar"));
        assert_eq!(p.imports, vec!["Data.Nat".to_string()]);
        let data = p
            .definitions
            .iter()
            .find(|d| d.kind == DefKind::Data)
            .expect("data definition extracted");
        assert_eq!(data.local_name, "Tree");
        assert_eq!(data.qualified_name, "Foo.Bar.Tree");
        assert!(data.body.contains("leaf"));
        assert!(data.body.contains("node"));
        assert!(data.references.contains(&"leaf".to_string()));
        assert_eq!(data.line_start, 4);
    }

    #[test]
    fn parses_record_and_function() {
        let p = parse_str(
            "module M where\n\
             record Point : Set where\n\
             \x20\x20field x : Nat\n\
             \x20\x20      y : Nat\n\
             \n\
             origin : Point\n\
             origin = record { x = 0 ; y = 0 }\n",
        );
        let rec = p
            .definitions
            .iter()
            .find(|d| d.kind == DefKind::Record)
            .expect("record extracted");
        assert_eq!(rec.local_name, "Point");
        let fun = p
            .definitions
            .iter()
            .find(|d| d.kind == DefKind::Function)
            .expect("function extracted");
        assert_eq!(fun.local_name, "origin");
        assert!(fun.body.contains("origin = record"));
    }

    #[test]
    fn extracts_leading_doc() {
        let p = parse_str(
            "module M where\n\
             -- The empty tree.\n\
             -- One leaf, no children.\n\
             data Tree : Set where\n\
             \x20\x20leaf : Tree\n",
        );
        let data = &p.definitions[0];
        let doc = data.leading_doc.as_deref().expect("doc captured");
        assert!(doc.contains("empty tree"));
        assert!(doc.contains("One leaf"));
    }

    #[test]
    fn body_sha256_is_deterministic() {
        let p1 = parse_str(
            "module M where\n\
             data X : Set where\n\
             \x20\x20c : X\n",
        );
        let p2 = parse_str(
            "module M where\n\
             data X : Set where\n\
             \x20\x20c : X\n",
        );
        assert_eq!(
            p1.definitions[0].body_sha256_hex, p2.definitions[0].body_sha256_hex,
            "byte-identical input must yield byte-identical hash"
        );
    }
}

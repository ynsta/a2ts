# ADR 0008: Reviewed Glossary and Validated Turn Reconstruction

- Status: Accepted
- Date: 2026-10-03

## Context

Acoustic attribution can alternate speakers inside a sentence. Preserving every header prevents refinement from repairing these boundaries. Sending entire context exports is expensive and duplicates content. Context is proofread and supplies trusted spelling for either RPG sessions or work meetings.

## Decision

Extract a deterministic frequency-ranked dictionary using offline language membership filtering, explicit names and aliases, and export deduplication. Supply bounded relevant JSON entries to refinement. Keep source/settings provenance in strict validated snapshots.

Default CLI refinement uses structured speaker/text records tied to indexed word spans. Require complete ordered coverage, existing speaker labels and bounded text changes, including small transcription-word restorations and dictionary-supported names. Invalid chunks retain raw text. Use word-bounded larger requests and optional conservative timestamps. Preserve raw snapshots and document metadata. Offer legacy polish mode with immutable headers; preserve direct Python API compatibility. Disable RPG instructions with the existing general-context flag.

## Consequences

Text continuity can repair obvious fragments but cannot prove acoustic identity. Review inferred assignments against raw audio. Larger blocks reduce calls without guaranteeing lower total latency. Dictionaries are language-specific; unsupported languages fail clearly. No model may edit the original raw snapshot directly.

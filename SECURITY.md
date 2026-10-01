# Security Policy & Biometric Data Privacy

## 1. Reporting Security Vulnerabilities

We take security vulnerabilities seriously. If you discover a security vulnerability in `a2ts`, please report it responsibly rather than opening a public issue.

- **Private Security Contact**: Please contact the maintainer directly via GitHub Security Advisories or email [stany.marcel@gmail.com](mailto:stany.marcel@gmail.com) with the subject `[SECURITY] a2ts vulnerability report`.
- **Response Time**: We endeavor to acknowledge reports within 48 hours and provide a remediation timeline.
- **Scope**: Vulnerabilities related to command injection, path traversal, malicious input parsing, or unintended data exfiltration.

---

## 2. Voice Biometrics & Privacy Notice (GDPR Compliance)

`a2ts` uses acoustic neural networks (such as SpeechBrain ECAPA-TDNN) to extract 192-dimensional numerical speaker embeddings, which are stored in `contexte/voice_profiles.json`.

### Biometric Data Classification

Under data privacy regulations including the European Union **General Data Protection Regulation (GDPR)** (Article 9):
- Acoustic speaker embeddings constitute **biometric data** because they capture unique physical and acoustic characteristics that can uniquely identify an individual.
- When generating, storing, or matching speaker voice profiles, you are processing special category biometric data.

### Privacy Guidelines & Responsibilities

1. **Explicit Participant Consent**:
   - Ensure that all recorded participants have given informed consent for their voices to be recorded, analyzed, and enrolled into acoustic biometric profiles.
2. **Confidentiality & Storage**:
   - Never commit `contexte/voice_profiles.json` or session directories (`.a2ts/sessions/`) to public repositories.
   - Keep voice profile files on encrypted local storage with restricted access permissions.
3. **Right to Erasure (Deletion)**:
   - Participants have the right to request deletion of their biometric profiles at any time.
   - To remove all enrolled voice profiles:
     ```bash
     rm -f contexte/voice_profiles.json
     ```
   - To remove a specific participant's profile, edit `contexte/voice_profiles.json` and remove the corresponding key under `"speakers"`, or remove their voice sample from the database.
4. **Third-Party / Cloud Boundaries**:
   - Core transcription and acoustic diarization run 100% locally on your machine. Voice embeddings are never transmitted to external services.
   - If using the optional `--refine` pass with `agy`, transcript text (not audio or voice embeddings) may be sent to an LLM provider. Do not include sensitive personal identifying information in prompts if using cloud-hosted LLM endpoints.

---

## 3. Subprocess & Command Execution Security

`a2ts` implements strict defensive measures against command injection and path traversal:
- All external tool executions (`ffmpeg`, `ffprobe`, `agy`) strictly avoid `shell=True` and invoke argument vectors (`list[str]`) directly.
- File paths supplied via CLI or metadata are resolved and validated to prevent directory traversal outside designated working or cache directories.
- User-supplied metadata (YAML frontmatter, Craig `speakers.md`, `info.txt`) is parsed using safe parsers (`yaml.safe_load`).

---

## 4. Test Fixtures and Synthetic Roster Names

All character names, speaker tags, and Discord usernames appearing across automated tests, documentation, and fixtures are synthetic tabletop roleplaying game (TTRPG) campaign lore and test fixtures.

- Test fixtures use invented character names and invented Discord handles. Pattern scans of the git history found no credentials; this is not a guarantee.
- No Personally Identifiable Information (PII) or real-world personal data is intended to be contained in this repository.
- Audio test fixtures and synthetic speaker labels exist solely for regression testing, performance benchmarking, and development validation.


# Learning log

One section per episode: the concept, the naive version, what we measured, the fix, and what *you* wrote.
Design: [docs/superpowers/specs/2026-10-11-humanlike-receptionist-design.md](docs/superpowers/specs/2026-10-11-humanlike-receptionist-design.md)

## E0 — Setup

**Concept: the cascade pipeline.** A voice agent here is three models in a chain:
`mic → VAD → STT (Deepgram Nova-3) → turn detector → LLM (Gemma 4 31B) → TTS (Inworld TTS-2) → speaker`.
Each stage adds latency *and* errors, and errors compound: a misheard "thirteenth" becomes a wrong
booking no matter how good the LLM is. That's why we evaluate the **pipeline**, not each model's
leaderboard score.

**Things we found reading the installed source (livekit-agents 1.8.6), not the docs:**
- All three models are served by **LiveKit Inference**, so one LiveKit Cloud account is enough.
- Silero VAD is bundled and on by default (`agent_session.py`: `inference.VAD(model="silero")`).
- Turn handling lives in `livekit/agents/voice/turn.py` as `TurnHandlingOptions`:
  - `endpointing`: `mode` fixed/dynamic, `min_delay` 0.5 s, `max_delay` 3.0 s
  - `interruption`: adaptive mode, `min_duration` 0.5 s, false-interruption resume after 2.0 s
  - `preemptive_generation`: **on by default** (LLM starts before the turn is confirmed)
- Many of the talk's "fixes" are already defaults. Our naive episodes will switch them **off** first,
  so you can hear what each one is for.
- `session.update_options(endpointing_opts=...)` changes endpointing live (next turn), but
  interruption options can only be set when the session starts. That matters for E1b (per-user profiles).

**Your turn:** get LiveKit credentials and hear the agent (see README steps).

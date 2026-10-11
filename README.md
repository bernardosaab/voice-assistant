# Harbor View Hotel — humanlike voice receptionist

A learning project: a hotel-receptionist voice agent on [LiveKit Agents](https://docs.livekit.io/agents/),
built in "break it → measure it → fix it" episodes focused on what callers actually *hear*
(time to first audible audio, not TTFB; time to first sentence, not TTFT).

Stack: Deepgram Nova-3 → Gemma 4 31B → Inworld TTS-2, all via LiveKit Inference.

## Setup
1. Python is managed by uv (3.12): `uv sync`
2. Install the LiveKit CLI: `winget install LiveKit.LiveKitCLI`
3. Create a free LiveKit Cloud project, then link it: `lk cloud auth`
4. Write credentials to `.env.local` (git-ignored): `lk app env -w` or copy `.env.example`
5. Talk to it: `uv run agent.py console`

See [LEARNING.md](LEARNING.md) for the episode log and
[the design spec](docs/superpowers/specs/2026-10-11-humanlike-receptionist-design.md).

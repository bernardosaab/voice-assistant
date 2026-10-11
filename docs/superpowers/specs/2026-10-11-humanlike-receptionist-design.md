# Humanlike Hotel Receptionist Voice Agent — Design & Plan

## Context
This is a new, empty project. The goal is a voice agent that *feels* human. It's inspired by Jesse Hall's
LiveKit talk: pipelines fail differently than individual models, perceived latency is not the same as
measured latency, turn detection should work from audio, and the agent should buy time during tool
calls the way a person would. The domain is a **hotel receptionist** that really does things: checks
availability, holds and books rooms, takes a (simulated) payment, looks up bookings, and cancels them
under a policy. Slow work runs **asynchronously** while the agent keeps listening and talking.

The user is learning voice agents, so the project is built as **"break it → measure it → fix it"
episodes**. Each episode starts with a deliberately naive version that *works* but shows a real,
noticeable problem: you hear it, then the numbers confirm it. Then we fix it and compare before and
after. Claude writes the scaffolding and plumbing and explains the concept. The user writes 1–2 core
"learning pieces" per episode (TODO + hints + a failing test), and Claude reviews them.

Each fix sits behind a config flag (e.g. `--naive-turns`, `--metric=ttft`), so any "before" state can be
replayed later for comparison or to show someone. The design points below describe the **end state**
the episodes build toward.

## Decisions made
| Decision | Choice | Why |
|---|---|---|
| Framework | LiveKit Agents (Python, v1.x) | Built-in VAD, audio turn detector, interruptions, tools, metrics; we tune the knobs and build the humanlike layer on top |
| Transport | `console` mode first → LiveKit Cloud + Agents Playground | No server needed to start; the browser path gives real network latency; SIP can come later |
| STT | Deepgram Nova-3 (streaming) | Fast streaming partials, good for names and numbers |
| LLM | Gemma 4 31B IT via LiveKit Inference | Runs close to the media server, one key. Availability is verified at M0; fallback is a direct OpenAI-compatible host |
| TTS | Inworld TTS-2 (streaming) | User's pick. We judge it by *first audible audio*, not TTFB |
| Backend | SQLite + simulated delays (payment 3–8 s, random declines) | Delays are deliberate so we can practise buying time and async work |
| Python | 3.12 via `uv` (system is 3.9, too old) | `uv` is already installed |

All three models sit behind the plugin interfaces, so swapping one is a one-line config change and the
benchmark (M7) decides by evidence, not leaderboards. Swap candidates to benchmark later: Cartesia
Sonic / ElevenLabs Flash (TTS), Gemini Flash-class model (LLM baseline for tool accuracy).

## Architecture
```
mic ─► LiveKit room ─► VAD (Silero) ─► STT (Nova-3) ─► Turn detector (audio/text model)
                                                          │ end of turn
                                                          ▼
                          ┌──────────── Agent (LLM: Gemma) ◄─── chat context
                          │   tools ──► hotel/ (SQLite, policy, payments) ──► background tasks
                          ▼                                                       │ result
                TTS (Inworld) ─► speaker                     inject into context ◄┘ + generate_reply
                    ▲
        metrics/ tap: true end-of-speech, first sentence, first NON-SILENT audio frame
```

### Project layout
```
voice_assistant/
  pyproject.toml            # uv-managed, Python 3.12
  .env.example              # LIVEKIT_URL/API_KEY/SECRET, DEEPGRAM, INWORLD keys
  agent.py                  # entrypoint: AgentSession config + `console`/`dev` modes
  receptionist/
    agent.py                # Receptionist(Agent): instructions, tools, node overrides
    prompts.py              # voice-style system prompt (short turns, read-backs, no markdown)
    tools.py                # @function_tool wrappers → hotel/ service calls
    background.py           # async job runner (payment etc.) + result injection
    speech.py               # TTS text normalisation (dates, prices, confirmation codes)
  hotel/
    db.py, schema.sql       # rooms, room_types, reservations, holds, payments
    availability.py         # search + pricing
    booking.py              # hold → confirm (idempotent), modify, lookup
    policy.py               # cancellation policy engine (pure functions)
    payments.py             # fake processor: latency, declines, idempotency keys
  metrics/
    perceived.py            # per-turn timeline: true EoS → first sentence → first audible audio
    report.py               # JSONL → p50/p90 latency budget table
  evals/
    scenarios/*.yaml        # scripted conversations (mind change, interruption, cancellation…)
    test_agent_behaviour.py # LiveKit text-mode agent tests (tool calls, judged replies)
  tests/                    # pytest for hotel/ and metrics/
  docs/superpowers/specs/2026-10-11-humanlike-receptionist-design.md
  LEARNING.md               # running log of concepts + "your turn" exercises
```

### Key design points
1. **Perceived latency (`metrics/perceived.py`).** Per turn we record:
   - `t_user_eos`: when the user *really* stopped talking. That's the VAD end-of-speech event minus its
     silence-duration wait, which the framework's numbers hide.
   - `t_first_token` vs `t_first_sentence`: we override `llm_node` and time when the first chunk that
     ends a sentence leaves the LLM. TTS can't start until then.
   - `t_tts_ttfb` vs `t_first_audible`: we override `tts_node` and find the first audio frame whose
     RMS energy is above a threshold. That exposes leading silence padding (the 200 ms TTFB / ~1 s real
     audio gap the user described).
   - Headline metric: **perceived response latency = t_first_audible − t_user_eos**. Logged to JSONL
     next to LiveKit's built-in metrics so the gap is visible.
2. **Latency budget (target).** Turn detection ≈ 200–400 ms, LLM first sentence ≈ 300–500 ms, TTS
   first audible ≈ 150–250 ms, network ≈ 50–100 ms. **Under 1.5 s is acceptable; aim for ~600–800 ms.**
   Levers: preemptive generation (start the LLM on the transcript before the turn is confirmed),
   endpointing delays, short first sentences in the prompt, region choice.
3. **Buying time.** Fast tools (<~700 ms) just run. Slow tools say a short, varied, context-aware filler
   *before* awaiting ("Let me check what we have for those dates…"), never the same phrase twice in a row.
4. **Async work while listening.** `start_payment` returns immediately ("processing started") and runs
   in a background task. The user can keep chatting (breakfast? parking?). When the task finishes, the
   result goes into the chat context and the agent calls `generate_reply` at the next natural gap. If
   the user is mid-utterance it waits; it never talks over them.
5. **Mind changes and interruptions.** Booking is a state machine: `hold` (expires in 10 min) →
   `confirmed` → `cancelled`. Changing dates or room type releases and replaces the hold, so there are
   no orphaned or duplicate bookings. Interruptions use LiveKit's barge-in, plus false-interruption
   recovery (resume when the "interruption" was a cough or "mm-hm").
6. **Speakable output (`speech.py`).** Dates come out as "Friday the 14th", prices as "two hundred
   and ten dollars", confirmation codes are spelled out in groups, and markdown is stripped. Bookings
   are always read back before confirming.
7. **Tools.** `check_availability`, `get_room_details`, `create_hold`, `update_hold`, `start_payment`
   (async), `confirm_booking`, `lookup_booking`, `cancel_booking` (policy-aware: quotes the fee and
   asks for consent first), `answer_hotel_faq` (parking, breakfast, check-in times from a small
   facts file), `transfer_to_human` (simulated handoff for out-of-scope or upset callers).
8. **Cancellation policy (`policy.py`).** Free until 48 h before check-in; 50% between 48 h and 24 h;
   100% after that or for no-shows; non-refundable rates are never refundable. Pure functions, fully
   unit-tested.
9. **Error handling.** Payment declines → offer another card or a hold. Tool exceptions → a spoken
   apology plus a retry or alternative, never silence. STT/TTS provider errors → LiveKit fallback
   adapters (added at M6 if time allows).

## Episodes: break it → measure it → fix it
Each episode: **(1) naive build → (2) symptom you hear → (3) measurement that proves it → (4) fix →
(5) before/after numbers** logged to `LEARNING.md`.

| # | Naive version (the "broken" state) | Symptom you'll notice | How we measure it | Fix | **Your turn** |
|---|---|---|---|---|---|
| E0 | — | — | — | Setup: `git init`, uv + Py 3.12, `.env.example`, spec into `docs/`, LiveKit Cloud keys, verify Gemma on LiveKit Inference, hello-world in `console` | Get keys, run it |
| E1 | **Silence-only endpointing**: VAD with a fixed short silence timeout, no turn detector | Cuts you off on "I'd like a room for… um… two nights". Raising the timeout makes it feel sluggish | Scripted utterances with mid-sentence pauses → % of turns cut off vs added delay | Audio/semantic turn detector + tuned `min/max_endpointing_delay` | Run the pause script on both modes; pick the endpointing values |
| E1b | **One endpointing config for everyone** (even well-tuned) | A slow, deliberate caller still gets cut off; a fast caller waits on every turn | Two synthetic personas (short vs long pause distributions) → cut-off % and added wait per persona | `UserTurnProfile` (SQLite, aggregates only: pause p50/p90, false/missed-interruption rates, observed_turns, `adaptation_enabled`) + rule-based `TurnPersonalisationService` with safety bounds. Cold start: LiveKit `dynamic` mode until ≥20 turns. Live: `session.update_options(endpointing_opts=…)` (next turn). Interruption settings only at session start (can't change mid-session in 1.8.6). User ID = caller number / participant identity / `"console-user"`. Opt-out + delete profile | Write the streaming profile update (EMA / quantile estimate) and `recommend_endpointing(profile)` |
| E2 | **Dashboard trusts vendor metrics**: TTS TTFB + LLM TTFT only | Dashboard says ~200 ms TTS, but the reply *feels* close to a second late | Perceived tap: true end of speech (VAD minus its silence wait) → first **audible** frame. The difference shows leading silence padding and the VAD wait | Report perceived latency as the headline metric; trim leading silence; per-stage budget table | Write `first_audible_offset(frames, threshold)` (RMS), test-first with synthetic audio |
| E3 | **TTFT as the LLM metric** + chatty prompt ("Thank you so much for calling the Grand Hotel, I'd be absolutely delighted to…") + LLM output buffered until complete | TTFT looks great (~250 ms) but speech starts late | Time to first *sentence* vs TTFT, per turn | Stream sentence-by-sentence into TTS, short first clause in the prompt, preemptive generation | Write the first-sentence detector (handles "Mr.", "$210.50", "10 a.m.") |
| E4 | **Blocking tools, no filler**: availability lookup and payment just `await` (1–8 s) | Dead air; you say "hello?…" and it's unclear whether it heard you | Silence gaps > 1.5 s per call; how often the user speaks into dead air | Context-aware, varied fillers before slow tools; latency threshold for when to use them | Write the filler picker (no repeats, matches the tool) |
| E5 | **Payment still blocks the conversation** (filler, then waiting) | You can't ask "is breakfast included?" while the card is processing | Turns blocked during tool execution | Background job runner; result injected into context and spoken at a natural gap, never over the user | Implement the "wait for a natural gap" rule |
| E6 | **`book_room` writes the reservation immediately**; payment has no idempotency key | "Actually make that Saturday" → **two** bookings; a retry after timeout → charged twice | Eval scenario checks the DB for duplicate reservations/charges | Hold → confirm state machine, `update_hold`, idempotency keys | Write `policy.cancellation_fee()` against provided tests + the mind-change eval |
| E7 | **Raw LLM text to TTS**, default interruption settings | Reads "$210.00" as "dollar two one zero…", says "asterisk asterisk", reads ISO dates; a cough or "mm-hm" stops it mid-sentence | Eval transcripts flag unspeakable tokens; count of false interruptions | `speech.py` normalisation, read-backs, false-interruption recovery, interruption tuning | Write two eval scenarios that break the agent, then fix them |
| E7b | **Trusts the transcript blindly** | "Tuesday the thirteenth" heard as "thirtieth" → books the wrong date; "fifteen" vs "fifty"; misspelled surname → booking lookup fails | Eval scenarios that inject known STT confusions (13/30, 15/50, names) → % of wrong bookings | Clarify only when it matters: low STT word confidence on dates/numbers/names, weekday-vs-date conflicts ("the 30th isn't a Tuesday, the 13th is"), contrastive questions ("one-three or three-zero?"), mandatory read-back before irreversible actions. No over-confirming of low-risk details | Write `find_ambiguities(transcript_words, calendar)` → list of things to confirm |
| E8 | **Leaderboard-picked models** | — (the question becomes "is this really the best pipeline?") | Run the scenario suite across model swaps (TTS: Inworld vs Cartesia/ElevenLabs Flash; LLM: Gemma vs a Flash-class baseline) | Choose by *our* perceived latency + tool accuracy, not leaderboards | Pick the swaps; interpret the matrix |

How the episodes connect: E1–E3 are about *timing* on a bare talking agent. The hotel backend (SQLite,
availability, payments sim) gets built at the start of E4 because the tools need it. E4–E6 are about
*doing work* like a person. E7 is about *sounding* like a person. E8 ties it together with evidence.

## Verification
- `uv run pytest`: hotel/, policy, speech normalisation, perceived-latency maths (synthetic audio
  with known leading silence must give the right TTFA).
- `uv run pytest evals/`: LiveKit text-mode agent tests check correct tool calls for booking, mind
  change, cancellation-with-fee-consent, and async payment completion.
- Per episode: run the episode's script/eval with the naive flag and then the fixed flag. The
  before/after numbers go into `LEARNING.md`, and the fix only counts once the numbers move.
- Manual: `uv run agent.py console`. Run the hotel demo script: book, interrupt mid-read-back, change
  dates, pay (keep chatting while it processes), cancel within 48 h. Then `uv run -m metrics.report`
  to see the p50/p90 table of perceived vs vendor-reported latency.
- M1+: connect through LiveKit Cloud Playground and compare console vs network latency.

## Open items to resolve at M0 (not blocking the design)
- Confirm the exact Gemma 4 31B model ID on LiveKit Inference (fallback: a direct OpenAI-compatible host).
- Confirm the Inworld plugin name/options in the current livekit-agents release.
- LiveKit's open-source agent benchmark mentioned in the talk: find it and see whether M7 can reuse it.

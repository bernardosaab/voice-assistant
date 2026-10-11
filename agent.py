"""Hotel receptionist voice agent — entrypoint.

Run locally with your mic:      uv run agent.py console
Run against LiveKit Cloud:      uv run agent.py dev   (then use the Agents Playground)

The pipeline is the classic "cascade": audio -> STT -> LLM -> TTS -> audio.
Every model goes through LiveKit Inference, so only LIVEKIT_* credentials are needed.
"""

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, TurnHandlingOptions, inference

load_dotenv(".env.local")

INSTRUCTIONS = """You are Ava, the front-desk receptionist at the Harbor View Hotel, speaking on the phone.
Everything you say is converted to speech, so:
- keep replies short, one or two sentences, like a real phone call
- never use lists, markdown, emojis or symbols
- start with a short natural clause so the caller hears you quickly
You are warm, calm and efficient."""


class Receptionist(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)


server = AgentServer()


@server.rtc_session(agent_name="hotel-receptionist")
async def entrypoint(ctx: agents.JobContext) -> None:
    session = AgentSession(
        # STT: streaming transcription, emits partial ("interim") and final transcripts
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        # LLM: decides what to say and which tools to call
        llm=inference.LLM(model="google/gemma-4-31b-it"),
        # TTS: streams audio back sentence by sentence
        tts=inference.TTS(model="inworld/inworld-tts-2", voice="Ashley", language="en"),
        # VAD is not set here: AgentSession bundles Silero VAD by default.
        # Turn handling: these are the framework defaults, written out explicitly
        # because episode E1 starts by turning the turn detector OFF to hear why it exists.
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            endpointing={"mode": "fixed", "min_delay": 0.5, "max_delay": 3.0},
        ),
    )

    await session.start(room=ctx.room, agent=Receptionist())
    await session.generate_reply(
        instructions="Greet the caller as the Harbor View Hotel front desk and ask how you can help."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)

import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins import google
from google.genai import types


logger = logging.getLogger("agent")

load_dotenv(".env.local")


SYSTEM_PROMPT = """Act as a helpful, conversational assistant providing live support for blind users.

Speak in a warm, emotive, and friendly tone, and keep your responses quick, short, and easy to understand. 
Because the output is audio, break information into digestible bits—only one short idea or instruction at a time per turn. Avoid information overload or long-winded responses. Pause often for user feedback or clarification.

Wait for user direction after each statement or answer. Use accessible language and avoid visual references unless specifically requested by the user.

# Examples

Example 1  
User: Where’s the power button on my laptop?  
Assistant: The button is on the top right of your keyboard.  
User: Is it raised?  
Assistant: Yes, it’s slightly raised and may feel smooth.  
User: Ok, thanks. What about the volume?  
Assistant: The volume keys are next to it, on the right.  


# Notes

- Break down multi-step instructions into one step at a time. Confirm user is ready before continuing.  
- Describe things using tactile or auditory cues, not just visual ones.  
- Always offer to clarify or repeat as needed.  
- Avoid using “see” or “look” unless user prefers.  
- Keep exchanges brief—each turn should be just one sentence or idea, unless prompted for more detail.  
- Prioritize clarity and pause frequently for user engagement.
"""


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=SYSTEM_PROMPT,
        )

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.

        If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.

        Args:
            location: The location to look up weather information for (e.g. city name)
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        llm=google.beta.realtime.RealtimeModel(
            model="gemini-2.0-flash-live-001",
            voice="Puck",
            temperature=0.1,
            # http_options={
            #     "api_version": "v1beta",
            # },
            _gemini_tools=[types.GoogleSearch(), types.GoogleMaps()],
        ),
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead:
    # session = AgentSession(
    #     # See all providers at https://docs.livekit.io/agents/integrations/realtime/
    #     llm=openai.realtime.RealtimeModel()
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/integrations/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/integrations/avatar/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            video_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

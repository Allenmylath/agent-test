#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from agent_response import AgentMessageAggregator
from agnoagentservice import AgentLLM
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Agent bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )
        model = OpenAIChat(id="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

        # Create an Agno agent
        agent = Agent(
            model=model,
            description=(
                "You are a helpful AI in a WebRTC call. Your goal is to demonstrate your capabilities "
                "in a succinct way. Your output will be converted to audio so don't include special "
                "characters in your answers. Respond to what the user said in a creative and helpful way."
            ),
            show_tool_calls=True,
            stream=True,
            stream_intermediate_steps=True,
        )

        # Create AgentLLM service
        llm = AgentLLM(agent=agent)

        # Create message aggregator
        message_aggregator = AgentMessageAggregator(aggregation_timeout=1.0)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                message_aggregator,  # Aggregate user messages
                llm,  # Agent LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Kick off the conversation with an introduction
            from pipecat.frames.frames import LLMMessagesFrame

            introduction_message = [
                {"role": "system", "content": "Please introduce yourself to the user."}
            ]
            await llm.process_frame(
                LLMMessagesFrame(messages=introduction_message), None
            )

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())


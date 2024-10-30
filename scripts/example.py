"""Example script to test the MusicAssistant client."""

from __future__ import annotations

import argparse
import asyncio
import logging

from music_assistant_client import MusicAssistantClient

# Get parsed passed in arguments.
parser = argparse.ArgumentParser(description="MusicAssistant Client Example.")
parser.add_argument(
    "url",
    type=str,
    help="URL of MASS server, e.g. http://localhost:8095",
)
parser.add_argument(
    "--log-level",
    type=str,
    default="info",
    help="Provide logging level. Example --log-level debug, default=info, "
    "possible=(critical, error, warning, info, debug)",
)

args = parser.parse_args()


if __name__ == "__main__":
    # configure logging
    logging.basicConfig(level=args.log_level.upper())

    async def run_mass() -> None:
        """Run the MusicAssistant client."""
        # run the client
        async with MusicAssistantClient(args.url, None) as client:
            # start listening
            await client.start_listening()

    # run the client
    asyncio.run(run_mass())

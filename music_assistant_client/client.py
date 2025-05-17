"""Music Assistant Client: Manage a Music Assistant server remotely."""

from __future__ import annotations

import asyncio
import logging
import urllib.parse
import uuid
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, Self

from music_assistant_models.api import (
    CommandMessage,
    ErrorResultMessage,
    EventMessage,
    ResultMessageBase,
    ServerInfoMessage,
    SuccessResultMessage,
    parse_message,
)
from music_assistant_models.enums import EventType, ImageType
from music_assistant_models.errors import ERROR_MAP
from music_assistant_models.event import MassEvent
from music_assistant_models.provider import ProviderInstance, ProviderManifest

from .config import Config
from .connection import WebsocketsConnection
from .constants import API_SCHEMA_VERSION
from .exceptions import ConnectionClosed, InvalidServerVersion, InvalidState
from .music import Music
from .player_queues import PlayerQueues
from .players import Players

if TYPE_CHECKING:
    from types import TracebackType

    from aiohttp import ClientSession
    from music_assistant_models.media_items import ItemMapping, MediaItemImage, MediaItemType
    from music_assistant_models.queue_item import QueueItem

EventCallBackType = Callable[[MassEvent], Coroutine[Any, Any, None] | None]
EventSubscriptionType = tuple[
    EventCallBackType, tuple[EventType, ...] | None, tuple[str, ...] | None
]


class MusicAssistantClient:
    """Manage a Music Assistant server remotely."""

    def __init__(self, server_url: str, aiohttp_session: ClientSession | None) -> None:
        """Initialize the Music Assistant client."""
        self.server_url = server_url
        self.connection = WebsocketsConnection(server_url, aiohttp_session)
        self.logger = logging.getLogger(__package__)
        self.logger.debug("Initializing MusicAssistantClient for server: %s", server_url)
        self._result_futures: dict[str | int, asyncio.Future[Any]] = {}
        self._subscribers: list[EventSubscriptionType] = []
        self._stop_called: bool = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._config = Config(self)
        self._players = Players(self)
        self._player_queues = PlayerQueues(self)
        self._music = Music(self)
        # below items are retrieved after connect
        self._server_info: ServerInfoMessage | None = None
        self._provider_manifests: dict[str, ProviderManifest] = {}
        self._providers: dict[str, ProviderInstance] = {}

    @property
    def server_info(self) -> ServerInfoMessage | None:
        """Return info of the server we're currently connected to."""
        return self._server_info

    @property
    def providers(self) -> list[ProviderInstance]:
        """Return all loaded/running Providers (instances)."""
        return list(self._providers.values())

    @property
    def provider_manifests(self) -> list[ProviderManifest]:
        """Return all Provider manifests."""
        return list(self._provider_manifests.values())

    @property
    def config(self) -> Config:
        """Return Config handler."""
        return self._config

    @property
    def players(self) -> Players:
        """Return Players handler."""
        return self._players

    @property
    def player_queues(self) -> PlayerQueues:
        """Return PlayerQueues handler."""
        return self._player_queues

    @property
    def music(self) -> Music:
        """Return Music handler."""
        return self._music

    def get_provider_manifest(self, domain: str) -> ProviderManifest:
        """Return Provider manifests of single provider(domain)."""
        return self._provider_manifests[domain]

    def get_provider(
        self, provider_instance_or_domain: str, return_unavailable: bool = False
    ) -> ProviderInstance | None:
        """Return provider by instance id or domain."""
        # lookup by instance_id first
        if prov := self._providers.get(provider_instance_or_domain):
            if return_unavailable or prov.available:
                return prov
            if not prov.is_streaming_provider:
                # no need to lookup other instances because this provider has unique data
                return None
            provider_instance_or_domain = prov.domain
        # fallback to match on domain
        # note that this can be tricky if the provider has multiple instances
        # and has unique data (e.g. filesystem)
        for prov in self._providers.values():
            if prov.domain != provider_instance_or_domain:
                continue
            if return_unavailable or prov.available:
                return prov
        self.logger.debug("Provider %s is not available", provider_instance_or_domain)
        return None

    def get_image_url(self, image: MediaItemImage, size: int = 0) -> str:
        """Get (proxied) URL for MediaItemImage."""
        assert self.server_info
        if image.remotely_accessible and not size:
            return image.path
        if image.remotely_accessible and size:
            # get url to resized image(thumb) from weserv service
            return (
                f"https://images.weserv.nl/?url={urllib.parse.quote(image.path)}"
                f"&w=${size}&h=${size}&fit=cover&a=attention"
            )
        # return imageproxy url for images that need to be resolved
        # the original path is double encoded
        encoded_url = urllib.parse.quote(urllib.parse.quote(image.path))
        return (
            f"{self.server_info.base_url}/imageproxy?path={encoded_url}"
            f"&provider={image.provider}&size={size}"
        )

    def get_media_item_image_url(
        self,
        item: MediaItemType | ItemMapping | QueueItem,
        type: ImageType = ImageType.THUMB,  # noqa: A002
        size: int = 0,
    ) -> str | None:
        """Get image URL for MediaItem, QueueItem or ItemMapping."""
        # handle queueitem with media_item attribute
        if (media_item := getattr(item, "media_item", None)) and (
            img := self.music.get_media_item_image(media_item, type)
        ):
            return self.get_image_url(img, size)
        if img := self.music.get_media_item_image(item, type):
            return self.get_image_url(img, size)
        return None

    def subscribe(
        self,
        cb_func: EventCallBackType,
        event_filter: EventType | tuple[EventType, ...] | None = None,
        id_filter: str | tuple[str, ...] | None = None,
    ) -> Callable[[], None]:
        """Add callback to event listeners.

        Returns function to remove the listener.
            :param cb_func: callback function or coroutine
            :param event_filter: Optionally only listen for these events
            :param id_filter: Optionally only listen for these id's (player_id, queue_id, uri)
        """
        if isinstance(event_filter, EventType):
            event_filter = (event_filter,)
        if isinstance(id_filter, str):
            id_filter = (id_filter,)
        listener = (cb_func, event_filter, id_filter)
        self._subscribers.append(listener)

        def remove_listener() -> None:
            self._subscribers.remove(listener)

        return remove_listener

    async def connect(self) -> None:
        """Connect to the remote Music Assistant Server."""
        self.logger.debug("Attempting to connect to server: %s", self.server_url)
        self._loop = asyncio.get_running_loop()
        if self.connection.connected:
            # already connected
            return
        # NOTE: connect will raise when connecting failed
        self.logger.debug("Calling self.connection.connect()")
        result = await self.connection.connect()
        self.logger.debug("self.connection.connect() returned")
        info = ServerInfoMessage.from_dict(result)

        # basic check for server schema version compatibility
        if info.min_supported_schema_version > API_SCHEMA_VERSION:
            # our schema version is too low and can't be handled by the server anymore.
            await self.connection.disconnect()
            msg = (
                f"Schema version is incompatible: {info.schema_version}, "
                f"the server requires at least {info.min_supported_schema_version} "
                " - update the Music Assistant client to a more "
                "recent version or downgrade the server."
            )
            raise InvalidServerVersion(msg)

        self._server_info = info

        self.logger.info(
            "Connected to Music Assistant Server %s, Version %s, Schema Version %s",
            info.server_id,
            info.server_version,
            info.schema_version,
        )

    async def send_command(
        self,
        command: str,
        require_schema: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Send a command and get a response."""
        self.logger.debug("Sending command: %s with args: %s", command, kwargs)
        if not self.connection.connected or not self._loop:
            msg = "Not connected"
            raise InvalidState(msg)

        if (
            require_schema is not None
            and self.server_info is not None
            and require_schema > self.server_info.schema_version
        ):
            msg = (
                "Command not available due to incompatible server version. Update the Music "
                f"Assistant Server to a version that supports at least api schema {require_schema}."
            )
            raise InvalidServerVersion(msg)

        command_message = CommandMessage(
            message_id=uuid.uuid4().hex,
            command=command,
            args=kwargs,
        )
        future: asyncio.Future[Any] = self._loop.create_future()
        self._result_futures[command_message.message_id] = future
        await self.connection.send_message(command_message.to_dict())
        try:
            return await future
        finally:
            self._result_futures.pop(command_message.message_id)

    async def send_command_no_wait(
        self,
        command: str,
        require_schema: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Send a command without waiting for the response."""
        self.logger.debug("Sending command (no wait): %s with args: %s", command, kwargs)
        if not self.server_info:
            msg = "Not connected"
            raise InvalidState(msg)

        if require_schema is not None and require_schema > self.server_info.schema_version:
            msg = (
                "Command not available due to incompatible server version. Update the Music "
                f"Assistant Server to a version that supports at least api schema {require_schema}."
            )
            raise InvalidServerVersion(msg)
        command_message = CommandMessage(
            message_id=uuid.uuid4().hex,
            command=command,
            args=kwargs,
        )
        await self.connection.send_message(command_message.to_dict())

    async def start_listening(self, init_ready: asyncio.Event | None = None) -> None:
        """Connect (if needed) and start listening to incoming messages from the server."""
        self.logger.debug("Starting listening task")
        self.logger.debug("Calling self.connect()")
        await self.connect()
        self.logger.debug("self.connect() returned")

        # fetch initial state
        # we do this in a separate task to not block reading messages
        async def fetch_initial_state() -> None:
            self.logger.debug("Fetching initial state")
            self.logger.debug("Fetching providers")
            self._providers = {
                x["instance_id"]: ProviderInstance.from_dict(x)
                for x in await self.send_command("providers")
            }
            self.logger.debug("Fetched %s providers", len(self._providers))
            self.logger.debug("Fetching provider manifests")
            self._provider_manifests = {
                x["domain"]: ProviderManifest.from_dict(x)
                for x in await self.send_command("providers/manifests")
            }
            self.logger.debug("Fetched %s provider manifests", len(self._provider_manifests))
            self.logger.debug("Fetching player queues state")
            await self._player_queues.fetch_state()
            self.logger.debug("Fetched player queues state")
            self.logger.debug("Fetching players state")
            await self._players.fetch_state()
            self.logger.debug("Fetched players state")

            if init_ready is not None:
                self.logger.debug("Setting init_ready event")
                init_ready.set()

        fetch_state_task = asyncio.create_task(fetch_initial_state())

        try:
            self.logger.debug("Starting message receiving loop")
            # keep reading incoming messages
            while not self._stop_called:
                msg = await self.connection.receive_message()
                self._handle_incoming_message(msg)
        except ConnectionClosed:
            self.logger.debug("Connection closed during listening")
        finally:
            fetch_state_task.cancel()
            await self.disconnect()

    async def disconnect(self) -> None:
        """Disconnect the client and cleanup."""
        self.logger.debug("Attempting to disconnect from server")
        self._stop_called = True
        # cancel all command-tasks awaiting a result
        for future in self._result_futures.values():
            future.cancel()
        self.logger.debug("Calling self.connection.disconnect()")
        await self.connection.disconnect()
        self.logger.debug("self.connection.disconnect() returned")

    def _handle_incoming_message(self, raw: dict[str, Any]) -> None:
        """
        Handle incoming message.

        Run all async tasks in a wrapper to log appropriately.
        """
        self.logger.debug("Handling incoming message: %s", raw)
        msg = parse_message(raw)
        # handle result message
        if isinstance(msg, ResultMessageBase):
            self.logger.debug("Handling ResultMessageBase with message_id: %s", msg.message_id)
            future = self._result_futures.get(msg.message_id)

            if future is None:
                # no listener for this result
                return
            if isinstance(msg, SuccessResultMessage):
                self.logger.debug(
                    "Received SuccessResultMessage for message_id: %s", msg.message_id
                )
                future.set_result(msg.result)
                return
            if isinstance(msg, ErrorResultMessage):
                self.logger.debug(
                    "Received ErrorResultMessage for message_id: %s with error: %s",
                    msg.message_id,
                    msg.error_code,
                )
                exc = ERROR_MAP[msg.error_code]
                future.set_exception(exc(msg.details))
                return

        # handle EventMessage
        if isinstance(msg, EventMessage):
            self.logger.debug("Received event: %s", msg)
            self._handle_event(msg)
            return

        # Log anything we can't handle here
        self.logger.debug(
            "Received message with unknown type '%s': %s",
            type(msg),
            msg,
        )

    def _handle_event(self, event: MassEvent) -> None:
        """Forward event to subscribers."""
        self.logger.debug("Handling event: %s", event.event)
        if self._stop_called:
            return

        assert self._loop

        if event.event == EventType.PROVIDERS_UPDATED:
            self._providers = {x["instance_id"]: ProviderInstance.from_dict(x) for x in event.data}

        for cb_func, event_filter, id_filter in self._subscribers:
            self.logger.debug(
                "Checking subscriber: %s for event: %s", cb_func.__name__, event.event
            )
            if not (event_filter is None or event.event in event_filter):
                self.logger.debug("Subscriber %s event filter mismatch", cb_func.__name__)
                continue
            if not (id_filter is None or event.object_id in id_filter):
                self.logger.debug("Subscriber %s id filter mismatch", cb_func.__name__)
                continue
            self.logger.debug("Calling subscriber: %s for event: %s", cb_func.__name__, event.event)
            if asyncio.iscoroutinefunction(cb_func):
                asyncio.run_coroutine_threadsafe(cb_func(event), self._loop)
            else:
                self._loop.call_soon_threadsafe(cb_func, event)

    async def __aenter__(self) -> Self:
        """Initialize and connect the connection to the Music Assistant Server."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Exit context manager."""
        await self.disconnect()
        return None

    def __repr__(self) -> str:
        """Return the representation."""
        conn_type = self.connection.__class__.__name__
        prefix = "" if self.connection.connected else "not "
        return f"{type(self).__name__}(connection={conn_type}, {prefix}connected)"

"""Several generic/basic helpers and utilities complementing the Music Assistant Client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from music_assistant_models.media_items import SearchResults


def compact_media_item_dict(item: dict[str, Any]) -> dict[str, Any]:
    """Return compacted MediaItem dict."""
    for key in (
        "metadata",
        "provider_mappings",
        "favorite",
        "timestamp_added",
        "timestamp_modified",
        "mbid",
    ):
        item.pop(key, None)
    for key, value in item.items():
        if isinstance(value, dict):
            item[key] = compact_media_item_dict(value)
        elif isinstance(value, list):
            for subitem in value:
                if not isinstance(subitem, dict):
                    continue
                compact_media_item_dict(subitem)
    return item


def searchresults_as_compact_dict(search_results: SearchResults) -> dict[str, Any]:
    """Return compacted search result dict."""
    dict_result: dict[str, list[dict[str, Any]]] = search_results.to_dict()
    for media_type_key in dict_result:
        for item in dict_result[media_type_key]:
            if not isinstance(item, dict):
                # guards against invalid data
                continue  # type: ignore[unreachable]
            # return limited result to prevent it being too verbose
            compact_media_item_dict(item)
    return dict_result

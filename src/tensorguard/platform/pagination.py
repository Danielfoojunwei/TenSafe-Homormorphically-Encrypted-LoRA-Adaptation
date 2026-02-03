"""
Cursor-based pagination utilities for TenSafe API.

This module provides standardized cursor pagination following industry best practices:
- Cursor-based pagination for stable results
- Consistent response format across all paginated endpoints
- Support for both forward and backward pagination
- Efficient database queries with proper indexing
"""

import base64
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Generic, List, Optional, TypeVar

from fastapi import Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Standard pagination parameters for API requests."""

    cursor: Optional[str] = Field(
        None,
        description="Cursor for pagination. Use the 'next_cursor' from the previous response.",
    )
    limit: int = Field(
        20,
        ge=1,
        le=100,
        description="Number of items to return per page (1-100).",
    )
    order: str = Field(
        "desc",
        pattern="^(asc|desc)$",
        description="Sort order: 'asc' for oldest first, 'desc' for newest first.",
    )


@dataclass
class CursorData:
    """Internal cursor data structure."""

    id: str
    timestamp: datetime
    direction: str = "next"

    def encode(self) -> str:
        """Encode cursor data to base64 string."""
        data = {
            "id": self.id,
            "ts": self.timestamp.isoformat() if self.timestamp else None,
            "dir": self.direction,
        }
        json_str = json.dumps(data, sort_keys=True)
        return base64.urlsafe_b64encode(json_str.encode()).decode()

    @classmethod
    def decode(cls, cursor: str) -> "CursorData":
        """Decode cursor string to CursorData."""
        try:
            json_str = base64.urlsafe_b64decode(cursor.encode()).decode()
            data = json.loads(json_str)
            return cls(
                id=data["id"],
                timestamp=datetime.fromisoformat(data["ts"]) if data.get("ts") else None,
                direction=data.get("dir", "next"),
            )
        except Exception as e:
            logger.warning(f"Invalid cursor: {e}")
            raise ValueError("Invalid pagination cursor")


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Standard paginated response format.

    This format is consistent with industry standards (Stripe, GitHub API)
    and provides all necessary information for client-side pagination.
    """

    data: List[Any] = Field(description="List of items in this page")
    has_more: bool = Field(description="Whether there are more items after this page")
    next_cursor: Optional[str] = Field(
        None,
        description="Cursor to fetch the next page. None if no more pages.",
    )
    previous_cursor: Optional[str] = Field(
        None,
        description="Cursor to fetch the previous page. None if on first page.",
    )
    total_count: Optional[int] = Field(
        None,
        description="Total number of items (only included if requested and efficient to compute).",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "data": [{"id": "tc_123", "name": "Training Client 1"}],
                "has_more": True,
                "next_cursor": "eyJpZCI6InRjXzEyMyIsInRzIjoiMjAyNC0wMS0wMVQwMDowMDowMCJ9",
                "previous_cursor": None,
                "total_count": 42,
            }
        }


class PageInfo(BaseModel):
    """Pagination metadata for GraphQL-style responses."""

    has_next_page: bool
    has_previous_page: bool
    start_cursor: Optional[str] = None
    end_cursor: Optional[str] = None


def paginate_query_params(
    cursor: Optional[str] = Query(
        None,
        description="Pagination cursor from previous response",
        example="eyJpZCI6InRjXzEyMyIsInRzIjoiMjAyNC0wMS0wMVQwMDowMDowMCJ9",
    ),
    limit: int = Query(
        20,
        ge=1,
        le=100,
        description="Number of items per page",
    ),
    order: str = Query(
        "desc",
        regex="^(asc|desc)$",
        description="Sort order (asc or desc)",
    ),
    include_total: bool = Query(
        False,
        description="Include total count (may be slow for large datasets)",
    ),
) -> dict:
    """
    FastAPI dependency for pagination query parameters.

    Usage:
        @router.get("/items")
        async def list_items(pagination: dict = Depends(paginate_query_params)):
            cursor_data = pagination.get("cursor_data")
            limit = pagination["limit"]
            ...
    """
    cursor_data = None
    if cursor:
        try:
            cursor_data = CursorData.decode(cursor)
        except ValueError:
            pass  # Will use default (start from beginning)

    return {
        "cursor": cursor,
        "cursor_data": cursor_data,
        "limit": limit,
        "order": order,
        "include_total": include_total,
    }


def create_paginated_response(
    items: List[Any],
    limit: int,
    id_field: str = "id",
    timestamp_field: str = "created_at",
    total_count: Optional[int] = None,
    has_previous: bool = False,
    first_item_cursor: Optional[str] = None,
) -> PaginatedResponse:
    """
    Create a standardized paginated response.

    Args:
        items: List of items to paginate
        limit: Requested page size
        id_field: Field name for item ID
        timestamp_field: Field name for timestamp (for cursor)
        total_count: Optional total count
        has_previous: Whether there are previous pages
        first_item_cursor: Cursor for the first item (for previous page)

    Returns:
        PaginatedResponse with proper cursor data
    """
    has_more = len(items) > limit
    if has_more:
        items = items[:limit]  # Trim to requested limit

    next_cursor = None
    if has_more and items:
        last_item = items[-1]
        # Handle both dict and object access
        if isinstance(last_item, dict):
            item_id = last_item.get(id_field)
            item_ts = last_item.get(timestamp_field)
        else:
            item_id = getattr(last_item, id_field, None)
            item_ts = getattr(last_item, timestamp_field, None)

        if item_id:
            cursor_data = CursorData(
                id=str(item_id),
                timestamp=item_ts if isinstance(item_ts, datetime) else datetime.utcnow(),
                direction="next",
            )
            next_cursor = cursor_data.encode()

    return PaginatedResponse(
        data=items,
        has_more=has_more,
        next_cursor=next_cursor,
        previous_cursor=first_item_cursor,
        total_count=total_count,
    )


def apply_cursor_filter(
    query,
    cursor_data: Optional[CursorData],
    id_column,
    timestamp_column,
    order: str = "desc",
):
    """
    Apply cursor-based filtering to a SQLAlchemy/SQLModel query.

    This function modifies the query to:
    1. Filter based on cursor position
    2. Apply proper ordering
    3. Handle edge cases

    Args:
        query: SQLAlchemy query object
        cursor_data: Decoded cursor data (or None for first page)
        id_column: SQLAlchemy column for ID
        timestamp_column: SQLAlchemy column for timestamp
        order: Sort order ('asc' or 'desc')

    Returns:
        Modified query with cursor filtering applied
    """
    if cursor_data:
        cursor_ts = cursor_data.timestamp
        cursor_id = cursor_data.id

        if order == "desc":
            # Get items older than cursor
            query = query.filter(
                (timestamp_column < cursor_ts)
                | ((timestamp_column == cursor_ts) & (id_column < cursor_id))
            )
        else:
            # Get items newer than cursor
            query = query.filter(
                (timestamp_column > cursor_ts)
                | ((timestamp_column == cursor_ts) & (id_column > cursor_id))
            )

    # Apply ordering
    if order == "desc":
        query = query.order_by(timestamp_column.desc(), id_column.desc())
    else:
        query = query.order_by(timestamp_column.asc(), id_column.asc())

    return query


class Paginator:
    """
    Helper class for paginating in-memory collections.

    Useful for paginating results that are already loaded in memory,
    such as filtered lists or aggregated data.
    """

    def __init__(
        self,
        items: List[Any],
        id_field: str = "id",
        timestamp_field: str = "created_at",
    ):
        self.items = items
        self.id_field = id_field
        self.timestamp_field = timestamp_field

    def get_item_id(self, item: Any) -> str:
        """Extract ID from item."""
        if isinstance(item, dict):
            return str(item.get(self.id_field, ""))
        return str(getattr(item, self.id_field, ""))

    def get_item_timestamp(self, item: Any) -> Optional[datetime]:
        """Extract timestamp from item."""
        if isinstance(item, dict):
            ts = item.get(self.timestamp_field)
        else:
            ts = getattr(item, self.timestamp_field, None)

        if isinstance(ts, datetime):
            return ts
        return None

    def paginate(
        self,
        cursor: Optional[str] = None,
        limit: int = 20,
        order: str = "desc",
        include_total: bool = False,
    ) -> PaginatedResponse:
        """
        Paginate the items.

        Args:
            cursor: Pagination cursor
            limit: Page size
            order: Sort order
            include_total: Whether to include total count

        Returns:
            PaginatedResponse
        """
        # Sort items
        sorted_items = sorted(
            self.items,
            key=lambda x: (self.get_item_timestamp(x) or datetime.min, self.get_item_id(x)),
            reverse=(order == "desc"),
        )

        # Apply cursor filter
        start_idx = 0
        if cursor:
            try:
                cursor_data = CursorData.decode(cursor)
                for i, item in enumerate(sorted_items):
                    if self.get_item_id(item) == cursor_data.id:
                        start_idx = i + 1
                        break
            except ValueError:
                pass

        # Get page of items (fetch limit + 1 to check for more)
        page_items = sorted_items[start_idx : start_idx + limit + 1]

        return create_paginated_response(
            items=page_items,
            limit=limit,
            id_field=self.id_field,
            timestamp_field=self.timestamp_field,
            total_count=len(self.items) if include_total else None,
            has_previous=start_idx > 0,
        )

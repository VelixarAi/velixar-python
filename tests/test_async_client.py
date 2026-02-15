"""Tests for Velixar async client."""

import pytest
import httpx
from unittest.mock import Mock, patch, AsyncMock

from velixar import AsyncVelixar, VelixarError, AuthenticationError
from velixar.types import Memory, SearchResult, MemoryTier


class TestAsyncVelixarClient:
    """Test the asynchronous Velixar client."""

    def setup_method(self):
        """Set up test client."""
        self.client = AsyncVelixar(api_key="test-key")

    def test_initialization(self):
        """Test client initialization."""
        assert self.client.api_key == "test-key"
        assert "api.velixarai.com" in self.client.base_url

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.request')
    async def test_store_memory(self, mock_request):
        """Test storing a memory."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "mem-123"}
        mock_request.return_value = mock_response

        result = await self.client.store("test content")
        
        assert result == "mem-123"
        mock_request.assert_called_once_with(
            "POST", 
            "/memory",
            json={"content": "test content", "tier": 2}
        )

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.request')
    async def test_search_memories(self, mock_request):
        """Test searching memories."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "memories": [{"id": "mem-1", "content": "test memory"}],
            "count": 1
        }
        mock_request.return_value = mock_response

        result = await self.client.search("test query")
        
        assert isinstance(result, SearchResult)
        assert len(result.memories) == 1
        assert result.count == 1
        
        mock_request.assert_called_once_with(
            "GET",
            "/memory/search",
            params={"q": "test query", "limit": 10}
        )

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.request')
    async def test_get_memory(self, mock_request):
        """Test getting a memory by ID."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "memory": {"id": "mem-1", "content": "test memory"}
        }
        mock_request.return_value = mock_response

        result = await self.client.get("mem-1")
        
        assert isinstance(result, Memory)
        assert result.id == "mem-1"
        
        mock_request.assert_called_once_with("GET", "/memory/mem-1")

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.request')
    async def test_delete_memory(self, mock_request):
        """Test deleting a memory."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"deleted": True}
        mock_request.return_value = mock_response

        result = await self.client.delete("mem-1")
        
        assert result is True
        mock_request.assert_called_once_with("DELETE", "/memory/mem-1")

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.request')
    async def test_error_handling(self, mock_request):
        """Test error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_request.return_value = mock_response

        with pytest.raises(AuthenticationError):
            await self.client.store("test")

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client as async context manager."""
        async with AsyncVelixar(api_key="test-key") as client:
            assert client.api_key == "test-key"
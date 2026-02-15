import pytest
import httpx
from unittest.mock import patch
from velixar import Velixar
from velixar.exceptions import AuthenticationError, VelixarError
from velixar.types import Memory, SearchResult


class TestVelixar:
    def test_init_with_api_key(self):
        client = Velixar(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.velixarai.com/v1"

    def test_init_without_api_key_raises_error(self):
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(AuthenticationError):
                Velixar()

    def test_init_with_custom_base_url(self):
        client = Velixar(api_key="test-key", base_url="https://custom.api.com")
        assert client.base_url == "https://custom.api.com"

    @pytest.fixture
    def client(self):
        return Velixar(api_key="test-key")

    def test_store(self, client, httpx_mock):
        httpx_mock.add_response(json={"id": "mem-123"})
        
        result = client.store("test content")
        
        assert result == "mem-123"
        request = httpx_mock.get_request()
        assert request.method == "POST"
        assert request.url.path == "/memory"
        assert "test content" in request.content.decode()

    def test_search(self, client, httpx_mock):
        mock_response = {
            "memories": [{"id": "mem-1", "content": "test"}],
            "count": 1
        }
        httpx_mock.add_response(json=mock_response)
        
        result = client.search("query")
        
        assert isinstance(result, SearchResult)
        assert len(result.memories) == 1
        assert result.count == 1
        request = httpx_mock.get_request()
        assert "q=query" in str(request.url)

    def test_get(self, client, httpx_mock):
        mock_memory = {"id": "mem-123", "content": "test content"}
        httpx_mock.add_response(json={"memory": mock_memory})
        
        result = client.get("mem-123")
        
        assert isinstance(result, Memory)
        assert result.id == "mem-123"
        request = httpx_mock.get_request()
        assert request.url.path == "/memory/mem-123"

    def test_delete(self, client, httpx_mock):
        httpx_mock.add_response(json={"deleted": True})
        
        result = client.delete("mem-123")
        
        assert result is True
        request = httpx_mock.get_request()
        assert request.method == "DELETE"
        assert request.url.path == "/memory/mem-123"

    def test_error_handling(self, client, httpx_mock):
        httpx_mock.add_response(status_code=500)
        
        with pytest.raises(VelixarError):
            client.store("test")
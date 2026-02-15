"""Custom exceptions for Velixar SDK."""


class VelixarError(Exception):
    """Base exception for Velixar SDK."""
    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class AuthenticationError(VelixarError):
    """Invalid or missing API key."""
    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(message, status_code=401)


class RateLimitError(VelixarError):
    """Rate limit exceeded."""
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int | None = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class NotFoundError(VelixarError):
    """Resource not found."""
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, status_code=404)


class ValidationError(VelixarError):
    """Invalid request parameters."""
    def __init__(self, message: str = "Invalid request"):
        super().__init__(message, status_code=400)


class InsufficientScopeError(VelixarError):
    """API key lacks required scope."""
    def __init__(self, message: str = "Insufficient scope for this operation"):
        super().__init__(message, status_code=403)

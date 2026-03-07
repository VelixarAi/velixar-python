import os
from dataclasses import dataclass
from typing import FrozenSet

@dataclass(frozen=True)
class Config:
    API_BASE_URL: str = os.environ.get('VELIXAR_API_BASE_URL', 'https://api.velixarai.com')
    BRAINIAC_API_URL: str = os.environ.get('VELIXAR_BRAINIAC_URL', 'https://api.velixarai.com')

def validate_config() -> None:
    if os.environ.get('ENV') == 'production':
        if not os.environ.get('VELIXAR_API_BASE_URL'):
            print('Warning: VELIXAR_API_BASE_URL not set, using default')
        if not os.environ.get('VELIXAR_BRAINIAC_URL'):
            print('Warning: VELIXAR_BRAINIAC_URL not set, using default')

config = Config()
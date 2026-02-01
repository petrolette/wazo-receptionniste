"""Configuration du service réceptionniste IA."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ARIConfig:
    """Configuration Asterisk ARI."""
    host: str = os.getenv("ARI_HOST", "127.0.0.1")
    port: int = int(os.getenv("ARI_PORT", "5039"))
    user: str = os.getenv("ARI_USER", "xivo")
    password: str = os.getenv("ARI_PASSWORD", "")
    app_name: str = os.getenv("ARI_APP", "receptionniste")

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/ari/events?app={self.app_name}&api_key={self.user}:{self.password}"


@dataclass
class OpenAIConfig:
    """Configuration OpenAI."""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    tts_model: str = "tts-1"
    tts_voice: str = "nova"  # Voix féminine naturelle
    stt_model: str = "whisper-1"
    chat_model: str = "gpt-4o-mini"


@dataclass
class WazoConfig:
    """Configuration Wazo API."""
    host: str = os.getenv("WAZO_HOST", "127.0.0.1")
    auth_port: int = int(os.getenv("WAZO_AUTH_PORT", "9497"))
    confd_port: int = int(os.getenv("WAZO_CONFD_PORT", "9486"))
    calld_port: int = int(os.getenv("WAZO_CALLD_PORT", "9500"))


@dataclass
class Service:
    """Un service de l'entreprise."""
    extension: str
    name: str


@dataclass
class CompanyConfig:
    """Configuration de l'entreprise."""
    name: str = os.getenv("COMPANY_NAME", "Toni Küpfer SA")
    greeting: str = os.getenv(
        "GREETING_MESSAGE",
        "Bonjour et bienvenue chez Toni Küpfer SA. Quel service souhaitez-vous joindre ?"
    )
    ring_timeout: int = int(os.getenv("RING_TIMEOUT", "3"))
    services: list[Service] = field(default_factory=list)

    def __post_init__(self):
        services_str = os.getenv("SERVICES", "")
        if services_str:
            for item in services_str.split(","):
                ext, name = item.split(":")
                self.services.append(Service(extension=ext.strip(), name=name.strip()))


@dataclass
class N8NConfig:
    """Configuration n8n webhook."""
    webhook_url: str = os.getenv("N8N_WEBHOOK_URL", "")


@dataclass
class Config:
    """Configuration globale."""
    ari: ARIConfig = field(default_factory=ARIConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    wazo: WazoConfig = field(default_factory=WazoConfig)
    company: CompanyConfig = field(default_factory=CompanyConfig)
    n8n: N8NConfig = field(default_factory=N8NConfig)

    # Chemins
    recordings_dir: str = "/app/recordings"
    audio_cache_dir: str = "/app/audio_cache"


# Instance globale
config = Config()

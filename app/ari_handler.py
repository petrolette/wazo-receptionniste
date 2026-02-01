"""Gestion des appels via Asterisk ARI."""

import asyncio
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import aiohttp
import websockets
import structlog

from app.config import config
from app.ai_handler import ai_handler

logger = structlog.get_logger()


class CallState(Enum):
    """États d'un appel."""
    GREETING = "greeting"
    WAITING_SERVICE_CHOICE = "waiting_service_choice"
    TRANSFERRING = "transferring"
    COLLECTING_MESSAGE = "collecting_message"
    ENDING = "ending"


@dataclass
class CallSession:
    """Session d'appel en cours."""
    channel_id: str
    caller_id: str
    state: CallState = CallState.GREETING
    target_service: Optional[str] = None
    message_info: dict = field(default_factory=dict)
    conversation: list = field(default_factory=list)
    retry_count: int = 0


class ARIHandler:
    """Gestionnaire des appels via Asterisk ARI."""

    def __init__(self):
        self.sessions: dict[str, CallSession] = {}
        self.http_session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        """Démarre le handler ARI."""
        self.http_session = aiohttp.ClientSession(
            auth=aiohttp.BasicAuth(config.ari.user, config.ari.password)
        )

        # Pré-générer les audios courants
        await ai_handler.pre_generate_common_audio()

        # Connexion WebSocket
        await self.connect_websocket()

    async def stop(self):
        """Arrête le handler."""
        if self.http_session:
            await self.http_session.close()

    async def connect_websocket(self):
        """Connexion WebSocket à ARI."""
        logger.info("ari_connecting", url=config.ari.ws_url)

        while True:
            try:
                async with websockets.connect(config.ari.ws_url) as ws:
                    logger.info("ari_connected")
                    async for message in ws:
                        await self.handle_event(json.loads(message))
            except Exception as e:
                logger.error("ari_connection_error", error=str(e))
                await asyncio.sleep(5)

    async def handle_event(self, event: dict):
        """Traite un événement ARI."""
        event_type = event.get("type")
        logger.debug("ari_event", type=event_type)

        handlers = {
            "StasisStart": self.on_stasis_start,
            "StasisEnd": self.on_stasis_end,
            "PlaybackFinished": self.on_playback_finished,
            "RecordingFinished": self.on_recording_finished,
            "ChannelHangupRequest": self.on_hangup,
            "ChannelDestroyed": self.on_channel_destroyed,
        }

        handler = handlers.get(event_type)
        if handler:
            await handler(event)

    async def on_stasis_start(self, event: dict):
        """Nouvel appel entrant."""
        channel = event.get("channel", {})
        channel_id = channel.get("id")
        caller_id = channel.get("caller", {}).get("number", "inconnu")

        logger.info("call_incoming", channel_id=channel_id, caller=caller_id)

        # Créer la session
        session = CallSession(channel_id=channel_id, caller_id=caller_id)
        self.sessions[channel_id] = session

        # Répondre à l'appel
        await self.answer_channel(channel_id)

        # Jouer le message d'accueil
        await self.play_greeting(channel_id)

    async def on_stasis_end(self, event: dict):
        """Appel terminé dans Stasis."""
        channel = event.get("channel", {})
        channel_id = channel.get("id")
        logger.info("stasis_end", channel_id=channel_id)

    async def on_channel_destroyed(self, event: dict):
        """Canal détruit."""
        channel = event.get("channel", {})
        channel_id = channel.get("id")

        if channel_id in self.sessions:
            session = self.sessions.pop(channel_id)
            logger.info("call_ended", channel_id=channel_id, caller=session.caller_id)

    async def on_hangup(self, event: dict):
        """Demande de raccroché."""
        channel = event.get("channel", {})
        channel_id = channel.get("id")
        logger.info("hangup_requested", channel_id=channel_id)

    async def on_playback_finished(self, event: dict):
        """Lecture audio terminée."""
        playback = event.get("playback", {})
        target_uri = playback.get("target_uri", "")

        # Extraire le channel_id
        if target_uri.startswith("channel:"):
            channel_id = target_uri.replace("channel:", "")
        else:
            return

        session = self.sessions.get(channel_id)
        if not session:
            return

        logger.info("playback_finished", channel_id=channel_id, state=session.state.value)

        # Selon l'état, lancer l'enregistrement
        if session.state == CallState.GREETING:
            session.state = CallState.WAITING_SERVICE_CHOICE
            await self.start_recording(channel_id)

        elif session.state == CallState.WAITING_SERVICE_CHOICE:
            await self.start_recording(channel_id)

        elif session.state == CallState.COLLECTING_MESSAGE:
            await self.start_recording(channel_id)

        elif session.state == CallState.ENDING:
            await self.hangup_channel(channel_id)

    async def on_recording_finished(self, event: dict):
        """Enregistrement terminé."""
        recording = event.get("recording", {})
        recording_name = recording.get("name")
        target_uri = recording.get("target_uri", "")

        if target_uri.startswith("channel:"):
            channel_id = target_uri.replace("channel:", "")
        else:
            return

        session = self.sessions.get(channel_id)
        if not session:
            return

        logger.info("recording_finished", channel_id=channel_id, recording=recording_name)

        # Récupérer et traiter l'audio
        audio_path = f"/var/spool/asterisk/recording/{recording_name}.wav"

        try:
            # Transcrire
            transcript = await ai_handler.speech_to_text(audio_path)

            if session.state == CallState.WAITING_SERVICE_CHOICE:
                await self.handle_service_choice(channel_id, transcript)

            elif session.state == CallState.COLLECTING_MESSAGE:
                await self.handle_message_collection(channel_id, transcript)

        except Exception as e:
            logger.error("recording_processing_error", error=str(e))
            await self.play_error_and_retry(channel_id)

    async def handle_service_choice(self, channel_id: str, transcript: str):
        """Traite le choix de service."""
        session = self.sessions.get(channel_id)
        if not session:
            return

        # Analyser l'intention
        result = await ai_handler.understand_intent(transcript)

        if result["service"]:
            # Service trouvé, transférer
            session.target_service = result["service"]
            session.state = CallState.TRANSFERRING

            # Annoncer le transfert
            audio_path = await ai_handler.text_to_speech(result["response"])
            await self.play_audio(channel_id, audio_path)

            # Transférer vers l'extension
            await self.transfer_to_extension(
                channel_id,
                result["service"].extension
            )
        else:
            # Pas compris, demander clarification
            session.retry_count += 1

            if session.retry_count >= 3:
                # Trop de tentatives, passer en prise de message
                await self.start_message_collection(channel_id)
            else:
                audio_path = await ai_handler.text_to_speech(result["response"])
                await self.play_audio(channel_id, audio_path)

    async def handle_message_collection(self, channel_id: str, transcript: str):
        """Traite la collecte d'informations pour un message."""
        session = self.sessions.get(channel_id)
        if not session:
            return

        # Ajouter à la conversation
        session.conversation.append({"role": "user", "content": transcript})

        # Analyser
        result = await ai_handler.collect_message_info(session.conversation, transcript)

        # Mettre à jour les infos
        session.message_info.update(result["info"])

        # Générer la réponse audio
        audio_path = await ai_handler.text_to_speech(result["response"])

        if result["complete"]:
            # Message complet, envoyer à n8n et terminer
            session.state = CallState.ENDING
            await self.send_message_to_n8n(session)

        session.conversation.append({"role": "assistant", "content": result["response"]})
        await self.play_audio(channel_id, audio_path)

    async def start_message_collection(self, channel_id: str):
        """Démarre la collecte de message."""
        session = self.sessions.get(channel_id)
        if not session:
            return

        session.state = CallState.COLLECTING_MESSAGE
        session.conversation = []

        message = "Le service est actuellement occupé. Puis-je prendre un message ? Quel est votre nom ?"
        audio_path = await ai_handler.text_to_speech(message)
        session.conversation.append({"role": "assistant", "content": message})

        await self.play_audio(channel_id, audio_path)

    async def transfer_to_extension(self, channel_id: str, extension: str):
        """Transfère l'appel vers une extension.

        Utilise un timeout pour détecter si personne ne répond.
        """
        session = self.sessions.get(channel_id)
        if not session:
            return

        logger.info("transferring", channel_id=channel_id, extension=extension)

        # Créer un canal vers l'extension
        try:
            # Originate vers l'extension avec timeout
            async with self.http_session.post(
                f"{config.ari.url}/ari/channels",
                params={
                    "endpoint": f"PJSIP/{extension}",
                    "app": config.ari.app_name,
                    "appArgs": f"transfer,{channel_id}",
                    "timeout": config.company.ring_timeout,
                    "callerId": session.caller_id,
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info("transfer_initiated", target_channel=data.get("id"))

                    # Attendre le timeout
                    await asyncio.sleep(config.company.ring_timeout + 1)

                    # Si on arrive ici, personne n'a répondu
                    # Passer en mode prise de message
                    if session.state == CallState.TRANSFERRING:
                        logger.info("transfer_timeout", channel_id=channel_id)
                        await self.start_message_collection(channel_id)
                else:
                    logger.error("transfer_failed", status=resp.status)
                    await self.start_message_collection(channel_id)

        except Exception as e:
            logger.error("transfer_error", error=str(e))
            await self.start_message_collection(channel_id)

    async def send_message_to_n8n(self, session: CallSession):
        """Envoie le message à n8n pour notification."""
        if not config.n8n.webhook_url:
            logger.warning("n8n_webhook_not_configured")
            return

        payload = {
            "caller_id": session.caller_id,
            "service": session.target_service.name if session.target_service else "Non spécifié",
            "nom": session.message_info.get("nom", "Non spécifié"),
            "societe": session.message_info.get("societe", "Non spécifiée"),
            "sujet": session.message_info.get("sujet", "Non spécifié"),
        }

        logger.info("sending_to_n8n", payload=payload)

        try:
            async with self.http_session.post(
                config.n8n.webhook_url,
                json=payload
            ) as resp:
                if resp.status == 200:
                    logger.info("n8n_notification_sent")
                else:
                    logger.error("n8n_notification_failed", status=resp.status)
        except Exception as e:
            logger.error("n8n_error", error=str(e))

    # === Méthodes ARI de bas niveau ===

    async def answer_channel(self, channel_id: str):
        """Répond à un appel."""
        async with self.http_session.post(
            f"{config.ari.url}/ari/channels/{channel_id}/answer"
        ) as resp:
            logger.debug("channel_answered", channel_id=channel_id, status=resp.status)

    async def play_audio(self, channel_id: str, audio_path: str):
        """Joue un fichier audio."""
        # Utiliser le dossier custom de Wazo
        # /app/audio_cache/fichier.wav -> custom/fichier
        filename = audio_path.split("/")[-1].replace(".wav", "")
        sound_path = f"custom/{filename}"

        async with self.http_session.post(
            f"{config.ari.url}/ari/channels/{channel_id}/play",
            params={"media": f"sound:{sound_path}"}
        ) as resp:
            logger.debug("playing_audio", channel_id=channel_id, path=sound_path, status=resp.status)

    async def play_greeting(self, channel_id: str):
        """Joue le message d'accueil."""
        audio_path = await ai_handler.text_to_speech(config.company.greeting)
        await self.play_audio(channel_id, audio_path)

    async def start_recording(self, channel_id: str, max_duration: int = 10):
        """Démarre un enregistrement."""
        import uuid
        recording_name = f"rec_{uuid.uuid4().hex[:8]}"

        async with self.http_session.post(
            f"{config.ari.url}/ari/channels/{channel_id}/record",
            params={
                "name": recording_name,
                "format": "wav",
                "maxDurationSeconds": max_duration,
                "maxSilenceSeconds": 2,  # Arrête après 2s de silence
                "beep": "no",
                "terminateOn": "#",  # L'appelant peut appuyer sur # pour terminer
            }
        ) as resp:
            logger.debug("recording_started", channel_id=channel_id, name=recording_name, status=resp.status)

    async def hangup_channel(self, channel_id: str):
        """Raccroche un appel."""
        async with self.http_session.delete(
            f"{config.ari.url}/ari/channels/{channel_id}"
        ) as resp:
            logger.debug("channel_hungup", channel_id=channel_id, status=resp.status)

    async def play_error_and_retry(self, channel_id: str):
        """Joue un message d'erreur et retente."""
        audio_path = await ai_handler.text_to_speech(
            "Je n'ai pas compris. Pouvez-vous répéter s'il vous plaît ?"
        )
        await self.play_audio(channel_id, audio_path)


# Instance globale
ari_handler = ARIHandler()

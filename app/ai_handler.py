"""Gestion de l'IA : OpenAI Whisper (STT), GPT-4, TTS."""

import os
import hashlib
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
import structlog

from app.config import config

logger = structlog.get_logger()


class AIHandler:
    """Gestionnaire IA pour la réceptionniste."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=config.openai.api_key)
        self.audio_cache = Path(config.audio_cache_dir)
        self.audio_cache.mkdir(parents=True, exist_ok=True)

        # Prompts système
        self.system_prompt = f"""Tu es la réceptionniste virtuelle de {config.company.name}.
Tu parles français de manière professionnelle et chaleureuse.

Services disponibles :
{self._format_services()}

Ton rôle :
1. Comprendre quel service l'appelant souhaite joindre
2. Si tu comprends, retourne le nom exact du service
3. Si tu ne comprends pas, demande poliment de répéter

Réponds UNIQUEMENT avec le nom du service ou une question de clarification.
Ne fais pas de phrases longues."""

        self.message_prompt = f"""Tu es la réceptionniste virtuelle de {config.company.name}.
L'appelant n'a pas pu joindre le service souhaité.
Tu dois collecter poliment :
1. Son nom
2. Sa société (si applicable)
3. Le sujet de son appel

Sois concis et professionnel. Une question à la fois."""

    def _format_services(self) -> str:
        """Formate la liste des services."""
        return "\n".join([f"- {s.name} (extension {s.extension})" for s in config.company.services])

    async def text_to_speech(self, text: str, use_cache: bool = True) -> str:
        """Convertit du texte en fichier audio.

        Args:
            text: Texte à convertir
            use_cache: Utiliser le cache pour les messages récurrents

        Returns:
            Chemin du fichier audio généré
        """
        # Générer un hash pour le cache
        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        cache_path = self.audio_cache / f"{text_hash}.wav"

        if use_cache and cache_path.exists():
            logger.debug("tts_cache_hit", text=text[:50])
            return str(cache_path)

        logger.info("tts_generating", text=text[:50])

        response = await self.client.audio.speech.create(
            model=config.openai.tts_model,
            voice=config.openai.tts_voice,
            input=text,
            response_format="wav"
        )

        # Sauvegarder le fichier
        with open(cache_path, "wb") as f:
            async for chunk in response.iter_bytes():
                f.write(chunk)

        logger.info("tts_generated", path=str(cache_path))
        return str(cache_path)

    async def speech_to_text(self, audio_path: str) -> str:
        """Convertit un fichier audio en texte.

        Args:
            audio_path: Chemin du fichier audio

        Returns:
            Transcription du texte
        """
        logger.info("stt_transcribing", path=audio_path)

        with open(audio_path, "rb") as audio_file:
            transcript = await self.client.audio.transcriptions.create(
                model=config.openai.stt_model,
                file=audio_file,
                language="fr"
            )

        text = transcript.text
        logger.info("stt_result", text=text)
        return text

    async def understand_intent(self, user_text: str) -> dict:
        """Comprend l'intention de l'appelant.

        Args:
            user_text: Ce que l'appelant a dit

        Returns:
            Dict avec 'service' (nom du service ou None) et 'response' (texte à dire)
        """
        logger.info("understanding_intent", user_text=user_text)

        response = await self.client.chat.completions.create(
            model=config.openai.chat_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0.3
        )

        ai_response = response.choices[0].message.content.strip()
        logger.info("intent_response", response=ai_response)

        # Chercher si la réponse correspond à un service
        for service in config.company.services:
            if service.name.lower() in ai_response.lower():
                return {
                    "service": service,
                    "response": f"Je vous transfère au {service.name}. Un instant s'il vous plaît."
                }

        # Pas de service trouvé, demander clarification
        return {
            "service": None,
            "response": ai_response
        }

    async def collect_message_info(self, conversation: list[dict], user_text: str) -> dict:
        """Collecte les informations pour un message.

        Args:
            conversation: Historique de la conversation
            user_text: Dernière réponse de l'appelant

        Returns:
            Dict avec 'complete' (bool), 'info' (dict), 'response' (str)
        """
        messages = [{"role": "system", "content": self.message_prompt}]
        messages.extend(conversation)
        messages.append({"role": "user", "content": user_text})

        # Demander à GPT d'analyser
        analysis_prompt = """Analyse la conversation et retourne un JSON:
{
    "complete": true/false,
    "info": {
        "nom": "...",
        "societe": "...",
        "sujet": "..."
    },
    "next_question": "question à poser si pas complet"
}"""

        messages.append({"role": "user", "content": analysis_prompt})

        response = await self.client.chat.completions.create(
            model=config.openai.chat_model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        import json
        result = json.loads(response.choices[0].message.content)

        if result.get("complete"):
            return {
                "complete": True,
                "info": result.get("info", {}),
                "response": "Merci pour ces informations. Nous vous rappellerons dès que possible. Au revoir et bonne journée."
            }
        else:
            return {
                "complete": False,
                "info": result.get("info", {}),
                "response": result.get("next_question", "Pouvez-vous me donner plus de détails ?")
            }

    async def pre_generate_common_audio(self):
        """Pré-génère les messages audio courants."""
        common_messages = [
            config.company.greeting,
            "Je vous transfère. Un instant s'il vous plaît.",
            "Le service est actuellement occupé. Puis-je prendre un message ?",
            "Puis-je avoir votre nom s'il vous plaît ?",
            "Et votre société ?",
            "Quel est le sujet de votre appel ?",
            "Merci pour ces informations. Nous vous rappellerons dès que possible. Au revoir et bonne journée.",
            "Je n'ai pas compris. Pouvez-vous répéter s'il vous plaît ?",
            "Au revoir et bonne journée."
        ]

        # Ajouter les messages de transfert pour chaque service
        for service in config.company.services:
            common_messages.append(f"Je vous transfère au {service.name}. Un instant s'il vous plaît.")

        logger.info("pre_generating_audio", count=len(common_messages))

        for message in common_messages:
            await self.text_to_speech(message, use_cache=True)

        logger.info("pre_generation_complete")


# Instance globale
ai_handler = AIHandler()

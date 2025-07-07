#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRANSWER - AI Meeting Assistant Backend
=======================================

EN: Real-time transcription, translation, and AI-powered response suggestions
PT: Transcrição em tempo real, tradução e sugestões de resposta com IA

Author: Transwer Team
Version: 2.0.0
License: MIT
"""

import os
import threading
import queue
import json
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import numpy as np
import sounddevice as sd
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

# === LOGGING SETUP ===
# EN: Configure logging for better debugging
# PT: Configura logging para melhor debug
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === DYNAMIC IMPORTS ===
# EN: Import optional dependencies with fallbacks
# PT: Importa dependências opcionais com fallbacks

# Vosk STT Engine / Motor STT Vosk
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
    logger.info("✅ Vosk library loaded / Biblioteca Vosk carregada")
except ImportError:
    VOSK_AVAILABLE = False
    logger.warning("⚠️ Vosk library not found / Biblioteca Vosk não encontrada")

# Google Cloud Services / Serviços Google Cloud
try:
    from google.cloud import speech, speech_v2, translate_v2 as translate
    from google.oauth2 import service_account
    GOOGLE_AVAILABLE = True
    logger.info("✅ Google Cloud libraries loaded / Bibliotecas Google Cloud carregadas")
except ImportError:
    GOOGLE_AVAILABLE = False
    logger.warning("⚠️ Google Cloud libraries not found / Bibliotecas Google Cloud não encontradas")

# Faster Whisper STT Engine / Motor STT Faster Whisper
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    logger.info("✅ Faster-Whisper library loaded / Biblioteca Faster-Whisper carregada")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("⚠️ Faster-Whisper library not found / Biblioteca Faster-Whisper não encontrada")

# OpenAI API / API OpenAI
try:
    from openai import OpenAI, OpenAIError
    OPENAI_AVAILABLE = True
    logger.info("✅ OpenAI library loaded / Biblioteca OpenAI carregada")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("⚠️ OpenAI library not found / Biblioteca OpenAI não encontrada")

# === APPLICATION CONFIGURATION ===
# EN: Load environment variables and setup Flask app
# PT: Carrega variáveis de ambiente e configura app Flask
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'transwer_secret_key_2025')
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# Configuration file / Arquivo de configuração
CONFIG_FILE = "transwer_config.json"

# === DATA CLASSES ===
# EN: Define data structures for better type safety
# PT: Define estruturas de dados para melhor segurança de tipos

@dataclass
class TranscriptionSegment:
    """
    EN: Represents a segment of transcribed text
    PT: Representa um segmento de texto transcrito
    """
    text: str
    timestamp: float
    is_final: bool
    confidence: Optional[float] = None

@dataclass
class TranslationSegment:
    """
    EN: Represents a translated text segment
    PT: Representa um segmento de texto traduzido
    """
    original_text: str
    translated_text: str
    target_language: str
    timestamp: float

@dataclass
class AudioBuffer:
    """
    EN: Manages audio data buffering
    PT: Gerencia buffering de dados de áudio
    """
    data: queue.Queue
    sample_rate: int = 16000
    channels: int = 1
    
    def clear(self):
        """Clear the audio buffer / Limpa o buffer de áudio"""
        with self.data.mutex:
            self.data.queue.clear()

# === GLOBAL STATE MANAGEMENT ===
# EN: Thread-safe global state management
# PT: Gerenciamento de estado global thread-safe

class TranswerState:
    """
    EN: Central state manager for the application
    PT: Gerenciador de estado central da aplicação
    """
    
    def __init__(self):
        # Configuration / Configuração
        self.config: Dict[str, Any] = {}
        
        # Audio management / Gerenciamento de áudio
        self.audio_buffer = AudioBuffer(data=queue.Queue())
        
        # Text buffers with thread-safe locks / Buffers de texto com locks thread-safe
        self._transcription_lock = threading.Lock()
        self._translation_lock = threading.Lock()
        self._transcription_buffer = ""  # EN: Complete transcribed text / PT: Texto transcrito completo
        self._translation_buffer = ""    # EN: Complete translated text / PT: Texto traduzido completo
        self._partial_text = ""          # EN: Current partial transcription / PT: Transcrição parcial atual
        
        # Processing control / Controle de processamento
        self.stop_event = threading.Event()
        self.processing_threads: List[threading.Thread] = []
        self.is_processing = False
        
        # Translation queue for real-time processing / Fila de tradução para processamento em tempo real
        self.translation_queue = queue.Queue()
        self.translation_thread: Optional[threading.Thread] = None
        
        logger.info("🔧 TranswerState initialized / TranswerState inicializado")
    
    @property
    def transcription_buffer(self) -> str:
        """Thread-safe access to transcription buffer / Acesso thread-safe ao buffer de transcrição"""
        with self._transcription_lock:
            return self._transcription_buffer
    
    @transcription_buffer.setter
    def transcription_buffer(self, value: str):
        """Thread-safe setting of transcription buffer / Configuração thread-safe do buffer de transcrição"""
        with self._transcription_lock:
            self._transcription_buffer = value
    
    def append_transcription(self, text: str) -> str:
        """
        EN: Thread-safely append text to transcription buffer
        PT: Adiciona texto ao buffer de transcrição de forma thread-safe
        """
        with self._transcription_lock:
            self._transcription_buffer += f" {text.strip()}"
            return self._transcription_buffer
    
    @property
    def translation_buffer(self) -> str:
        """Thread-safe access to translation buffer / Acesso thread-safe ao buffer de tradução"""
        with self._translation_lock:
            return self._translation_buffer
    
    @translation_buffer.setter
    def translation_buffer(self, value: str):
        """Thread-safe setting of translation buffer / Configuração thread-safe do buffer de tradução"""
        with self._translation_lock:
            self._translation_buffer = value
    
    def append_translation(self, text: str) -> str:
        """
        EN: Thread-safely append text to translation buffer
        PT: Adiciona texto ao buffer de tradução de forma thread-safe
        """
        with self._translation_lock:
            self._translation_buffer += f" {text.strip()}"
            return self._translation_buffer
    
    def clear_buffers(self):
        """
        EN: Clear all text buffers
        PT: Limpa todos os buffers de texto
        """
        with self._transcription_lock:
            self._transcription_buffer = ""
            self._partial_text = ""
        
        with self._translation_lock:
            self._translation_buffer = ""
        
        logger.info("🧹 All buffers cleared / Todos os buffers limpos")
    
    def stop_all_processing(self):
        """
        EN: Stop all processing threads safely
        PT: Para todas as threads de processamento com segurança
        """
        logger.info("🛑 Stopping all processing / Parando todo o processamento")
        
        # Signal all threads to stop / Sinaliza todas as threads para parar
        self.stop_event.set()
        
        # Clear audio buffer / Limpa buffer de áudio
        self.audio_buffer.clear()
        
        # Wait for threads to finish / Espera threads terminarem
        for thread in self.processing_threads:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        
        # Clean up / Limpeza
        self.processing_threads.clear()
        self.is_processing = False
        
        logger.info("✅ All processing stopped / Todo o processamento parado")

# EN: Global state instance
# PT: Instância de estado global
state = TranswerState()

# === CONFIGURATION MANAGEMENT ===
# EN: Handle application configuration loading and saving
# PT: Gerencia carregamento e salvamento de configuração da aplicação

def load_application_config() -> Dict[str, Any]:
    """
    EN: Load configuration from file or return defaults
    PT: Carrega configuração do arquivo ou retorna padrões
    """
    default_config = {
        "sttEngine": "vosk",
        "translationEngine": "openai", 
        "googleCreds": "",
        "apiKey": "",
        "audioDevice": 0,
        "translationLang": "pt-BR",
        "translationDisabled": False,
        "suggestionsEnabled": True,
        "googleRegion": "us-central1",
        "fastwhisperModel": "base",
        "computeType": "int8"
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                logger.info(f"📁 Configuration loaded from {CONFIG_FILE}")
                return {**default_config, **loaded_config}  # Merge with defaults
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"❌ Error loading config file: {e}")
    
    logger.info("📋 Using default configuration / Usando configuração padrão")
    return default_config

def save_application_config(config: Dict[str, Any]):
    """
    EN: Save configuration to file
    PT: Salva configuração no arquivo
    """
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info(f"💾 Configuration saved to {CONFIG_FILE}")
        socketio.emit('notification', {
            'status': 'success', 
            'message': 'Settings saved successfully / Configurações salvas com sucesso'
        })
    except Exception as e:
        logger.error(f"❌ Error saving config: {e}")
        socketio.emit('notification', {
            'status': 'error', 
            'message': f'Failed to save settings / Falha ao salvar configurações: {e}'
        })

# === UTILITY FUNCTIONS ===
# EN: Helper functions for credentials and processing
# PT: Funções auxiliares para credenciais e processamento

def get_google_credentials(config: Dict[str, Any]):
    """
    EN: Safely parse and return Google Cloud credentials
    PT: Analisa e retorna credenciais do Google Cloud com segurança
    """
    creds_json_str = config.get('googleCreds', '')
    if not creds_json_str:
        raise ValueError("Google Cloud JSON credentials are required / Credenciais JSON do Google Cloud são necessárias")
    
    try:
        creds_info = json.loads(creds_json_str)
        return service_account.Credentials.from_service_account_info(creds_info)
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Invalid Google credentials format / Formato de credenciais Google inválido: {e}")

def safe_emit_notification(status: str, message: str):
    """
    EN: Safely emit notification (handles potential SocketIO errors)
    PT: Emite notificação com segurança (trata possíveis erros do SocketIO)
    """
    try:
        socketio.emit('notification', {'status': status, 'message': message})
    except Exception as e:
        logger.error(f"Failed to emit notification: {e}")

def safe_emit_api_status(status: str, message: str):
    """
    EN: Safely emit API status update
    PT: Emite atualização de status da API com segurança
    """
    try:
        socketio.emit('api_status', {'status': status, 'message': message})
    except Exception as e:
        logger.error(f"Failed to emit API status: {e}")

# === AUDIO CAPTURE ===
# EN: Handle microphone audio capture in a separate thread
# PT: Gerencia captura de áudio do microfone em thread separada

def audio_capture_worker(device_index: int, sample_rate: int = 16000):
    """
    EN: Continuous audio capture from microphone
    PT: Captura contínua de áudio do microfone
    """
    try:
        device_info = sd.query_devices(device_index, 'input')
        device_name = device_info.get('name', f'Device {device_index}')
        
        logger.info(f"🎙️ Starting audio capture on: {device_name}")
        safe_emit_notification('info', f'Listening on: {device_name}')
        
        def audio_callback(indata, frames, time, status):
            """
            EN: Callback function for audio stream
            PT: Função callback para stream de áudio
            """
            if status:
                logger.warning(f"Audio stream status: {status}")
            
            if state.is_processing and not state.stop_event.is_set():
                try:
                    # EN: Convert to bytes and add to buffer
                    # PT: Converte para bytes e adiciona ao buffer
                    audio_bytes = bytes(indata)
                    state.audio_buffer.data.put(audio_bytes, block=False)
                except queue.Full:
                    logger.warning("Audio buffer full, dropping frame / Buffer de áudio cheio, descartando frame")
        
        # EN: Start audio stream
        # PT: Inicia stream de áudio
        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=4096,
            device=device_index,
            dtype='int16',
            channels=1,
            callback=audio_callback
        ):
            # EN: Keep thread alive until stop signal
            # PT: Mantém thread viva até sinal de parada
            state.stop_event.wait()
        
        logger.info("🔇 Audio capture stopped / Captura de áudio parada")
        
    except Exception as e:
        error_msg = f"Audio capture error / Erro na captura de áudio: {e}"
        logger.error(error_msg)
        safe_emit_notification('error', error_msg)
        
        # EN: Auto-stop if audio fails
        # PT: Para automaticamente se áudio falhar
        if state.is_processing:
            handle_stop_processing()

# === STT ENGINES ===
# EN: Speech-to-Text engine implementations
# PT: Implementações de motores de Speech-to-Text

def process_transcription_result(text: str, is_partial: bool = False):
    """
    EN: Central function to process transcription results
    PT: Função central para processar resultados de transcrição
    """
    if not text or not text.strip():
        return
    
    text = text.strip()
    
    if is_partial:
        # EN: Update partial text display
        # PT: Atualiza exibição de texto parcial
        state._partial_text = text
        socketio.emit('update_stt', {'text': text, 'is_partial': True})
    else:
        # EN: Finalize transcription and trigger translation
        # PT: Finaliza transcrição e aciona tradução
        state.append_transcription(text)
        socketio.emit('update_stt', {'text': text, 'is_partial': False})
        
        # EN: Queue for real-time translation
        # PT: Enfileira para tradução em tempo real
        if not state.config.get('translationDisabled', False):
            try:
                state.translation_queue.put(text, block=False)
            except queue.Full:
                logger.warning("Translation queue full / Fila de tradução cheia")
        
        logger.debug(f"📝 Transcription added: {text[:50]}...")

def run_vosk_stt_engine(config: Dict[str, Any]):
    """
    EN: Vosk speech recognition engine
    PT: Motor de reconhecimento de fala Vosk
    """
    if not VOSK_AVAILABLE:
        safe_emit_notification('error', 'Vosk engine not available / Motor Vosk não disponível')
        return
    
    model_path = "vosk-model-en-us"
    if not os.path.exists(model_path):
        safe_emit_notification('error', f'Vosk model not found at {model_path}')
        return
    
    try:
        logger.info("🔄 Loading Vosk model...")
        model = Model(model_path)
        recognizer = KaldiRecognizer(model, 16000)
        recognizer.SetWords(True)
        logger.info("✅ Vosk engine started / Motor Vosk iniciado")
        
        while not state.stop_event.is_set():
            try:
                audio_data = state.audio_buffer.data.get(timeout=0.5)
                
                if recognizer.AcceptWaveform(audio_data):
                    # EN: Final result
                    # PT: Resultado final
                    result = json.loads(recognizer.Result())
                    if result.get('text'):
                        process_transcription_result(result['text'], is_partial=False)
                else:
                    # EN: Partial result
                    # PT: Resultado parcial
                    partial_result = json.loads(recognizer.PartialResult())
                    if partial_result.get('partial'):
                        process_transcription_result(partial_result['partial'], is_partial=True)
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Vosk processing error / Erro no processamento Vosk: {e}")
                break
        
        logger.info("🛑 Vosk engine stopped / Motor Vosk parado")
        
    except Exception as e:
        logger.error(f"Failed to initialize Vosk / Falha ao inicializar Vosk: {e}")
        safe_emit_notification('error', f'Vosk initialization failed / Falha na inicialização do Vosk: {e}')

def run_fastwhisper_stt_engine(config: Dict[str, Any]):
    """
    EN: Faster Whisper speech recognition engine
    PT: Motor de reconhecimento de fala Faster Whisper
    """
    if not FASTER_WHISPER_AVAILABLE:
        safe_emit_notification('error', 'FastWhisper engine not available / Motor FastWhisper não disponível')
        return
    
    # EN: Configuration parameters
    # PT: Parâmetros de configuração
    sample_rate = 16000
    chunk_duration = 3  # seconds / segundos
    chunk_samples = chunk_duration * sample_rate
    model_size = config.get("fastwhisperModel", "base")
    compute_type = config.get("computeType", "int8")
    
    try:
        logger.info(f"🔄 Loading FastWhisper model '{model_size}' with compute_type '{compute_type}'...")
        model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        logger.info("✅ FastWhisper engine started / Motor FastWhisper iniciado")
        
        while not state.stop_event.is_set():
            audio_data = bytearray()
            current_samples = 0
            
            # EN: Collect audio data for processing
            # PT: Coleta dados de áudio para processamento
            try:
                while current_samples < chunk_samples and not state.stop_event.is_set():
                    chunk = state.audio_buffer.data.get(timeout=0.1)
                    audio_data.extend(chunk)
                    current_samples += len(chunk) // 2  # 16-bit audio
            except queue.Empty:
                if not audio_data:
                    continue
            
            if audio_data:
                try:
                    # EN: Convert to numpy array for Whisper
                    # PT: Converte para array numpy para Whisper
                    numpy_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # EN: Transcribe audio
                    # PT: Transcreve áudio
                    segments, _ = model.transcribe(
                        numpy_data, 
                        beam_size=5, 
                        vad_filter=True, 
                        language="en"
                    )
                    
                    # EN: Process results
                    # PT: Processa resultados
                    full_text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
                    if full_text:
                        process_transcription_result(full_text, is_partial=False)
                        
                except Exception as e:
                    logger.error(f"FastWhisper transcription error / Erro de transcrição FastWhisper: {e}")
        
        logger.info("🛑 FastWhisper engine stopped / Motor FastWhisper parado")
        
    except Exception as e:
        logger.error(f"Failed to initialize FastWhisper / Falha ao inicializar FastWhisper: {e}")
        safe_emit_notification('error', f'FastWhisper initialization failed / Falha na inicialização do FastWhisper: {e}')

def run_google_stt_legacy_engine(config: Dict[str, Any]):
    """
    EN: Google Cloud Speech-to-Text (Legacy) engine
    PT: Motor Google Cloud Speech-to-Text (Legacy)
    """
    if not GOOGLE_AVAILABLE:
        safe_emit_notification('error', 'Google Cloud not available / Google Cloud não disponível')
        return
    
    try:
        credentials = get_google_credentials(config)
        client = speech.SpeechClient(credentials=credentials)
        logger.info("✅ Google STT Legacy engine started / Motor Google STT Legacy iniciado")
        
        # EN: Configure streaming recognition
        # PT: Configura reconhecimento em streaming
        streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
                enable_automatic_punctuation=True,
                model="latest_long"
            ),
            interim_results=True,
        )
        
        def audio_generator():
            """Generate audio requests / Gera requisições de áudio"""
            while not state.stop_event.is_set():
                try:
                    audio_chunk = state.audio_buffer.data.get(timeout=0.2)
                    yield speech.StreamingRecognizeRequest(audio_content=audio_chunk)
                except queue.Empty:
                    if state.stop_event.is_set():
                        break
        
        # EN: Process streaming responses
        # PT: Processa respostas em streaming
        responses = client.streaming_recognize(
            config=streaming_config, 
            requests=audio_generator()
        )
        
        for response in responses:
            if not response.results or not response.results[0].alternatives:
                continue
            
            result = response.results[0]
            transcript = result.alternatives[0].transcript
            
            if result.is_final:
                process_transcription_result(transcript, is_partial=False)
            else:
                process_transcription_result(transcript, is_partial=True)
        
        logger.info("🛑 Google STT Legacy engine stopped / Motor Google STT Legacy parado")
        
    except ValueError as e:
        safe_emit_notification('error', str(e))
    except Exception as e:
        logger.error(f"Google STT Legacy error / Erro Google STT Legacy: {e}")
        safe_emit_notification('error', 'Google STT error - check credentials / Erro Google STT - verifique credenciais')

def run_google_chirp_engine(config: Dict[str, Any]):
    """
    EN: Google Cloud Speech-to-Text v2 (Chirp) engine
    PT: Motor Google Cloud Speech-to-Text v2 (Chirp)
    """
    if not GOOGLE_AVAILABLE:
        safe_emit_notification('error', 'Google Cloud not available / Google Cloud não disponível')
        return
    
    try:
        # EN: Parse credentials and setup client
        # PT: Analisa credenciais e configura cliente
        creds_info = json.loads(config.get('googleCreds', '{}'))
        project_id = creds_info.get('project_id')
        if not project_id:
            raise ValueError("Project ID not found in credentials / Project ID não encontrado nas credenciais")
        
        credentials = service_account.Credentials.from_service_account_info(creds_info)
        region = config.get("googleRegion", "us-central1")
        endpoint = f"{region}-speech.googleapis.com"
        
        client = speech_v2.SpeechClient(
            credentials=credentials, 
            client_options={"api_endpoint": endpoint}
        )
        
        recognizer_path = f"projects/{project_id}/locations/{region}/recognizers/_"
        logger.info(f"✅ Google Chirp engine started in region {region} / Motor Google Chirp iniciado na região {region}")
        
        # EN: Configure recognition
        # PT: Configura reconhecimento
        recognition_config = speech_v2.RecognitionConfig(
            auto_decoding_config={},
            language_codes=["en-US"],
            model="chirp",
            features=speech_v2.RecognitionFeatures(
                enable_automatic_punctuation=True
            )
        )
        
        streaming_config = speech_v2.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=speech_v2.StreamingRecognitionFeatures(
                interim_results=True
            )
        )
        
        def request_generator():
            """Generate streaming requests / Gera requisições em streaming"""
            # EN: Initial config request
            # PT: Requisição inicial de configuração
            yield speech_v2.StreamingRecognizeRequest(
                recognizer=recognizer_path,
                streaming_config=streaming_config
            )
            
            # EN: Audio data requests
            # PT: Requisições de dados de áudio
            while not state.stop_event.is_set():
                try:
                    audio_chunk = state.audio_buffer.data.get(timeout=0.2)
                    yield speech_v2.StreamingRecognizeRequest(audio=audio_chunk)
                except queue.Empty:
                    if state.stop_event.is_set():
                        break
        
        # EN: Process responses
        # PT: Processa respostas
        responses = client.streaming_recognize(requests=request_generator())
        
        for response in responses:
            if not response.results or not response.results[0].alternatives:
                continue
            
            result = response.results[0]
            transcript = result.alternatives[0].transcript
            
            if result.is_final:
                process_transcription_result(transcript, is_partial=False)
            else:
                process_transcription_result(transcript, is_partial=True)
        
        logger.info("🛑 Google Chirp engine stopped / Motor Google Chirp parado")
        
    except (ValueError, json.JSONDecodeError) as e:
        safe_emit_notification('error', str(e))
    except Exception as e:
        logger.error(f"Google Chirp error / Erro Google Chirp: {e}")
        safe_emit_notification('error', 'Google Chirp error - check setup / Erro Google Chirp - verifique configuração')

# === TRANSLATION ENGINES ===
# EN: Translation engine implementations with real-time processing
# PT: Implementações de motores de tradução com processamento em tempo real

def translation_worker():
    """
    EN: Background worker for real-time translation processing
    PT: Worker em background para processamento de tradução em tempo real
    """
    logger.info("🌐 Translation worker started / Worker de tradução iniciado")
    
    while not state.stop_event.is_set():
        try:
            # EN: Get text to translate with timeout
            # PT: Obtém texto para traduzir com timeout
            text_to_translate = state.translation_queue.get(timeout=1.0)
            
            if not text_to_translate.strip():
                continue
            
            # EN: Get current translation engine
            # PT: Obtém motor de tradução atual
            engine_name = state.config.get('translationEngine', 'openai')
            translation_engine = TRANSLATION_ENGINES.get(engine_name)
            
            if translation_engine:
                try:
                    translation_engine(text_to_translate, state.config)
                except Exception as e:
                    logger.error(f"Translation error / Erro de tradução: {e}")
                    safe_emit_notification('warning', 'Translation failed for segment / Tradução falhou para segmento')
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Translation worker error / Erro no worker de tradução: {e}")
    
    logger.info("🛑 Translation worker stopped / Worker de tradução parado")

def openai_translate_text(text: str, config: Dict[str, Any]):
    """
    EN: Translate text using OpenAI API
    PT: Traduz texto usando API OpenAI
    """
    if not OPENAI_AVAILABLE:
        return
    
    api_key = config.get('apiKey')
    if not api_key:
        raise ValueError("OpenAI API key required / Chave API OpenAI necessária")
    
    try:
        client = OpenAI(api_key=api_key)
        target_lang = config.get('translationLang', 'pt-BR')
        
        # EN: Create translation prompt
        # PT: Cria prompt de tradução
        messages = [
            {
                "role": "system", 
                "content": f"You are a professional translator. Translate the English text to {target_lang}. Be accurate and maintain the original meaning. Return only the translation without explanations."
            },
            {
                "role": "user", 
                "content": f"Translate this text to {target_lang}:\n\n{text}"
            }
        ]
        
        # EN: Get translation
        # PT: Obtém tradução
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=len(text.split()) * 3 + 50
        )
        
        translated_text = response.choices[0].message.content.strip()
        
        # EN: Update translation buffer and emit to client
        # PT: Atualiza buffer de tradução e emite para cliente
        state.append_translation(translated_text)
        socketio.emit('update_translation', {'text': translated_text})
        
        logger.debug(f"🌐 OpenAI translation: {text[:30]}... -> {translated_text[:30]}...")
        
    except OpenAIError as e:
        raise ConnectionError(f"OpenAI API error / Erro API OpenAI: {e}")
    except Exception as e:
        logger.error(f"OpenAI translation error / Erro tradução OpenAI: {e}")

def google_translate_text(text: str, config: Dict[str, Any]):
    """
    EN: Translate text using Google Translate API
    PT: Traduz texto usando API Google Translate
    """
    if not GOOGLE_AVAILABLE:
        return
    
    try:
        credentials = get_google_credentials(config)
        translate_client = translate.Client(credentials=credentials)
        target_lang = config.get('translationLang', 'pt-BR')
        
        # EN: Perform translation
        # PT: Realiza tradução
        result = translate_client.translate(text, target_language=target_lang)
        translated_text = result['translatedText']
        
        # EN: Update translation buffer and emit to client
        # PT: Atualiza buffer de tradução e emite para cliente
        state.append_translation(translated_text)
        socketio.emit('update_translation', {'text': translated_text})
        
        logger.debug(f"🌐 Google translation: {text[:30]}... -> {translated_text[:30]}...")
        
    except ValueError as e:
        safe_emit_notification('error', str(e))
    except Exception as e:
        logger.error(f"Google Translate error / Erro Google Translate: {e}")
        raise ConnectionError(f"Google Translate API error / Erro API Google Translate: {e}")

# === AI SUGGESTIONS ===
# EN: AI-powered response suggestions based on transcription
# PT: Sugestões de resposta com IA baseadas na transcrição

def generate_ai_suggestion(config: Dict[str, Any], suggestion_index: int):
    """
    EN: Generate AI response suggestion based on current transcription
    PT: Gera sugestão de resposta com IA baseada na transcrição atual
    """
    if not OPENAI_AVAILABLE:
        safe_emit_notification('warning', 'OpenAI not available for suggestions / OpenAI não disponível para sugestões')
        return
    
    api_key = config.get('apiKey')
    if not api_key:
        safe_emit_notification('warning', 'OpenAI API key needed for suggestions / Chave API OpenAI necessária para sugestões')
        return
    
    # EN: Get context from transcription buffer
    # PT: Obtém contexto do buffer de transcrição
    context = state.transcription_buffer
    if not context.strip():
        socketio.emit('update_one_suggestion', {
            'suggestion': 'Start listening to get suggestions / Inicie a escuta para obter sugestões',
            'index': suggestion_index
        })
        return
    
    try:
        client = OpenAI(api_key=api_key)
        
        # EN: Create context-aware prompt for suggestions
        # PT: Cria prompt consciente de contexto para sugestões
        prompt = f"""
        Based on this conversation transcript: "{context[-800:]}"
        
        Generate a helpful, professional response suggestion in English that:
        1. Is relevant to the conversation context
        2. Adds value to the discussion
        3. Is appropriate for a business/professional setting
        4. Is concise (max 20 words)
        5. Could realistically be said by a participant
        
        Return as JSON: {{"suggestion": "your suggestion here"}}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,  # EN: Higher creativity for diverse suggestions / PT: Maior criatividade para sugestões diversas
            max_tokens=100,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        suggestion = result.get('suggestion', 'No suggestion available')
        
        # EN: Emit suggestion to client
        # PT: Emite sugestão para cliente
        socketio.emit('update_one_suggestion', {
            'suggestion': suggestion,
            'index': suggestion_index
        })
        
        logger.debug(f"💡 Generated suggestion {suggestion_index}: {suggestion}")
        
    except Exception as e:
        logger.error(f"AI suggestion error / Erro sugestão IA: {e}")
        socketio.emit('update_one_suggestion', {
            'suggestion': 'Error generating suggestion / Erro ao gerar sugestão',
            'index': suggestion_index
        })

# === ENGINE MAPPINGS ===
# EN: Map engine names to their implementation functions
# PT: Mapeia nomes de motores para suas funções de implementação

STT_ENGINES: Dict[str, Callable] = {
    'vosk': run_vosk_stt_engine,
    'fast_whisper': run_fastwhisper_stt_engine,
    'google_stt_legacy': run_google_stt_legacy_engine,
    'google_chirp': run_google_chirp_engine
}

TRANSLATION_ENGINES: Dict[str, Callable] = {
    'openai': openai_translate_text,
    'google_translate': google_translate_text
}

# === FLASK ROUTES ===
# EN: Web application routes
# PT: Rotas da aplicação web

@app.route('/')
def index():
    """
    EN: Main application page
    PT: Página principal da aplicação
    """
    return render_template('index.html')

# === SOCKETIO EVENT HANDLERS ===
# EN: Handle WebSocket events from the client
# PT: Gerencia eventos WebSocket do cliente

@socketio.on('connect')
def handle_client_connect():
    """
    EN: Handle new client connection
    PT: Gerencia nova conexão de cliente
    """
    client_id = request.sid
    logger.info(f"👋 Client connected: {client_id}")
    
    try:
        # EN: Send available audio devices
        # PT: Envia dispositivos de áudio disponíveis
        devices = sd.query_devices()
        input_devices = [
            {'id': i, 'name': device['name']} 
            for i, device in enumerate(devices) 
            if device['max_input_channels'] > 0
        ]
        socketio.emit('audio_devices_list', input_devices, to=client_id)
        
    except Exception as e:
        logger.error(f"Error listing audio devices / Erro ao listar dispositivos de áudio: {e}")
    
    # EN: Send current configuration
    # PT: Envia configuração atual
    socketio.emit('config_loaded', state.config, to=client_id)
    
    # EN: Update API status
    # PT: Atualiza status da API
    safe_emit_api_status('inactive', 'Ready / Pronto')

@socketio.on('disconnect')
def handle_client_disconnect():
    """
    EN: Handle client disconnection
    PT: Gerencia desconexão de cliente
    """
    client_id = request.sid
    logger.info(f"👋 Client disconnected: {client_id}")

@socketio.on('start_processing')
def handle_start_processing(config: Dict[str, Any]):
    """
    EN: Start audio processing with given configuration
    PT: Inicia processamento de áudio com configuração dada
    """
    if state.is_processing:
        logger.warning("⚠️ Processing already active / Processamento já ativo")
        return
    
    logger.info("🚀 Starting processing / Iniciando processamento")
    
    # EN: Update global configuration
    # PT: Atualiza configuração global
    state.config = config
    
    # EN: Validate STT engine availability
    # PT: Valida disponibilidade do motor STT
    stt_engine_name = config.get('sttEngine', 'vosk')
    stt_engine_func = STT_ENGINES.get(stt_engine_name)
    
    if not stt_engine_func:
        safe_emit_notification('error', f'STT engine "{stt_engine_name}" not available')
        return
    
    # EN: Reset state for new session
    # PT: Reseta estado para nova sessão
    state.stop_event.clear()
    state.clear_buffers()
    state.audio_buffer.clear()
    state.is_processing = True
    
    # EN: Start audio capture thread
    # PT: Inicia thread de captura de áudio
    audio_thread = threading.Thread(
        target=audio_capture_worker,
        args=(config.get('audioDevice', 0),),
        daemon=True
    )
    audio_thread.start()
    state.processing_threads.append(audio_thread)
    
    # EN: Start STT engine thread
    # PT: Inicia thread do motor STT
    stt_thread = threading.Thread(
        target=stt_engine_func,
        args=(config,),
        daemon=True
    )
    stt_thread.start()
    state.processing_threads.append(stt_thread)
    
    # EN: Start translation worker if enabled
    # PT: Inicia worker de tradução se habilitado
    if not config.get('translationDisabled', False):
        state.translation_thread = threading.Thread(
            target=translation_worker,
            daemon=True
        )
        state.translation_thread.start()
        state.processing_threads.append(state.translation_thread)
    
    # EN: Notify client
    # PT: Notifica cliente
    emit('processing_started')
    safe_emit_api_status('active', 'Listening... / Ouvindo...')
    
    logger.info("✅ Processing started successfully / Processamento iniciado com sucesso")

@socketio.on('stop_processing')
def handle_stop_processing():
    """
    EN: Stop all audio processing
    PT: Para todo o processamento de áudio
    """
    if not state.is_processing:
        logger.warning("⚠️ No active processing to stop / Nenhum processamento ativo para parar")
        return
    
    logger.info("🛑 Stopping processing / Parando processamento")
    
    # EN: Stop all processing
    # PT: Para todo o processamento
    state.stop_all_processing()
    
    # EN: Notify client
    # PT: Notifica cliente
    emit('processing_stopped')
    safe_emit_api_status('inactive', 'Stopped / Parado')
    
    logger.info("✅ Processing stopped successfully / Processamento parado com sucesso")

@socketio.on('force_translate')
def handle_force_translate(config: Dict[str, Any]):
    """
    EN: Force translation of entire transcription buffer
    PT: Força tradução de todo o buffer de transcrição
    """
    # EN: Get current transcription buffer
    # PT: Obtém buffer de transcrição atual
    text_to_translate = state.transcription_buffer
    
    if not text_to_translate.strip():
        safe_emit_notification('info', 'Nothing to translate / Nada para traduzir')
        return
    
    logger.info(f"🔄 Force translating {len(text_to_translate)} characters")
    
    # EN: Get translation engine
    # PT: Obtém motor de tradução
    engine_name = config.get('translationEngine', 'openai')
    translation_engine = TRANSLATION_ENGINES.get(engine_name)
    
    if not translation_engine:
        safe_emit_notification('error', f'Translation engine "{engine_name}" not available')
        return
    
    # EN: Clear previous translation
    # PT: Limpa tradução anterior
    state.translation_buffer = ""
    socketio.emit('clear_translation')
    
    try:
        safe_emit_notification('info', 'Translating full text... / Traduzindo texto completo...')
        
        # EN: Split into chunks to avoid API limits
        # PT: Divide em chunks para evitar limites da API
        chunk_size = 400
        text_chunks = [
            text_to_translate[i:i + chunk_size] 
            for i in range(0, len(text_to_translate), chunk_size)
        ]
        
        # EN: Process each chunk
        # PT: Processa cada chunk
        for i, chunk in enumerate(text_chunks):
            try:
                translation_engine(chunk, config)
                # EN: Small delay to avoid rate limiting
                # PT: Pequeno atraso para evitar limitação de taxa
                if i < len(text_chunks) - 1:
                    time.sleep(0.3)
            except Exception as e:
                logger.error(f"Chunk translation error / Erro tradução chunk: {e}")
                safe_emit_notification('warning', f'Translation error in chunk {i+1}')
        
        safe_emit_notification('success', 'Translation completed / Tradução concluída')
        
    except Exception as e:
        logger.error(f"Force translation error / Erro tradução forçada: {e}")
        safe_emit_notification('error', f'Translation failed / Tradução falhou: {e}')

@socketio.on('regenerate_suggestion')
def handle_regenerate_suggestion(data: Dict[str, Any]):
    """
    EN: Generate new AI suggestion for given index
    PT: Gera nova sugestão IA para índice dado
    """
    suggestion_index = data.get('index')
    config = data.get('config', {})
    
    if suggestion_index is not None:
        logger.info(f"💡 Regenerating suggestion {suggestion_index}")
        
        # EN: Run in background thread to avoid blocking
        # PT: Executa em thread de fundo para evitar bloqueio
        suggestion_thread = threading.Thread(
            target=generate_ai_suggestion,
            args=(config, suggestion_index),
            daemon=True
        )
        suggestion_thread.start()

@socketio.on('save_settings')
def handle_save_settings(new_config: Dict[str, Any]):
    """
    EN: Save new configuration settings
    PT: Salva novas configurações
    """
    logger.info("💾 Saving new settings / Salvando novas configurações")
    state.config = new_config
    save_application_config(new_config)

# === APPLICATION INITIALIZATION ===
# EN: Initialize the application
# PT: Inicializa a aplicação

def initialize_application():
    """
    EN: Initialize application state and configuration
    PT: Inicializa estado e configuração da aplicação
    """
    logger.info("🚀 Initializing Transwer application / Inicializando aplicação Transwer")
    
    # EN: Load configuration
    # PT: Carrega configuração
    state.config = load_application_config()
    
    # EN: Log available engines
    # PT: Registra motores disponíveis
    available_stt = [name for name, func in STT_ENGINES.items() if func]
    available_translation = [name for name, func in TRANSLATION_ENGINES.items() if func]
    
    logger.info(f"🎙️ Available STT engines: {available_stt}")
    logger.info(f"🌐 Available translation engines: {available_translation}")
    
    logger.info("✅ Transwer application ready / Aplicação Transwer pronta")

# === MAIN ENTRY POINT ===
# EN: Application entry point
# PT: Ponto de entrada da aplicação

if __name__ == '__main__':
    try:
        # EN: Initialize application
        # PT: Inicializa aplicação
        initialize_application()
        
        # EN: Start the server
        # PT: Inicia o servidor
        logger.info("🌐 Starting Transwer server on http://localhost:5000")
        logger.info("🌐 Iniciando servidor Transwer em http://localhost:5000")
        
        socketio.run(
            app, 
            debug=False,  # EN: Set to False for production / PT: Definir como False para produção
            host='0.0.0.0',
            port=5000,
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Transwer / Encerrando Transwer")
        if state.is_processing:
            state.stop_all_processing()
    except Exception as e:
        logger.error(f"❌ Application error / Erro da aplicação: {e}")
        raise
    finally:
        logger.info("👋 Transwer shutdown complete / Encerramento do Transwer completo")
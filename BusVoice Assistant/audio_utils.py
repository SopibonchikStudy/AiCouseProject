import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import os
import re
import requests
import urllib3
import uuid
import io

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==================== ТОКЕН ====================

def _get_salute_token() -> str:
    """Получает access_token для SaluteSpeech API."""
    credentials_base64 = os.getenv("SALUTESPEECH_CREDENTIALS")
    if not credentials_base64:
        raise Exception("SALUTESPEECH_CREDENTIALS не найден в .env")
    
    response = requests.post(
        "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        headers={
            "Authorization": f"Basic {credentials_base64}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        },
        data={"scope": "SALUTE_SPEECH_PERS"},
        verify=False
    )
    
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Ошибка токена: {response.status_code}")

# ==================== РАСПОЗНАВАНИЕ ====================

def _convert_audio_to_wav_bytes(input_path: str) -> bytes:
    """Конвертирует аудио в WAV и возвращает байты."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()

def speech_to_text_salute(audio_file: str) -> str:
    """Распознавание речи через SaluteSpeech."""
    access_token = _get_salute_token()
    wav_bytes = _convert_audio_to_wav_bytes(audio_file)
    
    response = requests.post(
        "https://smartspeech.sber.ru/rest/v1/speech:recognize",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "audio/x-pcm;bit=16;rate=16000"
        },
        data=wav_bytes[44:],
        verify=False
    )
    
    if response.status_code == 200:
        result = response.json()
        if "result" in result and len(result["result"]) > 0:
            text = result["result"][0]
            if text and text.strip():
                return text.strip()
        return "[Ошибка: пустой результат распознавания]"
    else:
        raise Exception(f"Ошибка распознавания: {response.status_code}")

def speech_to_text_google(audio_file: str) -> str:
    """Запасной вариант: Google Speech Recognition."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
    except ValueError:
        audio_seg = AudioSegment.from_file(audio_file)
        audio_seg = audio_seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_seg.export(tmp.name, format="wav")
        with sr.AudioFile(tmp.name) as source:
            audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language="ru-RU")
    except sr.UnknownValueError:
        return "[Ошибка: не удалось распознать речь]"
    except sr.RequestError:
        return "[Ошибка: нет подключения к Google Speech]"

def speech_to_text(audio_file: str) -> str:
    """Распознавание речи. SaluteSpeech → Google (fallback)."""
    try:
        return speech_to_text_salute(audio_file)
    except Exception as e:
        print(f"SaluteSpeech распознавание: {e}, использую Google")
        return speech_to_text_google(audio_file)

# ==================== СИНТЕЗ ====================

def clean_text_for_speech(text: str) -> str:
    """Очищает текст от форматирования."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    for char in ['*', '#', '`', '→', '←', '↑', '↓', '⇒', '•', '·', '►', '◄', '▶']:
        text = text.replace(char, ' ')
    text = text.replace('…', '...')
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def text_to_speech_salute(text: str) -> str:
    """Синтез речи через SaluteSpeech."""
    access_token = _get_salute_token()
    response = requests.post(
        "https://smartspeech.sber.ru/rest/v1/text:synthesize",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/text"
        },
        params={"format": "wav16", "voice": "May_24000"},
        data=text.encode(),
        verify=False
    )
    if response.status_code == 200:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(response.content)
        return tmp.name
    else:
        raise Exception(f"Ошибка синтеза: {response.status_code}")

def text_to_speech_google(text: str) -> str:
    """Запасной вариант: gTTS."""
    tts = gTTS(text=text, lang="ru")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

def text_to_speech(text: str):
    """Синтез речи. SaluteSpeech → gTTS (fallback)."""
    clean_text = clean_text_for_speech(text)
    if not clean_text or len(clean_text) < 2:
        clean_text = "Нет текста для озвучивания."
    try:
        return text_to_speech_salute(clean_text)
    except Exception as e:
        print(f"SaluteSpeech синтез: {e}, использую gTTS")
        return text_to_speech_google(clean_text)
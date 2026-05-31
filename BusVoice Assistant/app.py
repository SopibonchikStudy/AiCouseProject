import streamlit as st
import os
import re
import time
import tempfile
import urllib3
from dotenv import load_dotenv
import pandas as pd

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

from agents import extract_chain, recommend_chain, invoke_mechanic_agent, invoke_dispatcher_agent
from audio_utils import speech_to_text, text_to_speech

st.set_page_config(page_title="BusVoice Assistant", layout="wide")
st.title("🚌 Бортовой ассистент водителя автобуса")

# Инициализация

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_input" not in st.session_state:
    st.session_state.user_input = None

if "last_audio_name" not in st.session_state:
    st.session_state.last_audio_name = None

# Сайдбар

with st.sidebar:
    st.header("🎛️ Элементы управления")
    
    # Загрузка CSV
    uploaded_schedule = st.file_uploader(
        "📅 Загрузить расписание (CSV)", 
        type="csv", 
        key="schedule_uploader"
    )
    if uploaded_schedule is not None:
        pd.read_csv(uploaded_schedule).to_csv("data/schedule.csv", index=False)
        st.success("Расписание загружено")
    
    st.header("⚡ Быстрые сообщения")
    cols = st.columns(2)
    with cols[0]:
        if st.button("🛑 Отказ тормозов", use_container_width=True):
            st.session_state.user_input = "Отказали тормоза, педаль проваливается"
        if st.button("💡 Фары", use_container_width=True):
            st.session_state.user_input = "Не работают фары, на улице темнеет"
    with cols[1]:
        if st.button("🌡️ Перегрев", use_container_width=True):
            st.session_state.user_input = "Двигатель перегревается, температура растёт"
        if st.button("🚪 Двери", use_container_width=True):
            st.session_state.user_input = "Заклинило заднюю дверь, не открывается"
    
    st.header("🎤 Голосовой ввод")
    audio_file = st.file_uploader(
        "Загрузите аудиозапись (.wav)", 
        type=["wav"], 
        key="audio_uploader"
    )
    
    # Обрабатываем аудио только если это новый файл
    if audio_file is not None:
        current_name = audio_file.name
        if current_name != st.session_state.last_audio_name:
            st.session_state.last_audio_name = current_name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                recognized = speech_to_text(tmp.name)
            st.info(f"🎤 Распознано: {recognized}")
            st.session_state.user_input = recognized
            st.rerun()
    else:
        st.session_state.last_audio_name = None

# Ввод текста

text_input = st.chat_input("Опишите неисправность...")
if text_input:
    st.session_state.user_input = text_input

# Основная часть

# Показываем историю чата
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Работа агентов

if st.session_state.user_input:
    user_input = st.session_state.user_input
    st.session_state.user_input = None  # Очищаем сразу
    
    # Добавляем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Генерируем ответ
    with st.chat_message("assistant"):
        with st.spinner("Анализирую..."):
            try:
                # Шаг 1: Извлечение симптомов
                extracted = extract_chain.invoke({"message": user_input})
                time.sleep(1)
                
                # Шаг 2: Агент-Механик
                mechanic_result = invoke_mechanic_agent(
                    f"Классифицируй неисправность и найди инструкцию. Сообщение водителя: {user_input}. Извлечённая информация: {extracted}"
                )
                time.sleep(2)

                # Шаг 3: Агент-Диспетчер
                dispatcher_result = invoke_dispatcher_agent(
                    f"Найди свободные автобусы для замены. Неисправность: {extracted}"
                )
                time.sleep(1)


                # Шаг 4: Финальная рекомендация
                final = recommend_chain.invoke({
                    "user_message": user_input,
                    "extracted_info": extracted,
                    "mechanic_result": mechanic_result,
                    "dispatcher_result": dispatcher_result
                })
                
                st.markdown(final)
                
                # Озвучка
                try:
                    audio_path = text_to_speech(final)
                    st.audio(audio_path, format="audio/wav")
                except Exception as e:
                    st.warning(f"Озвучка недоступна: {e}")
                
                st.session_state.messages.append({"role": "assistant", "content": final})
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    st.error("⚠️ Превышен лимит запросов. Подождите 30 секунд.")
                else:
                    st.error(f"❌ Ошибка: {error_msg}")
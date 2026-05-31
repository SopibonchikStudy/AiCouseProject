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

# ==================== ИНИЦИАЛИЗАЦИЯ ====================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_input" not in st.session_state:
    st.session_state.user_input = None

if "last_audio_name" not in st.session_state:
    st.session_state.last_audio_name = None

# ==================== БОКОВАЯ ПАНЕЛЬ ====================

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

# ==================== ТЕКСТОВЫЙ ВВОД ====================

text_input = st.chat_input("Опишите неисправность...")
if text_input:
    st.session_state.user_input = text_input

# ==================== ОСНОВНАЯ ОБЛАСТЬ ====================

# Показываем историю чата
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==================== ОБРАБОТКА СООБЩЕНИЯ ====================

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

                # =============== ОТЛАДОЧНЫЙ ВЫВОД ===============
                from ml_model import predict_severity
                import joblib
                
                ml_severity = predict_severity(user_input)
                model_info = "загружена из model.pkl" if os.path.exists("model.pkl") else "обучена заново"
                
                df_faults = pd.read_csv("data/faults.csv")
                actual_samples = len(df_faults)

                st.info(f"""
                🧪 **ОТЛАДКА: Результат ML-модели**
                - Входной текст: "{user_input}"
                - Предсказанный класс: **{ml_severity.upper()}**
                - Модель: LogisticRegression + TfidfVectorizer ({model_info})
                - Размер обучающей выборки: {actual_samples} примеров (data/faults.csv)
                - Классов: {df_faults['severity'].nunique()} ({', '.join(df_faults['severity'].value_counts().index.tolist())})
                - Пайплайн: TF-IDF векторизация → классификация на 3 класса
                """)
                
                with st.expander("🔍 Подробности работы пайплайна"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Извлечённая информация:**")
                        st.text(extracted)
                        st.write("**Результат агента-механика:**")
                        st.text(mechanic_result)
                    with col2:
                        st.write("**Результат агента-диспетчера:**")
                        st.text(dispatcher_result)
                        if ml_severity.lower() in mechanic_result.lower():
                            st.success("✅ ML-модель используется: класс совпадает с результатом механика")
                        else:
                            st.warning("⚠️ Класс из ML-модели не найден в ответе механика")
                # =============== КОНЕЦ ОТЛАДКИ ===============

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
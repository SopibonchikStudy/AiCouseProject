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

from agents import extract_chain, recommend_chain, mechanic_executor, dispatcher_executor
from audio_utils import speech_to_text, text_to_speech

st.set_page_config(page_title="BusVoice Assistant", layout="wide")
st.title("🚌 Бортовой ассистент водителя автобуса")

# ==================== ИНИЦИАЛИЗАЦИЯ ====================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_input" not in st.session_state:
    st.session_state.user_input = None

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
    if st.button("🛑 Отказ тормозов"):
        st.session_state.user_input = "Отказали тормоза, педаль проваливается"
    if st.button("🌡️ Перегрев"):
        st.session_state.user_input = "Двигатель перегревается, температура растёт"
    if st.button("💡 Фары"):
        st.session_state.user_input = "Не работают фары, на улице темнеет"
    if st.button("🚪 Двери"):
        st.session_state.user_input = "Заклинило заднюю дверь, не открывается"
    
    st.header("🎤 Голосовой ввод")
    audio_file = st.file_uploader(
        "Загрузите аудиозапись (.wav)", 
        type=["wav"], 
        key="audio_uploader"
    )
    
    # Обрабатываем аудио сразу при загрузке
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            recognized = speech_to_text(tmp.name)
        st.info(f"🎤 Распознано: {recognized}")
        st.session_state.user_input = recognized

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
                
                # Шаг 2: Агент-Механик (настоящий агент!)
                mechanic_result = mechanic_executor.invoke({
                    "input": f"Классифицируй неисправность и найди инструкцию. Сообщение водителя: {user_input}. Извлечённая информация: {extracted}"
                })["output"]
                time.sleep(2)

                # Шаг 3: Агент-Диспетчер (настоящий агент!)
                dispatcher_result = dispatcher_executor.invoke({
                    "input": f"Найди свободные автобусы для замены. Выбери наиболее подходящий (тот же маршрут или ближайший по времени). Неисправность: {extracted}"
                })["output"]
                time.sleep(1)
                # =============== ОТЛАДОЧНЫЙ ВЫВОД ===============
                # Прямой вызов ML-модели для демонстрации
                from ml_model import predict_severity
                ml_severity = predict_severity(user_input)
                
                # Загрузка модели для получения информации о ней
                import joblib
                import os
                model_info = "загружена из model.pkl" if os.path.exists("model.pkl") else "обучена заново"
                
                # Показываем результат ML-модели
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
                
                # Показываем промежуточные результаты цепочек
                with st.expander("🔍 Подробности работы пайплайна"):
                    st.write("**Извлечённая информация:**")
                    st.text(extracted)
                    st.write("**Результат агента-механика:**")
                    st.text(mechanic_result)
                    st.write("**Результат агента-диспетчера:**")
                    st.text(dispatcher_result)
                    
                    # Проверяем, что ML-модель реально используется
                    if ml_severity.lower() in mechanic_result.lower():
                        st.success("✅ ML-модель действительно используется: класс из модели совпадает с результатом механика")
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
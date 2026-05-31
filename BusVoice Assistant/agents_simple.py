from langchain_gigachat.chat_models import GigaChat
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import pandas as pd
import os
from ml_model import predict_severity
from rag import query_manual

# Инициализация LLM
llm = GigaChat(
    credentials=os.getenv("GIGACHAT_CREDENTIALS"),
    verify_ssl_certs=False,
    temperature=0.1
)

# Хранилище для истории диалогов (в памяти)
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Цепочка 1: Извлечение симптомов
extract_prompt = PromptTemplate(
    input_variables=["message"],
    template="""
Извлеки из сообщения водителя структурированную информацию:
Сообщение: {message}

Верни строго в формате:
Тип неисправности: <тип>
Симптомы: <описание>
Условия: <светлое/тёмное время суток, погода>
"""
)
extract_chain = extract_prompt | llm | StrOutputParser()

# Функции для прямого вызова (без агентов)
def get_mechanic_info(user_input: str, extracted: str) -> str:
    """Прямой вызов: классификация + поиск в инструкции."""
    severity = predict_severity(user_input)
    manual_info = query_manual(user_input)
    return f"Класс серьёзности: {severity}\nИнструкция: {manual_info}"

def get_dispatcher_info() -> str:
    """Прямой вызов: поиск свободных автобусов."""
    df = pd.read_csv("data/schedule.csv")
    available = df[df["status"] == "в парке"]
    if available.empty:
        return "Нет свободных автобусов. Все на линии."
    result = "Доступные автобусы для замены:\n"
    for _, row in available.iterrows():
        result += f"- Маршрут {row['route_id']} ({row['route_name']}), автобус {row['bus_id']}, водитель {row['driver']}, выезд в {row['departure_time']}\n"
    return result

# Цепочка 2: Финальная рекомендация
recommend_prompt = PromptTemplate(
    input_variables=["extracted_info", "mechanic_result", "dispatcher_result"],
    template="""
Ты — диспетчер автобусного парка. Дай водителю чёткий план действий.

Информация о неисправности: {extracted_info}
Заключение механика: {mechanic_result}
Информация о доступных автобусах: {dispatcher_result}

ПРАВИЛА:
- ОБЯЗАТЕЛЬНО укажи класс серьёзности неисправности из заключения механика (критическая, средняя или низкая).
- ОБЯЗАТЕЛЬНО перечисли ВСЕ доступные автобусы из информации диспетчера.
- Используй ТОЛЬКО ту инструкцию, которую дал механик. Не добавляй ничего от себя.
- Если класс "критическая" — немедленная остановка и замена автобуса.
- Если класс "средняя" — доехать до конечной, если безопасно, затем замена.
- Если класс "низкая" — продолжить рейс, отметить необходимость ремонта.

ОТВЕТ ДОЛЖЕН БЫТЬ В ТАКОМ ФОРМАТЕ:

Класс серьёзности: [критическая/средняя/низкая]

[План действий]

Доступные автобусы для замены:
[перечисли все автобусы из информации диспетчера]

ВАЖНО: Ответ должен быть ТОЛЬКО текстом, БЕЗ форматирования (без **, *, #, дефисов).
Пиши простыми предложениями.
"""
)
recommend_chain = recommend_prompt | llm | StrOutputParser()
from langchain_gigachat.chat_models import GigaChat
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import tool
from langgraph.prebuilt import create_react_agent
import pandas as pd
import os
from dotenv import load_dotenv
from ml_model import predict_severity
from rag import query_manual

load_dotenv()

llm = GigaChat(
    credentials=os.getenv("GIGACHAT_CREDENTIALS"),
    verify_ssl_certs=False,
    temperature=0
)

@tool
def classify_fault(text: str) -> str:
    """Классифицирует серьёзность неисправности по описанию. Возвращает: критическая, средняя или низкая."""
    return predict_severity(text)

@tool
def search_manual(query: str) -> str:
    """Ищет инструкцию по ремонту в базе знаний. Возвращает текст инструкции."""
    return query_manual(query)

@tool
def check_schedule(dummy: str = "") -> str:
    """Проверяет расписание и находит свободные автобусы для замены. Возвращает список доступных автобусов."""
    df = pd.read_csv("data/schedule.csv")
    available = df[df["status"] == "в парке"]
    if available.empty:
        return "Нет свободных автобусов. Все на линии."
    result = "Доступные автобусы для замены:\n"
    for _, row in available.iterrows():
        result += (
            f"Маршрут {row['route_id']} ({row['route_name']}), "
            f"автобус {row['bus_id']}, водитель {row['driver']}, "
            f"выезд в {row['departure_time']}\n"
        )
    return result

mechanic_tools = [classify_fault, search_manual]
dispatcher_tools = [check_schedule]

mechanic_agent = create_react_agent(model=llm, tools=mechanic_tools)
dispatcher_agent = create_react_agent(model=llm, tools=dispatcher_tools)

MECHANIC_SYSTEM = """Ты — механик автобусного парка. Твоя задача:

1. Вызови classify_fault для сообщения водителя
2. Вызови search_manual с тем же текстом
3. На основе инструкции из search_manual составь КОРОТКИЙ план для водителя (2-3 предложения, только конкретные действия)

Верни результат СТРОГО по шаблону:

Класс серьёзности: [результат classify_fault]
Инструкция: [сокращённый план для водителя на основе search_manual]

ПРАВИЛА:
- Пиши коротко: 2-3 предложения
- Только конкретные действия: "остановитесь", "проверьте", "вызовите"
- Не пиши "визуально осмотрите все топливопроводы на предмет..."
- Пиши: "Проверьте, нет ли подтёков топлива под автобусом. Если есть — остановитесь и вызовите техпомощь"
- Если есть риск — укажи: "Немедленно прекратить движение"
- Если безопасно — укажи: "Можно продолжить рейс, проверить на конечной"
- НЕ копируй методичку слово в слово. Сделай понятную инструкцию."""

DISPATCHER_SYSTEM = """Ты — диспетчер автобусного парка. Твоя задача — найти замену для неисправного автобуса.

ПОРЯДОК ДЕЙСТВИЙ:
1. ОБЯЗАТЕЛЬНО вызови инструмент check_schedule с параметром ""
2. Проанализируй список свободных автобусов
3. Выбери ОДИН наиболее подходящий автобус для замены и объясни почему именно он

ПРАВИЛА ВЫБОРА:
- Если неисправность критическая — выбери автобус с ближайшим временем выезда
- Если среди свободных есть автобус того же маршрута — выбери его
- Если все свободные на разных маршрутах — выбери с ближайшим временем
- Если check_schedule вернул "Нет свободных автобусов" — сообщи об этом

ФОРМАТ ОТВЕТА:
Рекомендуемая замена: [маршрут, автобус, водитель, время выезда]
Причина выбора: [одно предложение — почему именно этот автобус]
Все свободные автобусы: [краткий перечень остальных, если есть]"""

def invoke_mechanic_agent(input_text: str) -> str:
    result = mechanic_agent.invoke({
        "messages": [
            ("system", MECHANIC_SYSTEM),
            ("user", input_text)
        ]
    })
    return result["messages"][-1].content

def invoke_dispatcher_agent(input_text: str) -> str:
    result = dispatcher_agent.invoke({
        "messages": [
            ("system", DISPATCHER_SYSTEM),
            ("user", input_text)
        ]
    })
    return result["messages"][-1].content

extract_prompt = PromptTemplate(
    input_variables=["message"],
    template="""Извлеки из сообщения водителя структурированную информацию:
Сообщение: {message}

Верни строго в формате:
Тип неисправности: <тип>
Симптомы: <описание>
Условия: <светлое/тёмное время суток, погода>"""
)
extract_chain = extract_prompt | llm | StrOutputParser()

recommend_prompt = PromptTemplate(
    input_variables=["user_message", "extracted_info", "mechanic_result", "dispatcher_result"],
    template="""Ты — главный диспетчер автобусного парка. Сформируй ответ ВОДИТЕЛЮ строго по шаблону.

ДАННЫЕ:
Сообщение водителя: {user_message}
Извлечённая информация: {extracted_info}
Заключение механика: {mechanic_result}
Доступные автобусы: {dispatcher_result}

ШАБЛОН ОТВЕТА (заполни его, не меняя структуру):

Класс серьёзности: [выпиши класс из заключения механика]

План действий:
[выпиши инструкцию из заключения механика слово в слово]

Замена автобуса:
[выпиши сюда ВСЁ, что перечислено в "Доступные автобусы", слово в слово. Если там "Нет свободных автобусов" — напиши эту фразу]

НЕ ДОБАВЛЯЙ ничего от себя. НЕ МЕНЯЙ формулировки из заключения механика и списка автобусов. НЕ ИГНОРИРУЙ раздел "Замена автобуса". Ответ без форматирования."""
)
recommend_chain = recommend_prompt | llm | StrOutputParser()
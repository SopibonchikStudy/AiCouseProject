from langchain_gigachat.chat_models import GigaChat
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import tool
from langchain.agents import create_react_agent, AgentExecutor
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

# ==================== ИНСТРУМЕНТЫ ====================

@tool
def classify_fault(text: str) -> str:
    """
    Классифицирует серьёзность неисправности по описанию.
    Вход: текст неисправности (например, "отказали тормоза").
    Возвращает: критическая, средняя или низкая.
    """
    return predict_severity(text)

@tool
def search_manual(query: str) -> str:
    """
    Ищет инструкцию по ремонту в базе знаний.
    Вход: запрос (например, "тормозная система", "двигатель").
    Возвращает: текст инструкции из руководства.
    """
    return query_manual(query)

@tool
def check_schedule(dummy: str = "") -> str:
    """
    Проверяет расписание и находит свободные автобусы для замены.
    Не требует входных данных.
    Возвращает: список доступных автобусов с номерами, маршрутами и временем выезда.
    """
    df = pd.read_csv("data/schedule.csv")
    available = df[df["status"] == "в парке"]
    if available.empty:
        return "Нет свободных автобусов. Все на линии."
    result = ""
    for _, row in available.iterrows():
        result += (
            f"Маршрут {row['route_id']} ({row['route_name']}), "
            f"автобус {row['bus_id']}, водитель {row['driver']}, "
            f"выезд в {row['departure_time']}\n"
        )
    return result

# Наборы инструментов
mechanic_tools = [classify_fault, search_manual]
dispatcher_tools = [check_schedule]

# ==================== АГЕНТ-МЕХАНИК ====================

# Используем стандартный ReAct-промпт из хаба
from langchain import hub
react_prompt = hub.pull("hwchase17/react")

mechanic_agent = create_react_agent(llm, mechanic_tools, react_prompt)
mechanic_executor = AgentExecutor(
    agent=mechanic_agent,
    tools=mechanic_tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=5
)

# ==================== АГЕНТ-ДИСПЕТЧЕР ====================

# Кастомный промпт СО ВСЕМИ обязательными переменными
dispatcher_prompt = PromptTemplate.from_template("""
Ты — диспетчер автобусного парка. Твоя задача: найти свободные автобусы для замены.

У тебя есть доступ к следующим инструментам:
{tools}

Имена инструментов: {tool_names}

ОБЯЗАТЕЛЬНЫЕ ДЕЙСТВИЯ:
1. Вызови инструмент check_schedule для получения списка свободных автобусов.
2. Верни ТОЛЬКО те данные, которые вернул check_schedule.

ЖЁСТКИЕ ПРАВИЛА:
- НЕ придумывай номера автобусов, маршруты, имена водителей, время выезда.
- НЕ добавляй ничего от себя.
- НЕ используй фразы вроде "ближайший выезд через 15 минут".
- Если check_schedule вернул "Нет свободных автобусов" — напиши ровно это.
- Все числа и названия бери ТОЛЬКО из ответа check_schedule.
- Если в ответе check_schedule есть несколько автобусов — перечисли их все.

Используй следующий формат:

Question: вопрос, на который нужно ответить
Thought: что нужно сделать
Action: инструмент для вызова (должен быть один из [{tool_names}])
Action Input: входные данные для инструмента
Observation: результат работы инструмента
... (это Thought/Action/Action Input/Observation может повторяться)
Thought: Я получил достаточно информации
Final Answer: итоговый ответ

Начинаем!

Question: {input}
Thought: {agent_scratchpad}
""")

dispatcher_agent = create_react_agent(llm, dispatcher_tools, dispatcher_prompt)
dispatcher_executor = AgentExecutor(
    agent=dispatcher_agent,
    tools=dispatcher_tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=2
)

# ==================== ЦЕПОЧКИ ====================

# Цепочка 1: Извлечение симптомов из сообщения водителя
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

# Цепочка 2: Финальная рекомендация
recommend_prompt = PromptTemplate(
    input_variables=["user_message", "extracted_info", "mechanic_result", "dispatcher_result"],
    template="""
Ты — главный диспетчер автобусного парка. Дай водителю итоговый план действий.

Сообщение водителя: {user_message}
Извлечённая информация: {extracted_info}
Заключение механика (класс серьёзности и инструкция): {mechanic_result}
Информация о доступных автобусах: {dispatcher_result}

СФОРМИРУЙ ОТВЕТ СТРОГО В ФОРМАТЕ:

Класс серьёзности: [критическая/средняя/низкая]

План действий:
[конкретные шаги для водителя на основе инструкции механика]

Замена автобуса:
[перечисли ВСЕ автобусы из информации диспетчера, без изменений]
[если диспетчер сообщил "Нет свободных автобусов" — так и напиши]

ПРАВИЛА:
- Критическая = немедленная остановка и замена
- Средняя = доехать до конечной, если безопасно, затем замена
- Низкая = продолжить рейс, ремонт в парке
- НЕ придумывай номера автобусов, маршруты, время
- Используй ТОЛЬКО данные от механика и диспетчера
- Ответ ТОЛЬКО текстом, БЕЗ форматирования (**, *, #)
"""
)
recommend_chain = recommend_prompt | llm | StrOutputParser()
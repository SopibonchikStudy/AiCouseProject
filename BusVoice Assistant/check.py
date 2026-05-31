from agents import dispatcher_executor
from dotenv import load_dotenv
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

result = dispatcher_executor.invoke({
    "input": "Найди свободные автобусы для замены. Выбери наиболее подходящий."
})
print("=== Ответ диспетчера ===")
print(result["output"])
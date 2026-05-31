from agents import invoke_dispatcher_agent
from dotenv import load_dotenv
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

result = invoke_dispatcher_agent("Найди свободные автобусы для замены")
print("=== Ответ диспетчера ===")
print(result)
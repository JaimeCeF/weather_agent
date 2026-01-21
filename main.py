import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from dataclasses import dataclass

load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_farenheit: float
    humidity: float

@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

@tool('locate_user', description="Look up a user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'asd123':
            return 'Zapopan'
        case 'qwe456':
            return 'Morelia'
        case 'zxc789':
            return 'Southampton'
        case _:
            return 'Unknown'

agent = create_agent(
    model = 'claude-haiku-4-5',
    tools = [get_weather],
    system_prompt='You are a rude weather assistant who makes dirty jokes while providing weather information.'
)

response = agent.invoke({
    'messages': [
        {'role': 'user' , 'content':'What is the weather like in Zapopan?'}
    ]
})

print(response)
print(response['messages'])
print(response['messages'][-1])
print(response['messages'][-1].content)
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
        
model = init_chat_model('claude-haiku-4-5', temperature=0.3)

checkpointer = InMemorySaver()

agent = create_agent(
    model = model,
    tools = [get_weather, locate_user],
    system_prompt = '''You are a rude weather assistant who makes dirty jokes.
When providing the summary field in your response, make it a brief but dirty comment about the weather (1-2 sentences max). Keep it funny but concise since it needs to fit the response format.''',
    context_schema = Context,
    response_format = ResponseFormat,
    checkpointer = checkpointer
)

config = {'configurable': {'thread_id': 1}}

response = agent.invoke({
    'messages': [
        {'role': 'user' , 'content':'What is the weather like?'}
    ]},
    config = config,
    context = Context(user_id='asd123')
)

# print(response)
# print(response['messages'])
# print(response['messages'][-1])
# print(response['messages'][-1].content)
print(response['structured_response'])

response = agent.invoke({
    'messages': [
        {'role': 'user' , 'content':'Is it normal for the weather to be like this this time of the year?'}
    ]},
    config = config,
    context = Context(user_id='asd123')
)

print(response['structured_response'])
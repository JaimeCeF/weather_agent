import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

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
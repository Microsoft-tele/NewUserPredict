import openai
from flaml.autogen import AssistantAgent, UserProxyAgent

apiKey = "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"

openai.api_key = apiKey

llm_config = {
    "model": "gpt-3.5-turbo-0301",
    "functions": [
        {
            "name": "python",
            "description": "run cell in ipython and return the execution result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell": {
                        "type": "string",
                        "description": "Valid Python cell to execute.",
                    }
                },
                "required": ["cell"],
            },
        },
        {
            "name": "sh",
            "description": "run a shell script and return the execution result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Valid shell script to execute.",
                    }
                },
                "required": ["script"],
            },
        },
    ],
}

# create an AssistantAgent instance named "assistant"
assistant = AssistantAgent(name="assistant", llm_config=llm_config)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # in this mode, the agent will never solicit human input but always auto reply
)

# the assistant receives a message from the user, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""What date is today? Which big tech stock has the largest year-to-date gain this year? How much is the gain?""",
)
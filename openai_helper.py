"""
prepare to put into Learner
"""
from enum import Enum, auto
from typing import Callable, Dict, List, Union
#pylint:disable=no-name-in-module
from openai import ChatCompletion, OpenAI, RateLimitError

class OpenAIModel(Enum):
    """
    avoid stringly typed
    """
    THREEPOINTFIVETURBO = "gpt-3.5-turbo"

    def __str__(self) -> str:
        return self.value

class MyResponseFormat(Enum):
    """
    response format and not given not exposed
    and had to either not enter or enter as a
    Dict[str,str]
    """
    NOT_GIVEN = auto()
    JSON_RESPONSE = auto()

    def to_sendable(self) -> Dict[str,str]:
        """
        response format and not given not exposed
        """
        if self.value == MyResponseFormat.JSON_RESPONSE:
            return {"text": "json_object"}
        else:
            return {}

class Role(Enum):
    """
    avoid stringly typed
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    def __str__(self) -> str:
        return self.value

class Message:
    """
    avoid Dict[str,str]'ly typed
    """
    role: Role
    content : str

    def __init__(self,role,content):
        self.role = role
        self.content = content

    def to_sendable(self) -> Dict[str,str]:
        """
        convert to the type openai accepts
        because they didn't use a clean typed structure
        """
        return {"role" : str(self.role), "content" : self.content}

def make_completion(cur_client : OpenAI, model_type: OpenAIModel,
                    response_format : MyResponseFormat = MyResponseFormat.NOT_GIVEN) -> \
    Callable[[List[Message]],Union[ChatCompletion,RateLimitError]]:
    """
    curry client.chat.completions.create
    do errors as values unlike the exposed create
    when want JSON don't rely on the input message
        to ask for that
        you should not assume only the happy path for the input
    """
    def my_make_completion(messages : List[Message]) -> Union[ChatCompletion,RateLimitError]:
        actual_response_format = response_format.to_sendable()
        try:
            if len(actual_response_format) == 0:
                to_return = cur_client.chat.completions.create(
                    model=str(model_type),
                    messages = [
                        z.to_sendable() for z in messages
                    ]
                )
            else:
                message_json = Message(Role.SYSTEM,"Please respond with JSON")
                messages.insert(0,message_json)
                to_return = cur_client.chat.completions.create(
                    model=str(model_type),
                    response_format=actual_response_format,
                    messages = [
                        z.to_sendable() for z in messages
                    ]
                )
            return to_return
        except RateLimitError as e:
            return e

    return my_make_completion

"""
nothing
"""

from typing import List, Tuple

from openai import OpenAI
from openai_helper import Message, OpenAIModel, Role, make_completion
from combinatorial_conjecture import Learner, NAT

MyPromptType = str
MyTokenType = str
class MyLearner(Learner[MyPromptType,MyTokenType]):
    """
    given an integer it produces a list of bools,
    this doesn't actually learn, it just always produces a random list
    """
    def __init__(self, client : OpenAI, model_type : OpenAIModel, batch_size = 5):
        self.client = client
        self.model_type = model_type
        self.batch_size = batch_size
        self.curried_completion = make_completion(self.client, self.model_type)

    def get_batch_size(self) -> NAT:
        """
        convenient size
        """
        return self.batch_size

    def generate_token_list(self, prompt: MyPromptType) -> List[MyTokenType]:
        """
        send prompt as a user message
        """
        message_0 = Message(Role.SYSTEM,
        " ".join(["You are an assistant,",
        "with mathematical skill",
        "Try lots of cases that could be responses to the prompt",
        "Try weird degenerate examples. Try the trivial cases.",
        "Explore the fringes of what are possible good responses to the prompt.",
        "Use your knowledge of Coq and Lean."]))
        message_1 = Message(Role.USER,prompt)
        returned = self.curried_completion([message_0,message_1])
        if isinstance(returned,Exception):
            raise NotImplementedError
        return str(returned).split()

    def learn_from_these(self, samples: List[Tuple[MyPromptType, List[MyTokenType]]],
                        reinforce_these : bool) -> None:
        """
        no learning from being told these were good/bad lists of bool
        """
        if reinforce_these:
            # generate a user, assistant sequence of messages that treats
            # each in samples as a good case to emulate
            raise NotImplementedError
        else:
            # not as straightforward as the positive feedback
            raise NotImplementedError
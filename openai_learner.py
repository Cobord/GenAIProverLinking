"""
inherit from the generic abstract base class learner
specialized to one that uses openai
"""

import time
from typing import Callable, List, Optional, Tuple
#pylint:disable=import-error
from openai import AuthenticationError, OpenAI, RateLimitError
from openai_helper import Message, OpenAIModel, Role, make_completion
from combinatorial_conjecture import Learner, NAT, LearnerException

MyPromptType = str
MyTokenType = str
ReinforceType = Callable[[MyPromptType,List[MyTokenType]], List[Message]]
class OpenAILearner(Learner[MyPromptType,MyTokenType]):
    """
    given an string it produces a list of strings
    see the abstract base class
    """
    def __init__(self, client : OpenAI, model_type : OpenAIModel, batch_size = 5):
        self.client = client
        self.model_type = model_type
        self.batch_size = batch_size
        self.curried_completion = make_completion(self.client, self.model_type)
        message_0 = Message(Role.SYSTEM,
            " ".join(["You are an assistant,",
            "with mathematical skill",
            "Try lots of cases that could be responses to the prompt",
            "Try weird degenerate examples. Try the trivial cases.",
            "Explore the fringes of what are possible good responses to the prompt.",
            "Use your knowledge of Coq and Lean."]))
        returned = self.curried_completion([message_0])
        if isinstance(returned,AuthenticationError):
            # the client was invalid, won't be able to initialize this learner
            raise LearnerException("authentication failed when initializing") from returned
        if isinstance(returned,RateLimitError):
            #pylint:disable=invalid-name
            MAX_TRIALS = 5
            for trial in range(MAX_TRIALS):
                time.sleep(10*2**trial)
                returned = self.curried_completion([message_0])
                if isinstance(returned,AuthenticationError):
                    raise LearnerException("authentication failed when initializing") from returned
                if not isinstance(returned,RateLimitError):
                    break
            else:
                raise LearnerException("".join(["even after MAX_TRIALS slowing down the rate,",
                                                "we still got rate limit error"])) from returned
        #pylint:disable=pointless-string-statement
        """
        returned is the actual response to the prompt
        but we sent it just an initial prompt without the content of what math
        we actually want it to do, so the response is just an acknowledgement
        of the general task
         """
        self.__override_negative : Optional[ReinforceType] = None
        self.__override_positive : Optional[ReinforceType] = None

    def get_batch_size(self) -> NAT:
        """
        convenient size
        """
        return self.batch_size

    def generate_token_list(self, prompt: MyPromptType) -> List[MyTokenType]:
        """
        send prompt as a user message
        the List[MyTokenType] = List[str] is the part that is
        supposed to be parseable into the mathematical structure in question
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
            raise LearnerException("".join(["The learner initialized correctly",
                                                "but we still got runtime error"])) from returned
        return str(returned).split()

    def learn_from_these(self, samples: List[Tuple[MyPromptType, List[MyTokenType]]],
                        reinforce_these : bool) -> None:
        """
        positive or negative feedback of samples
        see the abstract base class docstring
        """
        #pylint:disable=no-else-raise
        messages_to_send = []
        if reinforce_these:
            # generate a user, assistant sequence of messages that treats
            # each in samples as a good case to emulate
            for sample_prompt, good_response in samples:
                next_messages = self.__positive_reinforce(sample_prompt, good_response)
                messages_to_send.extend(next_messages)
            after_learning_response = self.curried_completion(messages_to_send)
            if isinstance(after_learning_response, Exception):
                raise LearnerException("".join([
                    "The learner initialized correctly",
                    "but we still got runtime error"])) from after_learning_response
        else:
            # not as straightforward as the positive feedback
            for sample_prompt, good_response in samples:
                next_messages = self.__negative_reinforce(sample_prompt, good_response)
                messages_to_send.extend(next_messages)
            after_learning_response = self.curried_completion(messages_to_send)
            if isinstance(after_learning_response, Exception):
                raise LearnerException("".join([
                    "The learner initialized correctly",
                    "but we still got runtime error"])) from after_learning_response

    def override_positive(self, new_positive_reinforcer : ReinforceType):
        """
        not the generic positive reinforcement messages
        """
        self.__override_positive = new_positive_reinforcer

    def override_negative(self, new_negative_reinforcer : ReinforceType):
        """
        not the generic negative reinforcement messages
        """
        self.__override_negative = new_negative_reinforcer

    def __positive_reinforce(self,
                             sample_prompt : MyPromptType,
                             good_response: List[MyTokenType]) -> List[Message]:
        """
        the user, assistant sequence of messages for a single good sample to emulate
        """
        if self.__override_positive is not None:
            return self.__override_positive(sample_prompt, good_response)
        raise NotImplementedError

    def __negative_reinforce(self,
                             sample_prompt : MyPromptType,
                             bad_response: List[MyTokenType]) -> List[Message]:
        """
        the user, assistant sequence of messages for a single bad sample to avoid
        """
        if self.__override_negative is not None:
            return self.__override_negative(sample_prompt, bad_response)
        raise NotImplementedError

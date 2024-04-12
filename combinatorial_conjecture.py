"""
abstract structure for a learner that produces counterexamples to combinatorial conjectures
the learner and the combinatorial object are decoupled
the learner only knows about producing token streams, not how those token streams translate to
combinatorial objects / the conjecture on that sort of object
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import floor
import random
from typing import Callable, Generic, List, Optional, Tuple, TypeVar

TOKEN_TYPE = TypeVar("TOKEN_TYPE")  # pylint: disable=invalid-name
A = TypeVar('A')
B = TypeVar('B')
T_COMBINATORIAL_OBJECT = TypeVar("T_COMBINATORIAL_OBJECT", bound="CombinatorialObject")  # pylint: disable=invalid-name
NAT = int
SCORE = float
MAX_SCORE = 1e9
PERCENTILE = float
TRIPLE = Tuple[A,A,A]

def keep_these_idces(my_list: List[A], which_idces: List[NAT]) -> List[A]:
    """
    keep the specified indices only
    both modifies the input list and returns that modified list
    so you can do x=keep_these_idces(x,y) or _ = keep_these_idces(x,y)
    allowing chaining
    """
    which_idces.sort()
    my_list[:] = map(lambda x:my_list[x],which_idces)
    return my_list

def or_else(my_item: Optional[A], default: A) -> A:
    """
    the type explains it
    """
    return my_item if my_item is not None else default

def optional_fmap(my_f : Callable[[A],B],my_item : Optional[A]) -> Optional[B]:
    """
    the type explains it 
    """
    return None if my_item is None else my_f(my_item)

class CombinatorialObject(ABC,Generic[TOKEN_TYPE]):
    """
    some sort of combinatorial object that can be constructed from
    a list of TOKEN_TYPE
    """
    @abstractmethod
    def from_token_list(self: T_COMBINATORIAL_OBJECT,
                        token_list: List[TOKEN_TYPE]) -> Optional[T_COMBINATORIAL_OBJECT]:
        """
        construct the object from token list
        """

    def is_valid(self : T_COMBINATORIAL_OBJECT,
                token_list: List[TOKEN_TYPE]) -> bool:
        """
        did this token list properly produce a combinatorial object
        """
        return self.from_token_list(token_list) is not None


@dataclass(eq=False, order=False, frozen=True)
class CombinatorialConjecture(Generic[TOKEN_TYPE,T_COMBINATORIAL_OBJECT]):
    """
    some sort of conjecture on the combinatorial object
    along with a scoring function that is a measure of how `close` we are to
    violating the conjecture
    """
    object_default: T_COMBINATORIAL_OBJECT
    conjecture: Callable[[T_COMBINATORIAL_OBJECT], bool]
    scorer: Callable[[T_COMBINATORIAL_OBJECT], SCORE]

    def __and__(self, other) -> CombinatorialConjecture[TOKEN_TYPE,T_COMBINATORIAL_OBJECT]:
        """
        z3 = z1 & z2
        """
        return CombinatorialConjecture[TOKEN_TYPE,T_COMBINATORIAL_OBJECT](self.object_default,
            lambda z: self.conjecture(z) and other.conjecture(z),
            lambda z: min(self.scorer(z) , other.scorer(z)))

    def __or__(self, other) -> CombinatorialConjecture[TOKEN_TYPE,T_COMBINATORIAL_OBJECT]:
        """
        z3 = z1 | z2
        """
        return CombinatorialConjecture[TOKEN_TYPE,T_COMBINATORIAL_OBJECT](self.object_default,
            lambda z: self.conjecture(z) or other.conjecture(z),
            lambda z: max(self.scorer(z) , other.scorer(z)))

    def __invert__(self) -> CombinatorialConjecture[TOKEN_TYPE,T_COMBINATORIAL_OBJECT]:
        """
        z2 = ~z1
        """
        return CombinatorialConjecture[TOKEN_TYPE,T_COMBINATORIAL_OBJECT](self.object_default,
            lambda z: not(self.conjecture(z)),
            lambda z: -self.scorer(z))

    def from_token_list(self, token_list: List[TOKEN_TYPE]) -> Optional[T_COMBINATORIAL_OBJECT]:
        """
        from this token list produce a combinatorial object
        """
        return self.object_default.from_token_list(token_list)

    def is_valid(self, token_list: List[TOKEN_TYPE]) -> bool:
        """
        did this token list properly produce a combinatorial object
        """
        return self.object_default.is_valid(token_list)

    def filter_most_extreme_scores(self,tokens_for_many_objects : List[Tuple[A,List[TOKEN_TYPE]]],
            percentile_keep : PERCENTILE,
            best_vs_worst : bool = False) -> TRIPLE[List[Tuple[A,List[TOKEN_TYPE]]]]:
        """
        make the objects for each of the entries in tokens_for_many_objects
        compute their scores
        give back the best percentile of them (but made to be at least 1)
        as well as the worst percentile
            (usually the same number as the best
            but may be smaller in order to make sure best and worst are disjoint)
        as well as all the cases that didn't give valid objects
        """
        assert 0<=percentile_keep<=1
        invalids = list(filter(lambda z : not(self.is_valid(z[1])),tokens_for_many_objects))
        if percentile_keep == 0:
            tokens_for_many_objects[:] = []
            return tokens_for_many_objects,[],invalids
        tokens_for_many_objects[:] = list(filter(lambda z : self.is_valid(z[1]),
            tokens_for_many_objects))
        if percentile_keep == 1:
            return tokens_for_many_objects,[],invalids
        how_many_keep = floor(percentile_keep*len(tokens_for_many_objects))
        how_many_keep = max(how_many_keep,1)
        most_extreme_score = MAX_SCORE if best_vs_worst else -MAX_SCORE
        score_for_keeping : Callable[[Tuple[A,List[TOKEN_TYPE]]],SCORE] = \
            lambda item : or_else(optional_fmap(self.scorer,\
                            self.from_token_list(item[1])),most_extreme_score)
        tokens_for_many_objects.sort(key=score_for_keeping ,reverse = not best_vs_worst)
        how_many_bad = min(how_many_keep,len(tokens_for_many_objects)-how_many_keep)
        bad_cases = tokens_for_many_objects[-1:-how_many_bad-1:-1].copy()
        bad_cases.reverse()
        tokens_for_many_objects[:] = tokens_for_many_objects[:how_many_keep]
        return tokens_for_many_objects,bad_cases,invalids

    def test_conjecture(self, my_object_tokens: List[TOKEN_TYPE]) -> Optional[bool]:
        """
        test the conjecture on the object built from the given tokens
        """
        my_object = self.object_default.from_token_list(my_object_tokens)
        return None if my_object is None else self.conjecture(my_object)

class LearnerException(RuntimeError):
    """
    custom errors for learners failing at runtime
    """

class Learner(ABC, Generic[A,TOKEN_TYPE]):
    """
    some sort of function from A to lists of TOKEN_TYPE
    such that one can reinforce the good input,output pairs
    and punish the bad input,output pairs
    so eventually it can turn into the function we want
    """

    @abstractmethod
    def get_batch_size(self) -> NAT:
        """
        how many to produce in one round
        """

    @abstractmethod
    def generate_token_list(self, prompt: A) -> List[TOKEN_TYPE]:
        """
        produce a token list that as learn more
        is likely to represents some `good` class of combinatorial object
        (good is more likely to violate the conjecture)
        """

    def generate_batch(self, prompt: A) -> List[List[TOKEN_TYPE]]:
        """
        generate multiple token lists
        """
        how_many = self.get_batch_size()
        assert how_many > 0
        return [self.generate_token_list(prompt) for _ in range(how_many)]

    @abstractmethod
    def learn_from_these(self, samples: List[Tuple[A, List[TOKEN_TYPE]]],
                         reinforce_these : bool) -> None:
        """
        if reinforce_these, change the internal parameters of learner
        so next call to generate_batch produces more like these
        if not reinforce_these, change the internal parameters of learner
        so next call to generate_batch produces less like these
        """


@dataclass(eq=False,order=False,frozen=True)
class CombinatorialConjectureDisprover(Generic[A, TOKEN_TYPE, T_COMBINATORIAL_OBJECT]):
    """
    the learner takes care of the generation of token streams
    the conjecture takes care of turning those streams to objects and scoring those object/
        whether their score is so low as to be a counterexample
    combine those two here
    """
    my_conjecture: CombinatorialConjecture[TOKEN_TYPE,T_COMBINATORIAL_OBJECT]
    my_learner: Learner[A,TOKEN_TYPE]

    def learn_once(self, prompt: A,
            percentile_learn_from : PERCENTILE = .1) -> Optional[T_COMBINATORIAL_OBJECT]:
        """
        the learner generates a bunch of token lists that can be used to construct objects,
        build a bunch of them and score them
        reinforce the learner so it produces more like the best ones
        and less like the worst/invalid ones
        if a counterexample to the conjecture was produced in this batch, return it
        """
        candidates = self.my_learner.generate_batch(prompt)
        prompt_and_candidate = [(prompt, z) for z in candidates]
        prompt_and_candidate_low,prompt_and_candidate_high,prompt_and_candidate_invalid = \
            self.my_conjecture.filter_most_extreme_scores(
            prompt_and_candidate, percentile_learn_from, best_vs_worst = False)
        predicate: Callable[[Tuple[A, List[TOKEN_TYPE]]], bool] = lambda pc: or_else(
            optional_fmap(lambda b: not b,self.my_conjecture.test_conjecture(pc[1])), False)
        counter_example = next(filter(predicate, prompt_and_candidate_low), None)
        self.my_learner.learn_from_these(prompt_and_candidate_low,reinforce_these=True)
        self.my_learner.learn_from_these(prompt_and_candidate_high,reinforce_these=False)
        self.my_learner.learn_from_these(prompt_and_candidate_invalid,reinforce_these=False)
        return optional_fmap(lambda z: self.my_conjecture.from_token_list(z[1]),counter_example)

    def generate_counterexample(self, prompt: A, learn_limit: NAT,
        percentile_keep : PERCENTILE = .1) -> Optional[T_COMBINATORIAL_OBJECT]:
        """
        try learning a few times and
        if we find a counterexample in the meantime, return it 
        """
        for _ in range(learn_limit):
            counter_example = self.learn_once(prompt, percentile_keep)
            if counter_example is not None:
                return counter_example
        return None

def non_learning_test():
    """
    subclass the ABCs in a way
    that no learning takes place
    checks the plumbing of the pieces
    without the actual hard parts
    """
    MyTokenType = bool
    @dataclass
    class MyStructure(CombinatorialObject[MyTokenType]):
        """
        trivial construction from the token list
        just the identity
        """
        my_obj : List[MyTokenType]
        def from_token_list(self,
                        token_list: List[MyTokenType]) -> Optional[MyStructure]:
            """
            construct the object from token list
            """
            return MyStructure(token_list)

        def even_num_true(self) -> bool:
            """
            self explanatory
            """
            return self.count_true() % 2 == 0

        def count_true(self) -> int:
            """
            self explanatory
            """
            return sum(self.my_obj)

        def __repr__(self) -> str:
            return self.my_obj.__repr__()

    default_struct = MyStructure([])
    conj = CombinatorialConjecture[MyTokenType,MyStructure](default_struct,
            lambda z : z.even_num_true(),lambda z: z.count_true())

    sample = conj.from_token_list([True,False])
    assert (sample is not None) and (sample.my_obj == [True,False])
    assert not conj.test_conjecture([True,False])
    assert conj.test_conjecture([True,True])
    assert conj.test_conjecture([False])

    MyPromptType = int
    class MyLearner(Learner[MyPromptType,MyTokenType]):
        """
        given an integer it produces a list of bools,
        this doesn't actually learn, it just always produces a random list
        """
        def get_batch_size(self) -> NAT:
            """
            convenient size
            """
            return 5

        def generate_token_list(self, prompt: MyPromptType) -> List[MyTokenType]:
            """
            always produces a random list, no learning
            """
            return [random.choice([True,False]) for _ in range(prompt)]

        def learn_from_these(self, samples: List[Tuple[MyPromptType, List[MyTokenType]]],
                         reinforce_these : bool) -> None:
            """
            no learning from being told these were good/bad lists of bool
            """
            return

    cur_learner = MyLearner()
    disprover = CombinatorialConjectureDisprover[MyPromptType,
                                                 MyTokenType,
                                                 MyStructure](conj,cur_learner)
    print("The sample conjecture we are going to get counterexamples for is",
          "that lists of booleans have an even number of true")
    for trial_len in [0,1,2,6]:
        my_counter_example = disprover.generate_counterexample(trial_len,learn_limit = 10)
        print(f"Counterexample of length {trial_len} {my_counter_example}")

if __name__ == "__main__":
    non_learning_test()

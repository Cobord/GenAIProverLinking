"""
encapsulation of coq
https://github.com/ejgallego/pyCoq
"""

from __future__ import annotations
from typing import List, Optional
# import pycoq, coq
from combinatorial_conjecture import CombinatorialObject

class CoqObject(CombinatorialObject[str]):
    """
    encapsulation of coq
    """

    def __init__(self, opts, ppopts):
        if opts is None:
            self.opts = {"newtip": None, "ontop": 1, "lim": 20 }
        else:
            self.opts = opts
        if ppopts is None:
            self.ppopts = { "pp_format": ("PpSer",None),
                           "pp_depth": 1000,
                           "pp_elide": "...", "pp_margin": 90}
        else:
            self.ppopts = ppopts
        self.res = None

    def from_token_list(self,
                        token_list: List[str]) -> Optional[CoqObject]:
        """
        construct the object from token list
        """
        # res1 = coq.add(" ".join(token_list), self.opts)
        # res2 = coq.exec(5)
        # if res2 indicates some sort of error, return None
        # new_instance = CoqObject(self.opts,self.ppopts)
        # new_instance.res = res2
        # return new_instance
        raise NotImplementedError

if __name__ == "__main__":
    temp = CoqObject(None,None)
    assert isinstance(temp, CombinatorialObject)

"""Special control tokens for moral deliberation and alignment."""

from typing import List, Dict

BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
CONSCIENCE_START_TOKEN = "<|conscience|>"
CONSCIENCE_END_TOKEN = "<|/conscience|>"
REFUSE_TOKEN = "<|refuse|>"

ALL_SPECIAL_TOKENS: List[str] = [
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    CONSCIENCE_START_TOKEN,
    CONSCIENCE_END_TOKEN,
    REFUSE_TOKEN,
]

SPECIAL_TOKEN_IDS: Dict[str, int] = {
    BOS_TOKEN: 0,
    EOS_TOKEN: 1,
    PAD_TOKEN: 2,
    USER_TOKEN: 3,
    ASSISTANT_TOKEN: 4,
    CONSCIENCE_START_TOKEN: 5,
    CONSCIENCE_END_TOKEN: 6,
    REFUSE_TOKEN: 7,
}


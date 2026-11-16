"""Special control tokens for moral deliberation, voice turn-taking, and local action execution."""

from typing import List, Dict

BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

# Synthetic Conscience Deliberation Tokens
CONSCIENCE_START_TOKEN = "<|conscience|>"
CONSCIENCE_END_TOKEN = "<|/conscience|>"
REFUSE_TOKEN = "<|refuse|>"

# Local Tool & Action Execution Tokens
ACTION_START_TOKEN = "<|action|>"
ACTION_END_TOKEN = "<|/action|>"
OBSERVATION_START_TOKEN = "<|observation|>"
OBSERVATION_END_TOKEN = "<|/observation|>"

ALL_SPECIAL_TOKENS: List[str] = [
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    CONSCIENCE_START_TOKEN,
    CONSCIENCE_END_TOKEN,
    REFUSE_TOKEN,
    ACTION_START_TOKEN,
    ACTION_END_TOKEN,
    OBSERVATION_START_TOKEN,
    OBSERVATION_END_TOKEN,
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
    ACTION_START_TOKEN: 8,
    ACTION_END_TOKEN: 9,
    OBSERVATION_START_TOKEN: 10,
    OBSERVATION_END_TOKEN: 11,
}


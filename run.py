import os

from vllm import LLM, SamplingParams

prompts = [
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
'The center of the universe is',
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-125m")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

import logging
from typing import Optional

from langchain import LLMChain, PromptTemplate
from langchain.llms.base import BaseLLM

from nemoguardrails.actions.llm.utils import (
    get_multiline_response,
    llm_call,
    strip_quotes,
)
from nemoguardrails.llm.params import llm_params
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.llm.types import Task

log = logging.getLogger(__name__)

HALLUCINATION_NUM_EXTRA_RESPONSES = 2


async def check_hallucination_llama(
    llm_task_manager: LLMTaskManager,
    context: Optional[dict] = None,
    llm: Optional[BaseLLM] = None,
    use_llm_checking: bool = True,
):
    """Checks if the last bot response is a hallucination by checking multiple completions for self-consistency.

    :return: True if hallucination is detected, False otherwise.
    """

    bot_response = context.get("last_bot_message")
    last_user_message_string = context.get("last_user_message")
    num_responses = HALLUCINATION_NUM_EXTRA_RESPONSES
    
    # Use the "generate" call from langchain to get all completions in the same response.
    last_bot_prompt = PromptTemplate(template="{text}", input_variables=["text"])
    chain = LLMChain(prompt=last_bot_prompt, llm=llm)
    
    extra_responses = []
    for i in range(num_responses):
        result = chain.run(last_user_message_string)    
        result = get_multiline_response(result)
        result = strip_quotes(result)
        extra_responses.append(result)
    
    if len(extra_responses) == 0:
        print(f"No extra LLM responses were generated for '{bot_response}' hallucination check.")
        return False
    elif len(extra_responses) < num_responses:
        print(f"Requested {num_responses} extra LLM responses for hallucination check, "
            f"received {len(extra_responses)}.")
        
    if use_llm_checking:
        # Only support LLM-based agreement check in current version
        prompt = llm_task_manager.render_task_prompt(
            task=Task.CHECK_HALLUCINATION,
            context={
                "statement": bot_response,
                "paragraph": " ".join(extra_responses),
            },
        )

        with llm_params(llm):#, temperature=0.0):
            agreement = await llm_call(llm, prompt)

        agreement = agreement.lower().strip()
        print(f"Agreement result for looking for hallucination is {agreement}.")
        # Return True if the hallucination check fails
        return "no" in agreement

    return False

## We need to register the new action

app.register_action(check_hallucination_llama, name="check_hallucination_llama") 
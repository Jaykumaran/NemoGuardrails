Credits& References:

Forked from https://github.com/marvik-ai/llama2-nemo-guardrails

['NemoGuardrails Nvidia'](https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/docs/user_guides/guardrails-library.md)

['Marvik ai'](https://blog.marvik.ai/2023/10/09/enhancing-llama2-conversations-with-nemo-guardrails-a-practical-guide/)

['Plaban Nayak'](https://medium.aiplanet.com/retrieval-augmented-pipeline-with-actions-using-nemo-gaurdrails-447b84a5334b)

['Nivida Issues Thread Section'](https://github.com/NVIDIA/NeMo-Guardrails/issues/238) 

['Pinecone'](https://www.pinecone.io/learn/nemo-guardrails-intro/)

Special Thanks @Wildshire

                                                    GUARDRAILS
NEMOGUARDRAILS -Nvidia:
Guardrails (or "rails" for short) are specific ways of controlling the output of a large language model, such as not talking about politics, responding in a particular way to specific user requests, following a predefined dialog path, using a particular language style, extracting structured data, and more.


Read any Medium blog for free(as some blogs referred here are paywalled ):
https://www.youtube.com/shorts/T52uUUPkRNI


Start with: Youtube Tutorial then ref docs and blogs

https://youtu.be/SwqusllMCnE?feature=shared
https://youtu.be/3Vg5bEtaepQ?feature=shared
https://youtu.be/alDcHItL6yo?feature=shared


Above tutorial associated Notebooks:


https://www.pinecone.io/learn/nemo-guardrails-intro/
https://github.com/pinecone-io/examples/tree/master/learn/generation/chatbots/nemo-guardrails


Must read:
 https://ksankar.medium.com/generative-ai-guardrails-functional-compositional-to-address-the-enhanced-treat-surface-43366685e466

 
Docs:
https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/docs


NemoGuardrails Architecture Diagram:
https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/docs/architecture


Setting up Pipelines for various HF models:
 https://github.com/NVIDIA/NeMo-Guardrails/tree/develop/examples/configs/llm

 
Colang or Canonical form vs Semantic:
Canonical language is the "how" of saying things, while semantic is the "what" of it.
Canonical language is preferred for clarity and precision, while semantic analysis considers all possible meanings and interpretations.


NemoGuardrails Library: 
https://github.com/NVIDIA/NeMoGuardrails/blob/develop/docs/user_guides/guardrails-library.md


llm_call function calling:
 https://github.com/NVIDIA/NeMo-Guardrails/blob/develop/nemoguardrails/actions/llm/utils.py

 
NemoGuardrails Implementations:

With OpenAI:

https://towardsdatascience.com/configuring-nemo-guardrails-your-way-an-alternative-method-for-large-language-models-c82aaff78f6e

RAG with OpenAI and NemoGuardrails: 

https://medium.aiplanet.com/retrieval-augmented-pipeline-with-actions-using-nemo-gaurdrails-447b84a5334b

With Llama2:

https://blog.marvik.ai/2023/10/09/enhancing-llama2-conversations-with-nemo-guardrails-a-practical-guide/

Above blog repo: 

https://github.com/marvik-ai/llama2-nemo-guardrails

Issues faced with above implementation community discussion thread:

https://github.com/NVIDIA/NeMo-Guardrails/issues/238


My version of above repo fixed:

https://github.com/Jaykumaran/NemoGuardrails/tree/main/GuardRAILS%20LLAMA2/llama2-nemo-guardrails  #Read Markdown headings in notebooks carefully to understand setup.


Guardrails AI:

Docs: 

https://www.guardrailsai.com/docs

Example Notebooks:

https://github.com/guardrails-ai/guardrails/tree/main/docs/examples

SEMANTIC ROUTER: (Different from Guardrails)

Tutorial: 

https://www.youtube.com/watch?v=ro312jDqAh0
Notebooks: https://github.com/aurelio-labs/semantic-router/tree/main/docs
Blogs: https://medium.com/ai-insights-cobet/beyond-basic-chatbots-how-semantic-router-is-changing-the-game-783dd959a32d

LLAMA GUARD: (Purple Llama)
https://towardsdatascience.com/safeguarding-your-rag-pipelines-a-step-by-step-guide-to-implementing-llama-guard-with-llamaindex-6f80a2e07756



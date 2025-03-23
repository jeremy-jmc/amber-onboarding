# !python3 -m pip install evaluate rouge_score bert_score

import evaluate

# metrics = evaluate.list_evaluation_modules(module_type="comparison", include_community=True, with_details=True)
# metrics
# help(evaluate.load)

perplexity = evaluate.load("perplexity")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bert_score = evaluate.load("bertscore")


"""
https://www.datacamp.com/blog/llm-evaluation
"""
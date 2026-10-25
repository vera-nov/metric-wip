"""Lazy model loading; each caller runs in its own configured Python environment."""
def generation_model_class(model_name):
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if config.model_type in {"qwen3_5", "qwen3_5_moe", "gemma3"}:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def require_complete_generation(tokens, max_new_tokens, eos_token_id):
    """Do not turn cut-off translations or GREEN answers into valid cached scores."""
    ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
    if len(tokens) >= max_new_tokens and int(tokens[-1]) not in ids:
        raise ValueError("Generation hit max_new_tokens; increase the budget and rerun this stage")

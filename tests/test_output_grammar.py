"""Grammatiche di uscita non-vocab: Trie per frase (CoNLL) e sequenze (WOS).

Il controllo end-to-end su tutte le 20.744 frasi di CoNLL e le 46.985 di WOS
(ogni gold ammessa dal Trie costruito dal prompt reale) è stato fatto sui
dati veri; qui restano i casi che lo rendono vero con un tokenizer reale.
"""

from __future__ import annotations

import pytest
import torch

from src.grammar.grammar_logits_processor import build_logits_processor
from src.grammar.output_grammar import OutputGrammar, build_output_grammar
from src.utils.prompting import build_t2g_prompt, extract_query_source

CONLL_CFG = {
    "dataset": {"prompt_profile": "en-conll"},
    "grammar": {
        "mode": "source_spans",
        "span_types": ["PER", "ORG", "LOC", "MISC"],
        "empty_token": "NONE",
        "max_span_words": 10,
    },
}
WOS_CFG = {"dataset": {"prompt_profile": "en-wos"}, "grammar": {"mode": "sequences"}}
SENTENCE = "EU rejects German call to boycott British lamb ."


def _conll_grammar() -> OutputGrammar:
    return build_output_grammar(CONLL_CFG, vocab=["ORG:EU", "NONE"])


def _wos_grammar() -> OutputGrammar:
    return build_output_grammar(
        WOS_CFG,
        vocab=["CS", "Medical", "Machine_learning", "Depression"],
        sequences=["CS Machine_learning", "Medical Depression", "CS Machine_learning"],
    )


# ── Grammatica a parole (nessun tokenizer) ──────────────────────────────


def test_source_spans_accepts_only_pieces_of_the_sentence() -> None:
    g = _conll_grammar()
    assert g.accepts("ORG:EU MISC:German MISC:British", SENTENCE)
    assert g.accepts("NONE", SENTENCE)
    # Entità inventata (non è un pezzo della frase) e NONE non da solo.
    assert not g.accepts("PER:Mary", SENTENCE)
    assert not g.accepts("NONE ORG:EU", SENTENCE)
    assert not g.accepts("XYZ:EU", SENTENCE)
    # Senza frase si controlla solo la forma TIPO:qualcosa.
    assert g.accepts("PER:Mary", None)


def test_sequences_accept_only_whole_allowed_pairs() -> None:
    g = _wos_grammar()
    assert g.accepts("CS Machine_learning")
    assert not g.accepts("CS Depression")  # area di un altro dominio
    assert not g.accepts("Medical Medical")
    assert not g.accepts("Machine_learning CS")
    assert not g.accepts("CS")


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="grammar.mode"):
        build_output_grammar({"grammar": {"mode": "regex"}}, vocab=[])


# ── Prompt → frase ──────────────────────────────────────────────────────


def test_query_source_round_trips_zero_and_few_shot(tokenizer) -> None:
    zero = build_t2g_prompt(SENTENCE, tokenizer, profile="en-conll")
    few = build_t2g_prompt(
        SENTENCE,
        tokenizer,
        examples=[{"text": "Peter Blackburn", "gloss": "PER:Peter_Blackburn"}],
        profile="en-conll",
    )
    for prompt in (zero, few):
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        assert extract_query_source(decoded, "en-conll") == SENTENCE


# ── Trie ────────────────────────────────────────────────────────────────


def _allowed_after(proc, prompt_ids, text_prefix_ids):
    mask = proc.allowed_mask_for_prefixes(
        [text_prefix_ids], len(proc.tokenizer), prompts=[prompt_ids]
    )
    return set(torch.nonzero(mask[0]).flatten().tolist())


def test_span_trie_admits_gold_and_closes_after_none(tokenizer) -> None:
    g = _conll_grammar()
    proc = build_logits_processor(CONLL_CFG, g, sorted(g.vocab), tokenizer)
    prompt = build_t2g_prompt(SENTENCE, tokenizer, profile="en-conll")
    pids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id

    gold = tokenizer.encode("ORG:EU MISC:German MISC:British", add_special_tokens=False)
    for k in range(len(gold)):
        assert gold[k] in _allowed_after(proc, pids, gold[:k])
    assert eos in _allowed_after(proc, pids, gold)

    # Niente uscita vuota; NONE ammesso, e dopo NONE solo EOS.
    assert eos not in _allowed_after(proc, pids, [])
    none = tokenizer.encode("NONE", add_special_tokens=False)
    assert _allowed_after(proc, pids, none) == {eos}

    # Un'entità che non è nella frase non è generabile.
    mary = tokenizer.encode("PER:Mary", add_special_tokens=False)
    walked = all(
        mary[k] in _allowed_after(proc, pids, mary[:k]) for k in range(len(mary))
    )
    assert not walked


def test_span_mode_needs_the_prompt(tokenizer) -> None:
    g = _conll_grammar()
    proc = build_logits_processor(CONLL_CFG, g, sorted(g.vocab), tokenizer)
    with pytest.raises(ValueError, match="prompt"):
        proc.allowed_mask_for_prefixes([[]], len(tokenizer))


def test_sequence_trie_forces_domain_then_its_area_then_eos(tokenizer) -> None:
    g = _wos_grammar()
    proc = build_logits_processor(WOS_CFG, g, sorted(g.vocab), tokenizer)
    eos = tokenizer.eos_token_id

    def walk(text: str) -> bool:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return all(ids[k] in _allowed_after(proc, [], ids[:k]) for k in range(len(ids)))

    assert walk("CS Machine_learning")
    end = tokenizer.encode("CS Machine_learning", add_special_tokens=False)
    assert _allowed_after(proc, [], end) == {eos}
    assert not walk("CS Depression")
    assert not walk("Medical Medical")
    # Dopo il solo dominio la fine non è ammessa.
    assert eos not in _allowed_after(
        proc, [], tokenizer.encode("CS", add_special_tokens=False)
    )


# ── Validity e format seguono la grammatica ─────────────────────────────


def test_validity_and_format_follow_the_grammar(monkeypatch) -> None:
    from src.rewards import t2g_rewards
    from src.utils.metrics import check_gloss_validity

    monkeypatch.setattr(t2g_rewards, "_grammar", _conll_grammar())
    assert check_gloss_validity("ORG:EU", SENTENCE) == (True, "")
    assert check_gloss_validity("PER:Mary", SENTENCE) == (False, "grammar_violation")
    assert t2g_rewards.gloss_format_reward("ORG:EU MISC:German", SENTENCE) == 1.0
    # Una parola su due ammessa: gradino intermedio, non 1.0.
    assert t2g_rewards.gloss_format_reward("ORG:EU PER:Mary", SENTENCE) < 1.0
    # Etichetta più lunga di format_max_token_len (25) ma ammessa: niente
    # penalità di lunghezza, decide la grammatica.
    assert (
        t2g_rewards.gloss_format_reward(
            "MISC:Africa_Cup_of_Nations", "the Africa Cup of Nations final"
        )
        == 1.0
    )

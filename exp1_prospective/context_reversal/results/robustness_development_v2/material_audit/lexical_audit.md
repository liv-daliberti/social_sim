# Controlled-edit lexical audit

20 held-out parent groups; 240 rows. All variants remain in their parent fold.

Identical news across contexts: True. Identical unigram inventories: True.

| Baseline | Direction accuracy | Relevance balanced accuracy |
|---|---:|---:|
| news_unigram | 0.333 | 0.500 |
| full_unigram | 0.333 | 0.500 |
| full_bigram | 0.333 | 0.500 |
| full_character | 0.325 | 0.478 |

Identical word inventories block unigram context-type shortcuts. Bigram/character models can still encode relationship bindings; report their performance without selecting or dropping target failures.

# Frozen-material lexical audit

Exploratory diagnostic on authored, unvalidated development materials. No model responses entered this audit. Features use visible prompts only; mechanism-class metadata, when supplied, is used exclusively to form held-out groups.

All repetitions are deduplicated. Leave-one-family-out cross-validation keeps every context from a family out of training together; TF-IDF is fit inside each fold. Hyperparameters are fixed, with no search. Masked cases are excluded.

| Task | Features | Classifier | Accuracy | Balanced accuracy | Whole-family correctness |
|---|---|---|---:|---:|---:|
| relevance | none | training majority | 0.667 | 0.500 | 0.000 |
| relevance | news_only | multinomial_nb | 0.667 | 0.500 | 0.000 |
| relevance | news_only | logistic_regression | 0.333 | 0.500 | 0.000 |
| relevance | question_and_news | multinomial_nb | 0.667 | 0.500 | 0.000 |
| relevance | question_and_news | logistic_regression | 0.333 | 0.500 | 0.000 |
| relevance | full_visible_context_and_news | multinomial_nb | 0.667 | 0.500 | 0.000 |
| relevance | full_visible_context_and_news | logistic_regression | 0.517 | 0.600 | 0.200 |
| sign | none | training majority | 0.500 | 0.500 | 0.000 |
| sign | news_only | multinomial_nb | 0.500 | 0.500 | 0.000 |
| sign | news_only | logistic_regression | 0.500 | 0.500 | 0.000 |
| sign | question_and_news | multinomial_nb | 0.500 | 0.500 | 0.000 |
| sign | question_and_news | logistic_regression | 0.500 | 0.500 | 0.000 |
| sign | full_visible_context_and_news | multinomial_nb | 0.500 | 0.500 | 0.100 |
| sign | full_visible_context_and_news | logistic_regression | 0.475 | 0.475 | 0.100 |

## Leave-one-mechanism-class-out check

The authored families reuse 5 mechanism classes: biological_interaction (3 families), buffer_flow (7 families), competitive_allocation (7 families), queue_load (2 families), signal_interaction (1 families). They are not 20 independent mechanisms. This check holds every family from one class out together; the class labels never enter the classifier features.

| Task | Features | Classifier | Accuracy | Balanced accuracy | Whole-family correctness |
|---|---|---|---:|---:|---:|
| relevance | none | training majority | 0.667 | 0.500 | 0.000 |
| relevance | news_only | multinomial_nb | 0.667 | 0.500 | 0.000 |
| relevance | news_only | logistic_regression | 0.333 | 0.500 | 0.000 |
| relevance | question_and_news | multinomial_nb | 0.667 | 0.500 | 0.000 |
| relevance | question_and_news | logistic_regression | 0.333 | 0.500 | 0.000 |
| relevance | full_visible_context_and_news | multinomial_nb | 0.667 | 0.500 | 0.000 |
| relevance | full_visible_context_and_news | logistic_regression | 0.517 | 0.600 | 0.200 |
| sign | none | training majority | 0.500 | 0.500 | 0.000 |
| sign | news_only | multinomial_nb | 0.500 | 0.500 | 0.000 |
| sign | news_only | logistic_regression | 0.500 | 0.500 | 0.000 |
| sign | question_and_news | multinomial_nb | 0.500 | 0.500 | 0.000 |
| sign | question_and_news | logistic_regression | 0.500 | 0.500 | 0.000 |
| sign | full_visible_context_and_news | multinomial_nb | 0.475 | 0.475 | 0.100 |
| sign | full_visible_context_and_news | logistic_regression | 0.450 | 0.450 | 0.050 |

Relevance labels are positive/negative versus broken (2:1 class balance); sign uses positive versus negative (1:1). Whole-family correctness requires all three relevance labels, or both sign labels, to be correct. Labels are prediction targets and never feature text.

Identical news and question-plus-news guarantee no within-family discrimination. Full prompts may expose lexical relational cues. Success identifies an available lexical route; failure of these limited classifiers does not prove relational reasoning or eliminate more capable shortcuts.

Fixed settings: word unigram/bigram TF-IDF; MultinomialNB alpha=1; logistic regression C=1, balanced class weights, liblinear, maximum 2,000 iterations. Results are diagnostic, not confirmatory; materials and model inputs remain unchanged.

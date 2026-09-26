.DEFAULT_GOAL := help

OUTPUT ?=

.PHONY: help lint test review paper workshop submission

help:
	@printf '%s\n' \
	  'Repository review commands:' \
	  '  make review      Run lint checks and every maintained test' \
	  '  make lint        Run static checks over maintained source' \
	  '  make test        Run every maintained test suite' \
	  '  make paper       Build the canonical anonymous ICLR PDF\n  make workshop    Build the anonymous NeurIPS SocialAgent workshop PDF' \
	  '  make submission  Build the ICLR source archive (OUTPUT=path.zip optional)'

lint:
	./scripts/lint_active.sh

test:
	./scripts/test_active.sh

review: lint test

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

workshop:
	cd paper/socialagent2026 && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cp paper/socialagent2026/main.pdf paper/socialagent2026/forecasting-42.pdf

submission:
	cd paper && ./make_submission_zip.sh $(OUTPUT)

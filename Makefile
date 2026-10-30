.PHONY: all install test clean

all:
	python pipeline/generate.py
	python pipeline/score.py
	python pipeline/bounds.py
	python pipeline/coop_recourse.py

install:
	pip install -r requirements.txt

test:
	pytest -q

clean:
	rm -f results/*.csv results/*.json

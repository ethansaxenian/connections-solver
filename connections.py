import itertools
import math
from collections.abc import Iterable
from pathlib import Path

from gensim import downloader
from gensim.models import KeyedVectors

print("Enter words (ctrl + D to end):")
board = []
while True:
    try:
        line = input()
    except EOFError:
        break
    if len(line) > 0:
        board.append(line.strip().lower().replace(" ", "_").replace("'", ""))

DATASET = "word2vec-google-news-300"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

if not (DATA_DIR / f"{DATASET}.kv").exists():
    word_vectors: KeyedVectors = downloader.load(DATASET)
    word_vectors.save(str(DATA_DIR / f"{DATASET}.kv"))
else:
    word_vectors = KeyedVectors.load(str(DATA_DIR / f"{DATASET}.kv"), mmap="r")


def clean_words(board: list[str]) -> list[str]:
    words = []

    for word in board:
        if word in word_vectors:
            words.append(word)
        elif word.capitalize() in word_vectors:
            words.append(word.capitalize())
        elif word.upper() in word_vectors:
            words.append(word.upper())
        else:
            raise IndexError(f"{word} not found in dataset")

    return words


words = clean_words(board)


def choose(n: int, k: int) -> float:
    return math.factorial(n) / (math.factorial(n - k) * math.factorial(k))


def avg_similarity(group: tuple[str, str, str, str]) -> float:
    score = 0
    for comb in itertools.combinations(group, 2):
        score += word_vectors.similarity(*comb)
    return score / choose(4, 2)


def is_one_away(group: Iterable[str], checks: list[set[str]]) -> bool:
    for check in checks:
        if len(check.difference(group)) > 1:
            return False
    return True


not_it = set()

while len(words) > 0:
    one_away = []

    scores = sorted(
        ((comb, avg_similarity(comb)) for comb in itertools.combinations(words, 4)),
        key=lambda tup: tup[1],
        reverse=True,
    )

    for group, score in scores:
        if group in not_it:
            continue

        if not is_one_away(group, one_away):
            continue

        print()
        print([word.lower() for word in group], score)

        correct = input("correct? ").strip().lower()
        if correct == "y":
            for word in group:
                words.remove(word)
            break

        if correct == "1":
            one_away.append(set(group))

        not_it.add(group)

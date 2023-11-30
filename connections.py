import itertools
import math
from collections.abc import Iterable
from pathlib import Path

from gensim import downloader
from gensim.models import KeyedVectors

DATASET = "word2vec-google-news-300"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

GROUP_SIZE = 4

Group = tuple[str, ...]


def load_vectors(dataset: str) -> KeyedVectors:
    if not (DATA_DIR / f"{dataset}.kv").exists():
        word_vectors: KeyedVectors = downloader.load(dataset)
        word_vectors.save(str(DATA_DIR / f"{dataset}.kv"))
    else:
        word_vectors = KeyedVectors.load(str(DATA_DIR / f"{dataset}.kv"), mmap="r")

    return word_vectors


def get_words(word_vectors: KeyedVectors) -> list[str]:
    print("Enter words (ctrl + D to end):")

    words = []

    while True:
        try:
            line = input()
        except EOFError:
            break
        if len(line) > 0:
            word = (
                line.strip()
                .lower()
                .replace(" ", "_")
                .replace("'", "")
                .replace("-", "_")
            )
            if word in word_vectors:
                words.append(word)
            elif word.capitalize() in word_vectors:
                words.append(word.capitalize())
            elif word.upper() in word_vectors:
                words.append(word.upper())
            else:
                raise IndexError(f"{word} not found in dataset")

    if len(words) % GROUP_SIZE != 0:
        raise ValueError(f"Number of words must be a multiple of {GROUP_SIZE}")

    return words


def choose(n: int, k: int) -> float:
    return math.factorial(n) / (math.factorial(n - k) * math.factorial(k))


def avg_similarity(group: Group) -> float:
    score = 0
    for comb in itertools.combinations(group, 2):
        score += word_vectors.similarity(*comb)
    return score / choose(GROUP_SIZE, 2)


def is_one_away(group: Iterable[str], check: set[str]) -> bool:
    return len(check.difference(group)) == 1


def get_combinations(words: list[str]) -> Iterable[tuple[Group, float]]:
    return sorted(
        (
            (comb, avg_similarity(comb))
            for comb in itertools.combinations(words, GROUP_SIZE)
        ),
        key=lambda tup: tup[1],
        reverse=True,
    )


def get_next_match(
    words: list[str], previous_guesses: list[Group]
) -> tuple[Group, list[Group]]:
    guesses = []
    one_away = []
    far_away = []

    scores = get_combinations(words)

    for group, score in scores:
        if group in previous_guesses:
            continue

        if not all(is_one_away(group, check) for check in one_away):
            continue

        if any(is_one_away(group, check) for check in far_away):
            continue

        result = ""
        while result not in ("y", "n", "2+", "1"):
            print()
            print(
                [word.upper().replace("_", " ").replace("-", " ") for word in group],
                score,
            )

            result = (
                input(
                    "'y' (correct), '1' (one away), '2+' (two or more away), or 'n' (incorrect): "
                )
                .strip()
                .lower()
            )

        if result == "y":
            return group, guesses

        if result == "1":
            one_away.append(set(group))

        if result == "2+":
            far_away.append(set(group))

        guesses.append(group)

    return tuple(), guesses


if __name__ == "__main__":
    word_vectors = load_vectors(DATASET)
    words = get_words(word_vectors)

    guesses = []

    while len(words) > 0:
        group, new_guesses = get_next_match(words, guesses)

        guesses.extend(new_guesses)

        for word in group:
            words.remove(word)

"""Canonical denotations for explicit dates and simple durations in LongMemEval."""

from datetime import datetime
from fractions import Fraction
import re

WORD_NUMBERS = {
    word: i for i, word in enumerate(("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                                      "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                                      "sixteen", "seventeen", "eighteen", "nineteen"))
}
TENS = dict(zip(("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"), range(20, 100, 10)))
TIME = {"hour": 1, "day": 24, "week": 168}
QUANTITY = re.compile(r"(?P<number>\d+(?:\.\d+|/\d+)?|[a-z]+(?:[ -][a-z]+)??)"
                      r"(?:\s+(?P<unit>hours?|days?|weeks?|months?|years?))?")
ASK = re.compile(r"\bhow many (?:calendar )?(hours?|days?|weeks?|months?|years?)\b")
DELAY_QUESTION = re.compile(r"\bhow long\b")
INEXACT = re.compile(r"\b(?:about|around|approximately|roughly|at least|at most|more than|less than|or)\b")
COUNTED = "|".join((*WORD_NUMBERS, *TENS, "a", "an"))
EXPLICIT_DURATION = re.compile(
    rf"\b(?:\d+(?:\.\d+|/\d+)?|(?:{COUNTED})(?:[ -](?:{COUNTED}))?) +(?:hours?|days?|weeks?|months?|years?)\b"
)
MONTH_PATTERN = re.compile(r"[A-Za-z]+ +\d{4}")
WEEKDAY = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}


def quantity(text):
    clean = " ".join(text.lower().strip(" .").split())
    match = QUANTITY.fullmatch(clean)
    if match is None:
        return None
    word = match["number"]
    unit = match["unit"].removesuffix("s") if match["unit"] else None
    if word[0].isdigit():
        try:
            number = Fraction(word)
        except (ValueError, ZeroDivisionError):
            return None
        return number, unit
    if word in ("a", "an"):
        return (Fraction(1), unit) if unit else None
    if word in WORD_NUMBERS:
        return Fraction(WORD_NUMBERS[word]), unit
    parts = word.replace("-", " ").split()
    if parts[0] in TENS and len(parts) == 1:
        return Fraction(TENS[parts[0]]), unit
    if len(parts) == 2 and parts[0] in TENS and 1 <= WORD_NUMBERS.get(parts[1], 0) <= 9:
        return Fraction(TENS[parts[0]] + WORD_NUMBERS[parts[1]]), unit
    return None


def duration(text, question):
    expected = question.lower()
    inferred = ASK.search(expected)
    found = quantity(text)
    if found is None and (inferred or DELAY_QUESTION.search(expected)):
        clean = text.casefold()
        spans = EXPLICIT_DURATION.findall(clean) if not INEXACT.search(clean) else []
        if len(spans) == 1:
            found = quantity(spans[0])
    if found is None:
        return None
    value, unit = found
    if unit is None and inferred:
        unit = inferred[1].removesuffix("s")
    if unit in TIME:
        return "hours", value * TIME[unit]
    return (unit, value) if unit in ("month", "year") else None


def date(text):
    # Exact month/year or whole calendar date. An implicit year is left unresolved.
    normalized = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", text.strip(" ."), flags=re.I)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"):
        try:
            return "date", datetime.strptime(normalized, fmt).date().isoformat()
        except ValueError:
            continue
    if MONTH_PATTERN.fullmatch(normalized) or re.fullmatch(r"\d{4}-\d{2}", normalized):
        for fmt in ("%B %Y", "%b %Y", "%Y-%m"):
            try:
                return "month_year", datetime.strptime(normalized, fmt).strftime("%Y-%m")
            except ValueError:
                continue
    return ("weekday", normalized.casefold()) if normalized.casefold() in WEEKDAY else None


def denotation(text, question):
    return date(text) or duration(text, question)

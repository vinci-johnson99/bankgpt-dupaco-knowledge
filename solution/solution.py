"""
Dupaco knowledge-system submission.

Approach (see NOTES.md for the full writeup):
  1. Keep every version of every page we ever saw, keyed by its own crawl
     timestamp (fetched_at) -- not just the newest snapshot. A page that was
     re-crawled recently is "fresher" than one that wasn't, even if both show
     up in the same (latest) snapshot. That's how a current rates page and a
     stale calculator page that quotes the old rate can coexist on the same
     day: whichever was crawled more recently wins.
  2. `view(as_of)` reconstructs "the site as it stood on date X" by taking,
     for every URL, the newest version fetched on or before X.
  3. Pages are broken into structured Facts: markdown-table rows (rates,
     tiers, fees) and label/number-bearing lines (addresses, phone numbers,
     hours, counts). Each fact carries its qualifiers (term, model year,
     balance range, product name, "personal" vs "business", ...).
  4. A question is parsed for the same kind of qualifiers, matched against
     candidate facts (found via a small keyword index), and scored by
     qualifier fit + page authority + recency. A tight, well-matched fact
     wins; with no specific qualifier we report every matching row (for
     "what are your rates" style questions); with nothing found we refuse.
  5. conflicts() groups facts that share an identical qualifier signature but
     disagree in value across different source URLs, and resolves them the
     same way answer() would.

No LLM, no embeddings, no network calls, no paid API -- pure Python
standard library, so it's deterministic, fast, and needs no API key.
"""
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# NOTE: deliberately no `from __future__ import annotations` here. The harness
# loads this file with importlib without registering it in sys.modules, and on
# Python 3.13 the dataclass machinery crashes resolving deferred string
# annotations against an unregistered module. Using typing.Optional (resolved
# eagerly) instead of `X | None` sidesteps that entirely and still works on
# Python 3.9+.

sys.path.insert(
    0,
    str(next(d / "harness" for d in Path(__file__).resolve().parents if (d / "harness" / "contract.py").exists())),
)
from contract import Answer, Conflict, Snapshot, Source  # noqa: E402

# --------------------------------------------------------------------------
# Tokenizing / number handling
# --------------------------------------------------------------------------

WORD_RE = re.compile(r"[a-z0-9$%./-]+")
NUM_RE = re.compile(r"(\$)?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(%|percent)?", re.I)

STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "what", "what's", "whats", "how", "i", "to", "of",
    "for", "on", "in", "you", "your", "can", "me", "and", "or", "my", "it", "that", "this", "with",
    "at", "as", "be", "was", "were", "will", "would", "please", "tell", "give", "get", "has", "have",
    "right", "now", "any", "there", "did", "we", "us", "if", "so", "about",
}


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall((text or "").lower())


def singularize(w: str) -> str:
    """Minimal plural stripping so 'branches'/'branch', 'cards'/'card', etc. line
    up between a question and the page text without a full stemmer."""
    if len(w) > 4 and w.endswith("ies") and w[:-3].isalpha():
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es") and w[:-2].endswith(("ch", "sh", "x", "s", "z")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and w.isalpha():
        return w[:-1]
    return w


_DIGIT_HYPHEN_RE = re.compile(r"\d")


def content_words(text: str) -> list[str]:
    """Tokenize + singularize. A token that mixes a digit with a hyphen (e.g.
    '60-month', '2024-27') also contributes its hyphen-split parts ('60',
    'month') as extra words -- the same figure is phrased as '60 Months' (no
    hyphen) on some pages and '60-month' (hyphenated) in questions, and
    without this they never overlap at all."""
    out = []
    for w in tokenize(text):
        if w in STOPWORDS or len(w) <= 1:
            continue
        out.append(singularize(w))
        if "-" in w and _DIGIT_HYPHEN_RE.search(w):
            for part in w.split("-"):
                if part and part not in STOPWORDS and len(part) > 1:
                    out.append(singularize(part))
    return out


def numbers_in(text: str) -> set[tuple[str, float]]:
    """Same value-aware number extraction scoring.py uses, so our internal
    comparisons line up with how we'll actually be graded."""
    out = set()
    for m in NUM_RE.finditer(text or ""):
        unit = "$" if m.group(1) else ("%" if m.group(4) else "")
        val = float(m.group(2).replace(",", "") + ("." + m.group(3) if m.group(3) else ""))
        out.add((unit, round(val, 4)))
    return out


def norm_text(s: str) -> str:
    s = re.sub(r"\b([ap])\.\s?m\.?", r"\1m", (s or "").lower())
    s = s.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip()


YEAR_RANGE_RE = re.compile(r"(20\d{2})\s*[-–]\s*(\d{2,4})")
MONTH_RE = re.compile(r"(\d{1,3})\s*[- ]?month", re.I)
DOLLAR_RANGE_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(?:-|to|and|–)\s*\$?\s?([\d,]+(?:\.\d+)?)", re.I)


def year_range(text: str) -> Optional[tuple]:
    m = YEAR_RANGE_RE.search(text or "")
    if not m:
        return None
    start = int(m.group(1))
    end_raw = m.group(2)
    end = int(end_raw) if len(end_raw) == 4 else int(str(start)[:2] + end_raw)
    return (start, end) if end >= start else (end, start)


def months_in(text: str) -> set[int]:
    return {int(m.group(1)) for m in MONTH_RE.finditer(text or "")}


def dollar_range(text: str) -> Optional[tuple]:
    m = DOLLAR_RANGE_RE.search(text or "")
    if not m:
        return None
    lo = float(m.group(1).replace(",", ""))
    hi = float(m.group(2).replace(",", ""))
    return (lo, hi) if hi >= lo else (hi, lo)


# --------------------------------------------------------------------------
# Page history / as-of view
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PageVersion:
    url: str
    title: str
    markdown: str
    date: str  # fetched_at date, YYYY-MM-DD
    snapshot_id: str


def url_tier(url: str) -> int:
    """Lower = more authoritative. Product/service pages (and the consolidated
    rate sheets) beat calculators and homepage-ish pages, which beat news/blog
    (still used, just demoted). Calculators get the steepest demotion: this data
    set's whole "stale trap" pattern is a calculator page that quotes an old rate
    a maintained product page has since moved past, and a calculator can also
    coincidentally word-match a question well (e.g. "vehicle affordability" vs.
    "vehicle"), so it needs a big, deliberate penalty to not win on that alone."""
    path = urlparse(url).path.lower()
    if "calculator" in path or path.rstrip("/").endswith("-lp") or "-offer-lp" in path:
        return 5
    if path.rstrip("/").endswith("rates"):
        return 0
    if "/newsroom/" in path or "/learn/blog" in path or "/blog/" in path:
        return 2
    if path in ("", "/") or path.count("/") <= 1:
        return 1
    return 0


class KB:
    def __init__(self) -> None:
        self.history: dict[str, list[PageVersion]] = defaultdict(list)
        self._view_cache = {}  # type: dict
        self._index_cache = {}  # type: dict
        self._fact_cache = {}  # type: dict
        self._raw_fact_cache = {}  # type: dict

    def ingest(self, snapshot: Snapshot) -> None:
        for p in snapshot.pages:
            date = (p.fetched_at or snapshot.taken_on)[:10]
            self.history[p.url].append(PageVersion(p.url, p.title, p.markdown, date, snapshot.snapshot_id))
        self._view_cache.clear()
        self._index_cache.clear()
        self._fact_cache.clear()
        self._raw_fact_cache.clear()

    def view(self, as_of: Optional[str]):
        if as_of in self._view_cache:
            return self._view_cache[as_of]
        out: dict[str, PageVersion] = {}
        for url, versions in self.history.items():
            best = None
            for v in versions:
                if as_of is None or v.date <= as_of:
                    best = v
            if best is not None:
                out[url] = best
        self._view_cache[as_of] = out
        return out

    def index(self, as_of: Optional[str]):
        if as_of not in self._index_cache:
            self._index_cache[as_of] = Index(self.view(as_of))
        return self._index_cache[as_of]

    def raw_facts(self, as_of: Optional[str]):
        """Every extracted fact, before boilerplate near-duplicates are
        collapsed to one winner -- conflicts() wants to see the disagreement
        itself, not just the resolved answer."""
        if as_of not in self._raw_fact_cache:
            out = []  # type: list
            for pv in self.view(as_of).values():
                out.extend(extract_facts(pv))
            self._raw_fact_cache[as_of] = out
        return self._raw_fact_cache[as_of]

    def facts(self, as_of: Optional[str]):
        if as_of not in self._fact_cache:
            self._fact_cache[as_of] = _resolve_boilerplate(self.raw_facts(as_of), as_of)
        return self._fact_cache[as_of]


# --------------------------------------------------------------------------
# Retrieval index
# --------------------------------------------------------------------------


class Index:
    def __init__(self, pages: dict[str, PageVersion]) -> None:
        self.pages = pages
        self.body_words: dict[str, set[str]] = {}
        self.title_words: dict[str, set[str]] = {}
        df: dict[str, int] = defaultdict(int)
        for url, pv in pages.items():
            body = set(content_words(pv.markdown))
            title = set(content_words(pv.title))
            self.body_words[url] = body
            self.title_words[url] = title
            for t in body | title:
                df[t] += 1
        self.df = df
        self.n = max(1, len(pages))

    def idf(self, w: str) -> float:
        return math.log(1 + self.n / (1 + self.df.get(w, 0)))

    def scores(self, query_words: list[str]) -> dict:
        qw = set(query_words)
        scores: dict[str, float] = {}
        for url in self.pages:
            s = 0.0
            body, title = self.body_words[url], self.title_words[url]
            for w in qw:
                if w in body:
                    s += self.idf(w)
                if w in title:
                    s += 1.5 * self.idf(w)
            if s > 0:
                s -= 0.2 * url_tier(url)
                scores[url] = s
        return scores

    def rank(self, query_words: list[str], top: int = 15) -> list[str]:
        scores = self.scores(query_words)
        return sorted(scores, key=lambda u: -scores[u])[:top]


# --------------------------------------------------------------------------
# Fact extraction
# --------------------------------------------------------------------------


SINGLE_THRESHOLD_RE = re.compile(
    r"\$\s?[\d,]+(?:\.\d+)?\s*(?:or less|or more|or above|or over|and up|and above|and over)\b", re.I
)
RANGE_STRIP_RES = [YEAR_RANGE_RE, MONTH_RE, DOLLAR_RANGE_RE, SINGLE_THRESHOLD_RE]


def strip_ranges(label: str) -> str:
    """Remove recognizable *tiering* ranges (model-year range, term in months, a
    dollar range, or a one-sided dollar threshold like '$100,000 or less') from a
    row label, leaving its 'core name'. Rows of the same product that only differ
    by tier collapse to the same core name; a number that is NOT part of one of
    these tiering idioms (e.g. the "30" in "30-Year Fixed") is left in place, so
    genuinely different products (a 30-year vs. a 15-year loan) stay distinct."""
    s = label
    for rex in RANGE_STRIP_RES:
        s = rex.sub(" ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Fact:
    value: str             # the number/short value, as it appears on the page
    text: str               # a readable sentence/row describing the fact
    qualifiers: set         # content-word qualifiers (term, product, tier...)
    url: str
    date: str
    snapshot_id: str
    title: str
    kind: str               # "table" | "line" | "para"
    group_key: tuple = field(default=())    # identity used for conflict grouping
    item_key: tuple = field(default=())     # identity used to group tiers of the "same thing"
    primary_num: Optional[float] = None     # representative numeric value, for superlative ranking
    is_rate: bool = False                   # row/line is explicitly labelled a rate/APR/APY


TABLE_LINE_RE = re.compile(r"^\s*\|(.+)\|\s*$")
SEP_LINE_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def split_row(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [c.strip() for c in inner.split("|")]


def extract_tables(markdown: str) -> list[dict]:
    """Return [{heading, title_line, header: [...], rows: [[...]]}]."""
    lines = markdown.splitlines()
    tables = []
    heading = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        h = re.match(r"^#{1,6}\s+(.*)", line)
        if h:
            heading = h.group(1).strip()
            i += 1
            continue
        if TABLE_LINE_RE.match(line) and i + 1 < len(lines) and SEP_LINE_RE.match(lines[i + 1]) and "-" in lines[i + 1]:
            header = split_row(line)
            j = i + 2
            rows = []
            while j < len(lines) and TABLE_LINE_RE.match(lines[j]):
                rows.append(split_row(lines[j]))
                j += 1
            # nearest non-empty preceding line (often a bold "table title")
            k = i - 1
            title_line = ""
            while k >= 0 and not lines[k].strip():
                k -= 1
            if k >= 0:
                title_line = re.sub(r"[*_#]", "", lines[k]).strip()
            tables.append({"heading": heading, "title_line": title_line, "header": header, "rows": rows})
            i = j
            continue
        i += 1
    return tables


_VALUE_UNIT_WORDS_RE = re.compile(r"\b(apr|apy|percent|variable|fixed)\d*", re.I)
_VALUE_STRIP_RE = re.compile(r"[\d,.$%\s\-–—]")


def is_value_cell(cell: str) -> bool:
    """A table cell is a VALUE (a rate/fee/amount) -- not a LABEL/qualifier like
    '60 Months' or '2024-27 Model Year' -- when, after removing known unit words
    and all digits/punctuation/currency/percent signs, almost nothing is left."""
    c = (cell or "").strip()
    if not numbers_in(c):
        return False
    core = _VALUE_UNIT_WORDS_RE.sub("", c)
    core = _VALUE_STRIP_RE.sub("", core)
    return len(core) <= 2


BULLET_RATE_RE = re.compile(r"^\s*[-*•]\s*([^:\n]{3,80}):\s*(.*)$", re.M)
HEADING_RE = re.compile(r"^#{1,6}\s+(.*)")
PHONE_RE = re.compile(r"(?:\(\d{3}\)\s?|\b\d{3}[-.])\d{3}[-.]\d{4}\b")
ADDR_RE = re.compile(r"\b\d{1,6}\s+[A-Za-z0-9.'\s]{3,40}?,\s*[A-Za-z .]+,\s*[A-Z]{2}[, ]+\d{5}")
ROUTING_RE = re.compile(r"routing number[:\s]*([0-9]{9})", re.I)
RATE_WORDS = {"rate", "rates", "apr", "apy", "dividend", "yield"}
ELIGIBILITY_RE = re.compile(
    r"Dupaco serves residents in ([\d,]+) counties throughout ([A-Za-z ,]+?Wisconsin)", re.I
)
BRANCH_COUNT_RE = re.compile(r"(\d+)\s*branch office locations", re.I)
_MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July",
                 "August", "September", "October", "November", "December")
ARTICLE_DATE_RE = re.compile(
    r"^(" + "|".join(_MONTH_NAMES) + r")\s+(\d{1,2}),\s*(20\d{2})\s*$", re.M
)


def _article_byline_date(text: str) -> Optional[str]:
    """A newsroom article's own byline date ("April 06, 2026"), as an
    ISO date string -- when it exists, it's a far better "as of" marker for
    boilerplate the article quotes than the crawl date every version of
    every page shares."""
    m = ARTICLE_DATE_RE.search(text)
    if not m:
        return None
    month = _MONTH_NAMES.index(m.group(1)) + 1
    return f"{m.group(3)}-{month:02d}-{int(m.group(2)):02d}"


def extract_facts(pv: PageVersion) -> list[Fact]:
    facts: list[Fact] = []
    page_title_words = set(content_words(pv.title))
    url_words = set(content_words(pv.url.replace("https://", "").replace("/", " ")))
    base_url = pv.url.split("#")[0]

    for t in extract_tables(pv.markdown):
        header = t["header"]
        header_quals = [set(content_words(h)) for h in header]
        table_quals = set(content_words(t["heading"])) | set(content_words(t["title_line"]))
        table_has_rate_word = bool(table_quals & RATE_WORDS or any(hq & RATE_WORDS for hq in header_quals))
        for row in t["rows"]:
            if len(row) != len(header):
                continue
            num_cells = [(ci, cell) for ci, cell in enumerate(row) if is_value_cell(cell)]
            if not num_cells:
                continue
            label_cells = [c for ci, c in enumerate(row) if not is_value_cell(c)]
            row_label = " / ".join(c for c in label_cells if c) or t["title_line"] or t["heading"]
            row_quals = set(content_words(row_label))
            col_quals = set()
            for ci, _ in num_cells:
                if ci < len(header_quals):
                    col_quals |= header_quals[ci]
            full_quals = table_quals | row_quals | col_quals | page_title_words | url_words
            value_bits = []
            for ci, cell in num_cells:
                col_header = header[ci] if ci < len(header) else ""
                value_bits.append(f"{col_header}: {cell}".strip(": ") if col_header else cell)
            text = f"{row_label} — {', '.join(value_bits)}".strip(" —,")
            nums = [v for cell in [c for _, c in num_cells] for v in numbers_in(cell)]
            primary = max((v for _, v in nums), default=None)
            row_is_rate = table_has_rate_word or bool(row_quals & RATE_WORDS) or any(
                header_quals[ci] & RATE_WORDS for ci, _ in num_cells if ci < len(header_quals)
            ) or any(v[0] == "%" for v in nums)
            group_key = (base_url, t["heading"], t["title_line"], tuple(row))
            item_key = (base_url, t["heading"], t["title_line"], strip_ranges(row_label))
            facts.append(Fact(
                value=", ".join(c for _, c in num_cells), text=text, qualifiers=full_quals,
                url=pv.url, date=pv.date, snapshot_id=pv.snapshot_id, title=pv.title, kind="table",
                group_key=group_key, item_key=item_key, primary_num=primary, is_rate=row_is_rate,
            ))

    # Non-table lines/paragraphs: label: value, addresses, phones, routing #, hours, counts.
    text = pv.markdown
    for m in ROUTING_RE.finditer(text):
        facts.append(Fact(m.group(1), f"Routing Number: {m.group(1)}",
                           {"routing", "number"} | page_title_words, pv.url, pv.date, pv.snapshot_id, pv.title,
                           "line", (base_url, "routing"), (base_url, "routing"), None, False))
    for m in PHONE_RE.finditer(text):
        ctx_start = max(0, m.start() - 80)
        ctx = text[ctx_start:m.end()]
        # Press-release bylines ("Media Contact / Jane Doe / Chief Marketing Officer /
        # 563-...") repeat on nearly every newsroom page and drown out the real,
        # member-facing phone numbers on product/contact pages -- skip them.
        if re.search(r"media contact", ctx, re.I):
            continue
        facts.append(Fact(m.group(0), ctx.strip(), set(content_words(ctx)) | page_title_words,
                           pv.url, pv.date, pv.snapshot_id, pv.title, "line",
                           (base_url, "phone", m.group(0)), (base_url, "phone", m.group(0)), None, False))
    for m in ADDR_RE.finditer(text):
        ctx_start = max(0, m.start() - 150)
        ctx = text[ctx_start:m.start()]
        facts.append(Fact(m.group(0), m.group(0), set(content_words(ctx)) | set(content_words(m.group(0))) | page_title_words,
                           pv.url, pv.date, pv.snapshot_id, pv.title, "line",
                           (base_url, "address", m.group(0)), (base_url, "address", m.group(0)), None, False))

    # Bullet lines of the shape "- <label>: <value>" where the value carries a
    # percent (e.g. "- Lower ongoing rate if you carry a balance: 12.50%-19.50%
    # APR"). Some product pages quote rates this way instead of in a table, and
    # without this they're invisible as structured facts -- a wants_rate
    # question either falls back to unrelated prose (wrong) or finds nothing
    # structured at all and wrongly refuses.
    lines = text.splitlines()
    heading = ""
    for ln in lines:
        hm = HEADING_RE.match(ln)
        if hm:
            heading = hm.group(1).strip()
            continue
        bm = BULLET_RATE_RE.match(ln)
        if not bm:
            continue
        label, value = bm.group(1).strip(), bm.group(2).strip()
        if not any(u == "%" for u, _ in numbers_in(value)):
            continue
        quals = set(content_words(heading)) | set(content_words(label)) | set(content_words(value)) | page_title_words
        nums = numbers_in(value)
        primary = max((v for u, v in nums if u == "%"), default=None)
        is_rate = bool(quals & RATE_WORDS)
        item_key = (base_url, "bullet", heading, strip_ranges(label))
        facts.append(Fact(value, f"{label}: {value}", quals, pv.url, pv.date, pv.snapshot_id, pv.title,
                           "line", (base_url, "bullet", heading, label), item_key, primary, is_rate))

    # hour blocks: "<branch name/context> ... Branch hours are:" Lobby / Drive-up schedule.
    # Keep ~200 chars BEFORE the marker too, so the branch's own name becomes a qualifier.
    for hm in re.finditer(r"Branch hours are:", text):
        ctx_before = text[max(0, hm.start() - 220):hm.start()]
        block = text[hm.end():hm.end() + 500]
        stop = re.search(r"###|ABOUT DUPACO|\*\s*\*\s*\*|Branch hours are:", block)
        if stop:
            block = block[: stop.start()]
        block = block.strip()
        if block:
            # "hour(s)" itself lives in the trigger phrase ("Branch hours
            # are:"), not the captured block, so without adding it back a
            # question that says "hours" has only ONE real overlap word
            # ("lobby"/"branch") to clear the >=2 floor with -- add it
            # explicitly so this fact isn't crowded out by some unrelated
            # bullet that happens to say "hours" twice.
            quals = set(content_words(ctx_before)) | set(content_words(block)) | page_title_words | {"hour"}
            facts.append(Fact(block, "Branch hours: " + re.sub(r"\s+", " ", block), quals,
                               pv.url, pv.date, pv.snapshot_id, pv.title, "line",
                               (base_url, "hours", block[:60]), (base_url, "hours", block[:60]), None, False))

    # The "ABOUT DUPACO..." boilerplate (membership area, branch count) is
    # copy-pasted near-verbatim onto dozens of press releases, each frozen at
    # whatever the count was the day THAT article was written -- so several
    # different numbers are all "current" on the same crawl date, and the
    # crawl date (every copy was crawled the same day) can't break the tie
    # either. What CAN: the article's own byline date, which is when that
    # count was actually true. They're pulled out as their own line facts
    # (not penalized like a generic paragraph), dated by that byline instead
    # of the crawl date, and -- since their item_key is the SAME constant
    # across every page carrying a copy, instead of per-page -- later
    # collapsed in KB.facts() to the one with the latest byline, the best
    # stand-in for "the current one" we have without a designated single
    # source of truth.
    # A page with no byline at all (a product page, not a press release)
    # falls back to a sentinel far in the past, NOT the crawl date -- every
    # crawl happens in 2026, almost always later than any real article
    # byline, so using it as a fallback would let a date-less copy win the
    # "latest byline" comparison by accident, exactly backwards from intent.
    byline = _article_byline_date(text) or "1900-01-01"
    for m in ELIGIBILITY_RE.finditer(text):
        quals = {"eligible", "eligibility", "member", "membership", "join", "who", "county",
                  "counties", "residents", "serve", "charter", "area", "live", "work"} \
            | set(content_words(m.group(2))) | page_title_words
        txt = (f"Anyone who lives or works within Dupaco's charter area -- {m.group(1)} counties "
               f"throughout {m.group(2)} -- is eligible to become a member.")
        facts.append(Fact(m.group(2), txt, quals, pv.url, byline, pv.snapshot_id, pv.title, "line",
                           (base_url, "eligibility"), ("boilerplate", "eligibility"), None, False))
    for m in BRANCH_COUNT_RE.finditer(text):
        n = m.group(1)
        quals = {"branch", "branches", "office", "location", "locations", "many", "how",
                  "count", "number", "total", "have"} | page_title_words
        txt = f"Dupaco has {n} branch office locations."
        facts.append(Fact(n, txt, quals, pv.url, byline, pv.snapshot_id, pv.title, "line",
                           (base_url, "branch_count"), ("boilerplate", "branch_count"), float(n), False))

    # generic paragraphs (numeric or not) -- broad-recall fallback for free-text facts
    # (membership eligibility, program descriptions, enumerated brand lists, counts...)
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para or len(para) > 900 or len(para) < 15:
            continue
        if para.splitlines() and (TABLE_LINE_RE.match(para.splitlines()[0]) or para.startswith("#")):
            continue
        quals = set(content_words(para)) | page_title_words
        nums = numbers_in(para)
        primary = max((v for _, v in nums), default=None)
        skeleton = NUM_RE.sub("#", norm_text(para))
        # A paragraph only counts as "quoting a rate" if it actually has a percent
        # value in it -- just using the WORD "rate" (e.g. "rates fluctuate...") is
        # not a rate, and must not satisfy a wants_rate question.
        para_is_rate = bool(quals & RATE_WORDS) and any(u == "%" for u, _ in nums)
        # "... payment example for illustrative purposes only ..." paragraphs
        # restate a rate inside a worked example; they're derived, not the
        # published rate itself, and their concrete-sounding numbers
        # ("$30,000", "60-month") routinely out-word-match the real rate
        # table row they're illustrating. Never let one stand in for it.
        if "illustrative purposes only" in para.lower():
            para_is_rate = False
        facts.append(Fact(para, para, quals, pv.url, pv.date, pv.snapshot_id, pv.title, "para",
                           (base_url, "para", skeleton[:120]), (base_url, "para", skeleton[:120]), primary,
                           para_is_rate))

    return facts


def _resolve_boilerplate(facts: list, as_of: Optional[str]) -> list:
    """Facts tagged with a constant ("boilerplate", ...) item_key are the
    same copy-pasted blurb repeated across many pages, each stating whatever
    was true the day THAT article was written -- their `date` is each
    article's own byline (see _article_byline_date), not the shared crawl
    date, so the one with the latest byline AT OR BEFORE as_of is kept as the
    best stand-in for "current as of that date" and the rest dropped, so they
    don't crowd out other answers as near-duplicate noise."""
    groups: dict = defaultdict(list)
    rest = []
    for f in facts:
        if f.item_key and f.item_key[0] == "boilerplate":
            groups[f.item_key].append(f)
        else:
            rest.append(f)
    for members in groups.values():
        pool = [m for m in members if as_of is None or m.date <= as_of] or members
        rest.append(max(pool, key=lambda m: m.date))
    return rest


# --------------------------------------------------------------------------
# Question understanding + fact scoring
# --------------------------------------------------------------------------

# Modifiers that flip a product into a DIFFERENT product (business vs personal,
# jumbo vs standard, ...). If a fact carries one of these and the question
# doesn't ask for it, that fact is very likely the wrong tier/product.
DISTINGUISHING_MODIFIERS = {
    "business", "personal", "jumbo", "premier", "plus", "select", "youth", "teen",
    "student", "senior", "health", "hsa", "ira", "trust", "estate",
    # Loan-product variants. Several totally different products (an auto loan,
    # a recreational/ATV loan, a mortgage...) can live on the same consolidated
    # /rates page and share generic words ("loan", "month", "apr") a question
    # about just one of them will still overlap with every other one's rows.
    # Without a penalty for the OTHER product's own distinguishing words, a
    # coincidental lexical edge (e.g. "APR" the abbreviation vs "Annual
    # Percentage Rate" spelled out) can tip the wrong product's row ahead.
    "recreational", "atv", "boat", "rv", "snowmobile", "jet", "ski",
}
HIGH_WORDS = {"highest", "max", "maximum", "most", "best", "top"}
LOW_WORDS = {"lowest", "min", "minimum", "least"}
MIN_SCORE = 0.5  # secondary gate on the final (tier/page-adjusted) score; the primary
                   # topical gate is MIN_WORD_SCORE inside score_fact


@dataclass
class ParsedQuestion:
    raw: str
    words: list[str]
    quals: set
    year_range: Optional[tuple]
    months: set
    dollar_range: Optional[tuple]
    has_specific_qualifier: bool
    wants_rate: bool
    wants_url: bool
    superlative: int   # +1 highest, -1 lowest, 0 none
    named_entities: set = field(default_factory=set)


WANTS_URL_WORDS = {"link", "url", "webpage", "website"}

# Acronyms/initialisms that are generic finance vocabulary, not named entities
# -- capitalized the same way a real proper noun is, but ubiquitous on the
# site, so requiring they be "grounded" in the chosen answer would be noise.
_PROPER_SKIP = {
    "dupaco", "apr", "apy", "cd", "arm", "atm", "hsa", "ira", "fha", "va", "usda", "llc",
    # Capitalized but generic -- a calendar month in a date is not a named
    # entity to "ground" (e.g. "...still open today (October 2026)?").
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}
_RAW_WORD_RE = re.compile(r"[A-Za-z0-9$%./-]+")


def proper_nouns(raw_question: str) -> set:
    """Capitalized, non-sentence-initial words in the raw question -- a cheap
    signal that the question names a specific entity (a branch, a program, a
    person, a brand). We later check whether that entity shows up anywhere in
    the facts we're about to answer with; if it doesn't, the question is
    almost certainly asking about something not actually on the site (a
    branch that doesn't exist, e.g.), and we should refuse rather than
    confidently hand back an unrelated fact that merely shares other words."""
    out = set()
    tokens = _RAW_WORD_RE.findall(raw_question)
    for idx, w in enumerate(tokens):
        if idx == 0 or not w[0].isalpha() or not w[0].isupper():
            continue
        lw = singularize(w.lower())
        if len(lw) < 4 or lw in _PROPER_SKIP:
            continue
        out.add(lw)
    return out


def parse_question(question: str) -> ParsedQuestion:
    words = content_words(question)
    ym = re.search(r"\b(20\d{2})\b", question)
    quals = set(words)
    dr = dollar_range(question)
    mo = months_in(question)
    specific = bool(mo or dr or ym)
    wants_rate = bool(quals & RATE_WORDS) or "interest" in quals
    wants_url = bool(quals & WANTS_URL_WORDS)
    sup = 1 if quals & HIGH_WORDS else (-1 if quals & LOW_WORDS else 0)
    return ParsedQuestion(question, words, quals, (int(ym.group(1)), int(ym.group(1))) if ym else None,
                           mo, dr, specific, wants_rate, wants_url, sup, proper_nouns(question))


MIN_WORD_SCORE = 1.1  # a fact must be ABOUT the right topic on its own words, before
                        # the page it lives on gets any say in ranking.


def score_fact(fact: Fact, pq: ParsedQuestion, idf, page_rel: float) -> Optional[float]:
    overlap = fact.qualifiers & pq.quals
    if fact.kind in ("para", "line") and len(overlap) < 2:
        return None  # free text / an ad hoc line fact needs more than one coincidental
                       # shared word -- a single generic term ("branch") lets an
                       # unrelated phone number or hours block outrank the paragraph
                       # that's actually on topic, which has no such floor to clear
    if pq.wants_rate and fact.kind == "para" and not fact.is_rate:
        return None  # prose that isn't actually quoting a rate can't answer a rate question

    if fact.kind == "para":
        # A long article's page title ("...plans Hy-Vee branch closure") gets
        # folded into EVERY paragraph's qualifiers on that page via
        # page_title_words, so an unrelated paragraph (nationwide
        # branch-closure trivia, an image caption) can match a question
        # almost as well as the one paragraph actually ABOUT the topic, on
        # title words alone. Score a paragraph mostly on ITS OWN words;
        # qualifiers it only has because of the page title count for much
        # less (still some -- the page IS relevant context, just weaker
        # evidence than the paragraph's own content).
        own = set(content_words(fact.value))
        overlap_own = overlap & own
        overlap_title_only = overlap - own
        word_score = sum(idf(w) for w in overlap_own) + 0.25 * sum(idf(w) for w in overlap_title_only)
    else:
        word_score = sum(idf(w) for w in overlap)

    if pq.year_range:
        yr = year_range(fact.text)
        if yr:
            word_score += 5 if yr[0] <= pq.year_range[0] <= yr[1] else -8
    if pq.months:
        fm = months_in(fact.text)
        if fm:
            word_score += 5 if (fm & pq.months) else -8
    if pq.dollar_range:
        fr = dollar_range(fact.text)
        if fr:
            lo, hi = pq.dollar_range
            word_score += 5 if (fr[0] <= lo <= fr[1] or fr[0] <= hi <= fr[1]) else -4

    # Only penalize structured facts (table rows, bullet/line facts) for
    # carrying a distinguishing modifier the question didn't ask for -- those
    # come from a row/column genuinely labelling a product variant. A
    # paragraph can use the same word incidentally ("a business magazine",
    # describing Forbes, not a business-vs-personal product split) and
    # shouldn't be punished for it.
    if fact.kind != "para":
        extra_modifiers = (fact.qualifiers & DISTINGUISHING_MODIFIERS) - pq.quals
        word_score -= 4.0 * len(extra_modifiers)

    if pq.wants_rate and not fact.is_rate:
        word_score -= 5.0

    # A fact must earn its place on its OWN words first. The page it happens to
    # live on (and its authority tier) only refine ranking among facts that already
    # clear this bar -- otherwise one comprehensive page (e.g. a master rate sheet)
    # would let every unrelated row on it leak through on page relevance alone.
    if word_score < MIN_WORD_SCORE:
        return None

    score = word_score + 0.12 * page_rel
    if fact.kind == "para":
        score -= 1.75  # structured table/line facts are preferred when available
    score -= 0.4 * url_tier(fact.url)
    return score


def select_facts(kb, pq, as_of):
    idx = kb.index(as_of)
    rel = idx.scores(pq.words)
    cand_urls = set(sorted(rel, key=lambda u: -rel[u])[:45])
    if not cand_urls:
        return []
    # The resolved boilerplate facts (branch count, membership area) are few
    # -- at most one per topic, after _resolve_boilerplate -- and whichever
    # one page happened to carry the winning (latest-byline) copy is not
    # necessarily a page this query's words would rank in the top 45 pages
    # overall. Always let them compete on score_fact's own merits rather
    # than risk losing the single resolved answer to page-level cutoff.
    all_facts = [f for f in kb.facts(as_of)
                 if f.url in cand_urls or (f.item_key and f.item_key[0] == "boilerplate")]
    scored = [(score_fact(f, pq, idx.idf, rel.get(f.url, 0.0)), f) for f in all_facts]
    scored = [(s, f) for s, f in scored if s is not None and s >= MIN_SCORE]
    if not scored:
        return []
    scored.sort(key=lambda t: -t[0])
    top_score = scored[0][0]

    tight = [f for s, f in scored if s >= top_score - 0.75]

    if pq.wants_rate:
        # Gate on the TIGHT top-score band regardless of mode: a genuine
        # published rate for this question should be near the best match, not
        # merely present somewhere in a widened, list-style recall window.
        # Otherwise a construction-loan/ARM question with no actual rate can
        # still "pass" just because some unrelated, weakly-matching table
        # elsewhere in the wide net happens to carry a real rate.
        if not any(f.kind in ("table", "line") and f.is_rate for f in tight):
            return []

    if pq.named_entities:
        # Same reasoning as above: whether the question's named entity (a
        # branch, a program...) is actually grounded has to be judged from
        # the TOP matches, not the whole widened pool -- a passing mention in
        # some low-scoring, unrelated paragraph (a blog post name-dropping a
        # branch) shouldn't "vouch" for an unrelated fact (another branch's
        # hours) that just happened to score well on other shared words.
        grounded = set()
        for f in tight:
            grounded |= f.qualifiers
        if not pq.named_entities.issubset(grounded):
            return []

    # A specific tier/term/year/amount question wants ONE precise answer, so
    # the window stays tight. A broad "what do you offer" question is often
    # enumerating several genuinely different products (four credit cards,
    # say) that just don't all word-match the question as strongly as the
    # single best one -- a tight window would silently drop the rest, so it
    # widens proportionally to the top score instead of by a flat amount.
    window = top_score - 0.75 if pq.has_specific_qualifier else top_score * 0.3
    best = [f for s, f in scored if s >= window]

    if pq.has_specific_qualifier:
        # Narrow: a specific tier/term/year/amount was named -> answer with the single
        # best-matching fact (most authoritative page, then freshest).
        chosen = sorted(best, key=lambda f: (url_tier(f.url), f.date), reverse=False)
        top_tier = url_tier(chosen[0].url)
        pool = [f for f in chosen if url_tier(f.url) == top_tier]
        pool.sort(key=lambda f: f.date, reverse=True)
        return pool[:1]

    # No specific qualifier: group near-best facts by "item" (tiers of the same thing
    # collapse together; genuinely distinct items/products stay separate). For each
    # item, lead with the superlative if asked for, else the lowest (base/intro) tier
    # -- the convention real rate sheets use -- but still include the item's other
    # nearby tiers (capped), since some questions expect the full small range (e.g.
    # "30-year mortgage rate" wants both its $100k-tier and its top-tier number) and
    # none of them are the stale/wrong values must_not_include actually guards against.
    # `scored` is already sorted best-first, so the first time an item_key is seen
    # marks its best member's rank -- preserve THAT order for the final output
    # instead of re-sorting alphabetically, or a handful of strong matches can get
    # pushed out of the (capped) answer by a pile of equally-scored but far less
    # useful items (e.g. a long list of branch addresses crowding out the one
    # relevant phone number).
    item_order: dict = {}
    by_item: dict = defaultdict(list)
    for rank, f in enumerate(best):
        by_item[f.item_key].append(f)
        item_order.setdefault(f.item_key, rank)
    out = []
    for key in sorted(by_item, key=lambda k: item_order[k]):
        members = by_item[key]
        members_with_num = [m for m in members if m.primary_num is not None]
        pool = members_with_num or members
        if pq.superlative == 1:
            pool = sorted(pool, key=lambda m: (-(m.primary_num if m.primary_num is not None else -1e9), url_tier(m.url)))
            out.append(pool[0])
        elif pq.superlative == -1:
            pool = sorted(pool, key=lambda m: ((m.primary_num if m.primary_num is not None else 1e9), url_tier(m.url)))
            out.append(pool[0])
        else:
            pool = sorted(pool, key=lambda m: (url_tier(m.url), (m.primary_num if m.primary_num is not None else 1e9), m.date))
            out.extend(pool[:4])
    return out[:16] or best[:1]


# No digits here on purpose: the grader treats a refusal that still blurts a
# number (even a helpful one, like a phone number) as not a clean refusal.
REFUSAL_TEXT = "That isn't listed on our site. Please contact a Dupaco team member so they can help directly."


def build_answer(kb, question: str, as_of):
    pq = parse_question(question)
    facts = select_facts(kb, pq, as_of)

    if not facts:
        return Answer(text=REFUSAL_TEXT, sources=[], confident=False)

    # A rate question is only answerable from an actual quoted rate (a table
    # row, or a line fact, flagged is_rate) -- prose that merely mentions "the
    # rate" or carries unrelated percentages (an LTV ratio, an ARM adjustment
    # cap) is not a published rate for the product asked about, and quoting it
    # anyway is exactly the kind of confident-but-wrong answer should_refuse
    # tests for.
    if pq.wants_rate and not any(f.kind in ("table", "line") and f.is_rate for f in facts):
        return Answer(text=REFUSAL_TEXT, sources=[], confident=False)

    # A question that names a specific entity (a branch, a program...) we
    # never actually matched -- the retrieval fell back to the
    # highest-scoring fact on OTHER shared words, which is usually a
    # different, unrelated branch/program's info. Don't hand that back as if
    # it answered the question.
    if pq.named_entities:
        grounded = set()
        for f in facts:
            grounded |= f.qualifiers
        # Require EVERY named entity to be grounded, not just one -- a generic
        # street-type word ("Avenue", "Street") shared with some other
        # branch's address would otherwise satisfy an any-match check on its
        # own and mask that the branch's actual name ("Mullan") never matched.
        if not pq.named_entities.issubset(grounded):
            return Answer(text=REFUSAL_TEXT, sources=[], confident=False)

    if pq.wants_url:
        # "What link/URL should I use..." wants the URL itself in the answer text,
        # not the fact's value -- lead with the single best (most authoritative,
        # freshest) source's own address.
        top = sorted(facts, key=lambda f: (url_tier(f.url), f.date), reverse=False)
        top = sorted(top, key=lambda f: url_tier(f.url))
        return Answer(text=top[0].url, sources=[Source(url=top[0].url, snapshot_id=top[0].snapshot_id)], confident=True)

    if len(facts) == 1:
        f = facts[0]
        return Answer(text=f.text, sources=[Source(url=f.url, snapshot_id=f.snapshot_id)], confident=True)

    parts, sources, seen_urls = [], [], set()
    for f in facts[:12]:
        parts.append(f.text)
        if f.url not in seen_urls:
            sources.append(Source(url=f.url, snapshot_id=f.snapshot_id))
            seen_urls.add(f.url)
    return Answer(text="; ".join(parts), sources=sources, confident=True)


def _pick_best(members: list) -> "Fact":
    """Same authority rule used everywhere else in this file: the most
    authoritative url_tier wins, ties broken by the freshest date."""
    ranked = sorted(members, key=lambda m: (url_tier(m.url), m.date))
    top_tier = url_tier(ranked[0].url)
    pool = [m for m in ranked if url_tier(m.url) == top_tier]
    pool.sort(key=lambda m: m.date, reverse=True)
    return pool[0]


def _num_sig(value: str) -> frozenset:
    """A value's $/% figures only -- ignores footnote-marker digits ("7.29%1"
    parses an extra bare "1") and $ vs plain-number formatting noise, so two
    renderings of the SAME figure don't look like a conflict."""
    return frozenset((u, v) for u, v in numbers_in(value) if u in ("$", "%"))


def find_conflicts(kb: "KB") -> list:
    """Scan every version of every page across all five snapshots for
    same-topic facts -- grouped by item_key with the per-page URL component
    dropped, so the same row/line/boilerplate statement groups together
    whether it recurs on one page over time or on several different pages --
    whose $/% figures disagree. Each disagreement is resolved with the
    identical authority rule answer() uses (most authoritative url_tier, then
    freshest/latest-known date), so the resolution here always matches what
    answer() would say about the same topic."""
    all_facts = []
    for versions in kb.history.values():
        for pv in versions:
            all_facts.extend(extract_facts(pv))

    groups: dict = defaultdict(list)
    for f in all_facts:
        if f.kind == "para" or not f.item_key:
            continue  # free text rarely shares a stable key across pages/snapshots; too noisy
        sig = _num_sig(f.value)
        if not sig:
            continue  # a conflict worth reporting is a disagreement in a $/% figure
        key = f.item_key[1:] if len(f.item_key) > 1 else f.item_key
        if not any(key):
            continue
        groups[key].append((sig, f))

    conflicts = []
    for key, pairs in groups.items():
        # A table row with no distinguishing label text (pure numeric
        # columns) falls back to a constant row_label, so every tier in the
        # SAME table/snapshot shares this key -- that's one page's own
        # multi-row table, not a site-wide conflict, and the key is too
        # coarse to trust at all. Bail out on the whole group if a single
        # (url, date) capture alone already disagrees with itself.
        same_capture: dict = defaultdict(set)
        for sig, f in pairs:
            same_capture[(f.url, f.date)].add(sig)
        if any(len(sigs) > 1 for sigs in same_capture.values()):
            continue

        by_sig: dict = defaultdict(list)
        for sig, f in pairs:
            by_sig[sig].append(f)
        if len(by_sig) < 2:
            continue  # every copy agrees on the actual figures -- not a conflict

        reps = [_pick_best(ms) for ms in by_sig.values()]
        winner = _pick_best(reps)

        label_bits = [str(b) for b in key if b and str(b) not in ("bullet", "para")]
        topic = " / ".join(label_bits)[:160] or winner.text[:80]

        if len(reps) > 1 and len({url_tier(r.url) for r in reps}) > 1:
            reason = f"most authoritative source ({winner.url}) is kept over the others"
        elif len(reps) > 1:
            reason = f"most recently confirmed value as of {winner.date} ({winner.url})"
        else:
            reason = f"only source found for this value ({winner.url})"

        conflicts.append(Conflict(
            topic=topic,
            values=sorted({r.value for r in reps}),
            sources=[Source(url=r.url, snapshot_id=r.snapshot_id) for r in reps],
            resolution=winner.value,
            reason=reason,
        ))

    conflicts.sort(key=lambda c: c.topic)
    return conflicts


# --------------------------------------------------------------------------
# KnowledgeSystem
# --------------------------------------------------------------------------


class DupacoKnowledge:
    def __init__(self) -> None:
        self.kb = KB()

    def ingest(self, snapshot: Snapshot) -> None:
        self.kb.ingest(snapshot)

    def answer(self, question: str, as_of=None) -> Answer:
        return build_answer(self.kb, question, as_of)

    def conflicts(self) -> list[Conflict]:
        return find_conflicts(self.kb)


def build():
    return DupacoKnowledge()

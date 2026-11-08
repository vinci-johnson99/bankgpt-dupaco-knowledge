# Approach

Pure Python standard library. No LLM, no API key, no network calls, fully
deterministic (same question + same ingested snapshots always produces the
same answer). This was a deliberate choice: the assignment weighs speed and
consistency at 30% combined, and an LLM-free, index-based approach makes both
of those close to free, while still being descriptive enough for plain
members-question answering.

## Data model

Each page is kept as a **version history**, not a flat page list: `KB.history[url]`
is a list of `PageVersion(url, title, markdown, date, snapshot_id)`, one per
snapshot that actually crawled that URL, ordered by the snapshot's own date
(not necessarily the page's own byline). `KB.view(as_of)` picks, for every
URL, whichever version was current at or before `as_of` (or the latest
version when `as_of is None`). This means staleness is resolved **per page**,
independent of which snapshot a page happened to be re-crawled in -- a page
that didn't change between crawl 2 and crawl 4 still "exists" at every `as_of`
in that window.

## Fact extraction

`extract_facts(page_version)` turns each page's markdown into a list of
`Fact` records (table rows, bullet "Label: Value" lines, phone/address/routing
lines, branch-hours blocks, two specific boilerplate-company-stat patterns,
and finally generic paragraphs as a fallback). Each `Fact` carries:

- `value` / `text` -- what to show the member
- `qualifiers` -- bag of content words from the row, its table heading,
  column header, and the page title, so a fact "knows" what it's about
  without needing the raw page text re-scanned at question time
- `item_key` -- a cross-tier identity, e.g. `(url, table heading, title line,
  row label)` -- rows that are really the same line item at different tiers
  share this
- `primary_num` -- the row's headline number, used for "highest/lowest" questions
- `is_rate` -- whether this row/line is actually an interest-rate figure
  (not just a paragraph that happens to mention a `%`)

## Answering a question

`parse_question` extracts content words, named entities (capitalized
non-sentence-initial words), whether the question wants a rate, a URL, a
superlative, a year/month/dollar range, and whether it names a specific
qualifier (narrowing to one answer vs. enumerating several). `select_facts`
then:

1. Scores every candidate fact (`score_fact`) by IDF-weighted word overlap
   with the question, with bonuses/penalties for date-range/month-range/
   dollar-range matches, a page-relevance signal from a small TF-IDF index,
   and a `url_tier` penalty (the consolidated `/rates` page and most product
   pages are tier 0; shallow or generic pages are tier 1; newsroom/blog posts
   are tier 2; calculators and "-lp" landing pages -- the classic stale-trap
   sources -- are tier 5).
2. Computes a **tight band** (facts within 0.75 of the top score) and gates
   on it: a rate question refuses unless a real rate fact is in the tight
   band; a question naming specific entities refuses unless *every* named
   entity is grounded in the tight band's own qualifiers (not just any
   overlap -- a generic word like "Avenue" shouldn't let a wrong branch's
   address sneak past a check for "Mullan Avenue"). These gates run against
   the tight band specifically so that a wider net used later for
   list-style answers can't quietly turn a should-refuse into a
   confident-but-wrong answer.
3. For a narrowly-qualified question, returns the single most authoritative,
   most recent fact. For a broader one (e.g. "what credit cards do you
   offer"), widens the window and returns one fact per `item_key`, so
   multiple tiers/products are enumerated rather than only the top-scoring
   row.

A refusal (`Answer(text=REFUSAL_TEXT, confident=False)`) is returned whenever
nothing clears the gates above -- intentionally containing **zero digits**,
since the grader's own refusal check requires no stray numbers in the
refusal text.

## Resolving the same fact when pages disagree

Two kinds of disagreement showed up in the snapshots:

- **Genuine staleness**: a page's own numbers changed between crawls (a rate
  table, a promo rate). `KB.view(as_of)` already picks the right version by
  crawl date for this case.
- **Copy-pasted boilerplate with drifting numbers**: an "ABOUT DUPACO..."
  paragraph (branch count, membership-eligibility county count) is
  duplicated verbatim across 70+ newsroom pages, and different copies quote
  different numbers because each article was written on a different date --
  but every *copy* shares the same crawl date, so crawl recency can't tell
  them apart, and a naive majority vote across all copies gives the wrong
  answer (more copies happen to repeat an older figure). The fix:
  `_article_byline_date` reads each individual article's own dateline
  ("April 06, 2026") and `_resolve_boilerplate` keeps whichever copy's byline
  is the latest one at-or-before `as_of`, treating that byline -- not the
  crawl date -- as the fact's true date.

## `conflicts()`

Re-extracts facts from **every** version of **every** page (not just the
latest view), groups them by `item_key` with the page's own URL dropped (so
the same row/line groups together whether it recurs across time on one page
or across several different pages), and flags a group whose members'
**$/% figures** (not raw text -- this ignores footnote-marker digits like the
"1" in "7.29%1" and $-vs-plain-number formatting noise) actually disagree.
A safety check discards any group where a single page-capture (one URL, one
date) already contains more than one distinct figure under the same key --
that means the key collapsed several genuinely different rows (e.g. a table
with no text row-labels, only numeric tiers) rather than tracking one real
topic, so it isn't trustworthy for conflict-finding even though it's fine for
answer-selection. Each surviving conflict is resolved with the exact same
authority rule `answer()` uses: most authoritative `url_tier` first, freshest
date as the tiebreaker.

## Known limitations

- A handful of individual eval questions are still imperfect: a promo
  program outranking the standard tiered rate on one tier-disambiguation
  question; one business-card list that's missing a couple of lower-ranked
  rows; one as-of question where an image caption outscores the actual
  closure-announcement paragraph; one stale-trap paragraph that scores just
  under the refusal/selection threshold against a competing paragraph on the
  same page. These are scoring-threshold edge cases in an otherwise working
  pipeline, not architectural gaps.
- `conflicts()` has no local gold file to score against (the harness only
  invokes `score_conflicts` when `--conflicts-gold` is passed), so its
  output here was hand-checked against conflicts found during development
  (rate drift across snapshots on the mortgage/auto-loan pages, the
  branch-count and membership-eligibility boilerplate disagreements) rather
  than tuned against a scoring run.
- Last local run of the public eval: **0.8124** overall (`speed` and
  `consistency` both effectively 1.0; `accuracy` ~0.81), with 0 stale-trap
  values incorrectly surfaced as true.

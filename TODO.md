# Analysis Notebook: Related Works 2015 vs 2025 Comparison

## Hypothesis

> LLM availability (post-2022) has measurably altered the linguistic style, vocabulary, and structure of "Related Works" sections in Computer Science papers on arXiv.

---

## Data

| Item | Path | Format |
|------|------|--------|
| 2015 papers | `data/txt/{arxiv_id}/*.txt` | Plain text, one file per section |
| 2025 papers | `data-2025/txt/{arxiv_id}/*.txt` | Plain text, one file per section |

- Pipeline was run with `txt_related_only: true`, so **all txt files are related work sections**
- Files are named `{order:02d}_{sanitized_heading}.txt` (e.g. `02_Related_Work.txt`)
- Text contains `<cit.>` placeholders where LaTeX `\cite{}` commands were
- Multiple files per paper = subsections of related work; **concatenate per arxiv_id** for analysis

---

## Dependencies

```
pip install numpy pandas matplotlib seaborn scipy nltk
```

NLTK data needed:
```python
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('stopwords', quiet=True)
```

---

## Notebook Structure (`analysis.ipynb`)

### Section 0: Setup

- **Cell 0.1 [Markdown]:** Title, hypothesis, date
- **Cell 0.2 [Code]:** Imports and constants:
  ```python
  import pathlib, re, collections, warnings
  import numpy as np, pandas as pd
  import matplotlib.pyplot as plt, seaborn as sns
  from scipy import stats
  import nltk
  from nltk.tokenize import sent_tokenize, word_tokenize

  DATA_2015 = pathlib.Path("data/txt")
  DATA_2025 = pathlib.Path("data-2025/txt")
  CIT_PATTERN = re.compile(r"<cit\.>")

  sns.set_theme(style="whitegrid", palette="colorblind")
  plt.rcParams["figure.dpi"] = 120
  ```

---

### Section 1: Data Loading (4 cells)

- [ ] **Cell 1.1:** Load all `.txt` files into a DataFrame
  - Walk each data directory, read all txt files
  - Columns: `arxiv_id`, `year_group` ("2015" / "2025"), `raw_text`, `filename`
  - Check both directories exist, print clear error if missing

- [ ] **Cell 1.2:** Corpus statistics
  - Number of papers per group
  - Total / mean / median word count per group
  - Print summary table

- [ ] **Cell 1.3:** Per-paper aggregation
  - Concatenate all txt files per `arxiv_id` into a single `text` column → `df_papers`
  - Filter out papers with < 50 words (noise / extraction failures)

- [ ] **Cell 1.4 [Markdown]:** Commentary on corpus balance, filtering applied

---

### Section 2: Text Preprocessing (2 cells)

- [ ] **Cell 2.1:** Define helper functions:

  | Function | Description |
  |----------|-------------|
  | `tokenize_sentences(text)` | `nltk.sent_tokenize`, handles `<cit.>` gracefully |
  | `tokenize_words(text)` | `nltk.word_tokenize`, strip punctuation-only tokens |
  | `count_citations(text)` | Count `<cit.>` occurrences |
  | `count_paragraphs(text)` | Split on `\n\n` or blank lines |
  | `count_syllables(word)` | Regex vowel-group counter |
  | `flesch_reading_ease(text)` | `206.835 - 1.015*(words/sents) - 84.6*(syllables/words)` |
  | `gunning_fog(text)` | `0.4 * ((words/sents) + 100*(complex_words/words))` |
  | `is_passive(sentence)` | POS-tag: BE-verb (VBZ/VBP/VBD) + past participle (VBN) |
  | `type_token_ratio(words)` | `len(set(words)) / len(words)` |
  | `mattr(words, window=50)` | Moving Average TTR (length-independent) |
  | `hapax_legomena_ratio(words)` | Words appearing exactly once / total words |

  **Notes:**
  - MATTR (window=50) is preferred over standard TTR because TTR is biased by text length
  - Passive voice detection via POS tags is ~80-85% accurate, fine for comparative analysis
  - `count_syllables`: count groups of vowels `[aeiouy]+`, minimum 1 per word

- [ ] **Cell 2.2:** Apply feature extraction to `df_papers`, producing columns:
  - `n_sentences`, `n_words`, `n_paragraphs`, `n_citations`
  - `mean_sentence_length`, `median_sentence_length`
  - `ttr`, `mattr`, `hapax_ratio`
  - `flesch`, `gunning_fog`
  - `passive_ratio`
  - `cit_per_sentence`, `cit_per_100_words`

---

### Section 3: Linguistic Style Analysis (6 cells)

- [ ] **Cell 3.1 [Markdown]:** Section header, explanation of metrics

- [ ] **Cell 3.2:** Summary statistics table
  - Mean, median, std, min, max for each metric grouped by year
  - Display as styled DataFrame

- [ ] **Cell 3.3:** Visualization -- **2x3 grid of violin plots**
  1. Mean sentence length
  2. MATTR (vocabulary richness)
  3. Hapax legomena ratio
  4. Flesch reading ease
  5. Gunning Fog index
  6. Passive voice ratio

- [ ] **Cell 3.4:** Visualization -- **Overlaid KDE plots** for sentence length distributions
  - Pool all sentences (not per-paper means) to show full distributional shift
  - Color by year group

- [ ] **Cell 3.5:** Statistical tests for each metric:
  - Mann-Whitney U test (always -- data likely non-normal)
  - Welch's t-test (as parametric complement)
  - Effect sizes: **rank-biserial r** and **Cohen's d**
  - Display results as summary table with p-values and effect sizes

- [ ] **Cell 3.6 [Markdown]:** Interpretation of findings

---

### Section 4: LLM-Specific Marker Analysis (6 cells)

- [ ] **Cell 4.1 [Markdown]:** Methodology explanation
  - Based on Liang et al. (2024) "Mapping the Increasing Use of LLMs in Scientific Papers"
  - Two tiers: high-confidence markers (rare pre-LLM) vs elevated-frequency markers (existed but spiked)

- [ ] **Cell 4.2:** Define marker phrase dictionary:

  **High-confidence markers** (rare pre-LLM):
  | Label | Regex |
  |-------|-------|
  | delve(s) into | `\bdelves?\s+into\b` |
  | it is worth noting | `\bit\s+is\s+worth\s+noting\b` |
  | comprehensive overview | `\bcomprehensive\s+overview\b` |
  | in the realm of | `\bin\s+the\s+realm\s+of\b` |
  | plays a crucial/pivotal role | `\bplays\s+a\s+(?:crucial\|pivotal)\s+role\b` |
  | noteworthy | `\bnoteworthy\b` |
  | multifaceted | `\bmultifaceted\b` |
  | tapestry | `\btapestry\b` |
  | groundbreaking | `\bgroundbreaking\b` |
  | paving the way | `\bpaving\s+the\s+way\b` |
  | has garnered significant/considerable attention | `\bhas\s+garnered\s+(?:significant\|considerable)\s+attention\b` |
  | demonstrated remarkable | `\bdemonstrated\s+remarkable\b` |
  | leveraging the power | `\bleveraging\s+the\s+power\b` |
  | intricate | `\bintricate\b` |
  | underscores | `\bunderscores\b` |

  **Elevated-frequency markers** (existed before but spiked post-LLM):
  | Label | Regex |
  |-------|-------|
  | furthermore | `\bfurthermore\b` |
  | moreover | `\bmoreover\b` |
  | additionally | `\badditionally\b` |
  | landscape | `\blandscape\b` |
  | notably | `\bnotably\b` |
  | cutting-edge | `\bcutting[\s-]edge\b` |
  | in recent years | `\bin\s+recent\s+years\b` |
  | a growing body of | `\ba\s+growing\s+body\s+of\b` |
  | shed(s) light on | `\bsheds?\s+light\s+on\b` |

- [ ] **Cell 4.3:** For each paper, compute:
  - Raw count of each marker phrase
  - Per-1000-words normalized count
  - Binary "contains" flag
  - Build markers DataFrame

- [ ] **Cell 4.4:** Visualization -- **Grouped horizontal bar chart**
  - For each marker: % of papers containing it in 2015 vs 2025
  - Sort by ratio (2025 / 2015)
  - Use log scale on x-axis if ranges are wide

- [ ] **Cell 4.5:** Statistical tests per marker:
  - Chi-squared or Fisher's exact test (when expected counts < 5) on proportions
  - **Bonferroni correction** + **Benjamini-Hochberg FDR** (Bonferroni is conservative for 24+ tests)
  - Compute **odds ratios**
  - Mann-Whitney U on normalized frequencies
  - Display sorted by effect size

- [ ] **Cell 4.6:** Aggregate **"LLM marker score"**
  - Sum of all high-confidence marker counts per 1000 words
  - Violin plot comparing 2015 vs 2025
  - Mann-Whitney U test

---

### Section 5: Structural Pattern Analysis (6 cells)

- [ ] **Cell 5.1 [Markdown]:** Explain structural metrics

- [ ] **Cell 5.2:** Summary statistics for:
  - Section word count
  - Paragraph count
  - Total citation count (`<cit.>`)
  - Citations per sentence
  - Citations per 100 words

- [ ] **Cell 5.3:** Visualization -- **2x2 box plots**
  1. Section word count
  2. Paragraph count
  3. Total citation count
  4. Citation density (per sentence)

- [ ] **Cell 5.4:** Citation introduction pattern analysis
  - Examine N words before each `<cit.>` and classify:
    - **(a) Parenthetical/trailing:** "...method X `<cit.>`"
    - **(b) Author-named:** "Smith et al. `<cit.>`"
    - **(c) Verb-led:** "proposed by `<cit.>`", "introduced in `<cit.>`"
    - **(d) List-style:** "`<cit.>`, `<cit.>`, `<cit.>`"
  - Compare proportions between years
  - **Data quality check:** report % of papers with zero `<cit.>` markers per group

- [ ] **Cell 5.5:** Visualization -- **Scatter plot**
  - Citation count vs section word count
  - Colored by year group, with regression lines
  - Tests whether citation density patterns changed

- [ ] **Cell 5.6:** Statistical tests for all structural metrics (Mann-Whitney U + effect sizes)

---

### Section 6: Combined Summary Dashboard (3 cells)

- [ ] **Cell 6.1:** Master summary table
  - Columns: metric name, 2015 mean, 2015 median, 2025 mean, 2025 median, % change, Mann-Whitney U, p-value, effect size (r), significance flag
  - Sort by absolute effect size

- [ ] **Cell 6.2:** Visualization -- **Effect size dashboard**
  - Horizontal bar chart of rank-biserial correlation (r) for every metric
  - Error bars for confidence intervals
  - Colored by significance (p < 0.05 after correction)
  - This is the single most informative visualization in the notebook

- [ ] **Cell 6.3 [Markdown]:** Final discussion
  - Key findings
  - Limitations:
    - No per-category metadata (cs.CL vs cs.CV etc.) in txt files
    - Potential confounders: field growth, topic shifts
    - `<cit.>` handling differences between pipeline versions
    - Sample size effects
  - Future work:
    - Per-subcategory analysis (join arxiv_id back to metadata)
    - Temporal trends within 2022-2025
    - Comparison with non-CS fields

---

### Section 7: Appendix (2 cells)

- [ ] **Cell 7.1:** Export metrics to CSV
  ```python
  df_papers.to_csv("related_works_metrics.csv", index=False)
  ```

- [ ] **Cell 7.2:** Print environment info for reproducibility
  - Python version, package versions, data directory timestamps

---

## Implementation Notes

1. **MATTR > TTR:** Standard TTR is biased by text length (longer texts always get lower TTR). MATTR with window=50 is length-independent. Include both, emphasize MATTR in conclusions.

2. **Passive voice detection:** POS-tag approach (`nltk.pos_tag`) looking for BE-verb + past-participle (e.g. "was proposed", "has been studied"). ~80-85% accurate, acceptable since the same bias applies to both groups.

3. **Syllable counting:** Regex vowel-group counting (`[aeiouy]+`). Avoids the `textstat` dependency.

4. **Citation edge cases:** Some files may have citations stripped rather than replaced with `<cit.>`. Report % of papers with zero markers per group as a data quality check.

5. **Multiple comparisons:** Bonferroni is conservative for 24+ marker phrases. Report both Bonferroni and Benjamini-Hochberg FDR-adjusted p-values.

6. **Graceful failure:** Notebook should check data directory existence upfront and print clear instructions if missing.

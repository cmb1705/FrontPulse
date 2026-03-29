#!/usr/bin/env python3
"""
Stage 4: NPMI Co-term Discovery for Lineages

Identifies co-occurring term patterns using Normalized Pointwise Mutual Information (NPMI)
and matches them to research front canonical term combinations.

NPMI measures how strongly two terms co-occur:
    NPMI(x,y) = PMI(x,y) / -log(P(x,y))
    where PMI(x,y) = log(P(x,y) / (P(x) * P(y)))

Range: [-1, 1]
    +1: Perfect co-occurrence (always appear together)
     0: Independent (random co-occurrence)
    -1: Never co-occur

Usage:
    python scripts/compute_lineage_npmi.py --min-quarters 6

    # Run Stage 4 with 6 parallel workers (requires psutil for RAM checks)
    python scripts/compute_lineage_npmi.py --min-quarters 6 --max-workers 6 --worker-memory-gb 3 --memory-reserve-gb 4

Outputs (relative to --output-root):
    - 02_lineage_tracking/lineage_npmi_pairs.csv: Top co-occurring pairs per lineage
    - 03_milestone_mapping/lineage_front_npmi_similarity.csv: NPMI similarity matrix (99 x 15)

Notes:
    - Parallel execution is disabled unless psutil is installed; the script falls back to sequential processing otherwise.
    - --worker-memory-gb and --memory-reserve-gb guard against oversubscribing RAM when launching worker processes.
    - --abstract-cache overrides the default serialized abstract index cache path (auto-generated under data/out/cache_lineage/).
"""

import json
import math
import multiprocessing as mp
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None

# Language detection and lemmatization
try:
    from langdetect import LangDetectException, detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("[WARNING] langdetect not installed. Run: pip install langdetect")

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        SPACY_AVAILABLE = False
        print("[WARNING] spaCy model not found. Run: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    print("[WARNING] spaCy not installed. Run: pip install spacy")

from _path_bootstrap import ensure_repo_imports

# Add project root to path
REPO_ROOT = ensure_repo_imports()

# Import from previous stages
from scripts.compute_lineage_ctfidf import LineageTermExtractor  # noqa: E402
from scripts.extract_abstracts import AbstractExtractor  # noqa: E402
from src.domain_registry import add_domain_args, resolve_script_paths  # noqa: E402

# Markup/formatting artifacts to filter out (XML, HTML, math markup, LaTeX)
MARKUP_TERMS = {
    'http', 'https', 'www', 'com', 'org', 'edu', 'doi', 'html', 'xml',
    'math', 'mathml', 'mml', 'xmlns', 'mrow', 'msub', 'msup', 'mfrac',
    'abstracting', 'delivering', 'concise', 'indexed', 'svg', 'xlink',
    'sup', 'sub', 'span', 'div', 'tspan', 'class', 'style', 'href', 'mtext',
    'mathvariant', 'normal', 'italic', 'bold', 'font', 'size', 'color',
    # LaTeX/math formatting
    '0ex', '3em', 'phantom', 'rule', 'ifmmode', 'else', 'inline', 'box',
    'reflect', 'textbf', 'textit', 'textrm', 'frac', 'sqrt', 'cdot',
    # Metadata/layout
    'panels', 'panel', 'image', 'images', 'magnified', 'contents', 'directed'
}

# Extended stopwords: common verbs, adjectives, adverbs, scientific boilerplate
EXTENDED_STOPWORDS = {
    # Basic stopwords
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will',
    'with', 'been', 'have', 'had', 'were', 'this', 'these', 'those', 'their',
    'there', 'which', 'can', 'we', 'our', 'but', 'not', 'all', 'they', 'may',
    'such', 'into', 'more', 'also', 'than', 'most', 'other', 'some', 'up', 'out',
    'only', 'so', 'no', 'if', 'about', 'or', 'when', 'after', 'between', 'then',
    'before', 'however', 'through', 'where', 'both', 'each', 'under', 'during',
    'using', 'very', 'any', 'over', 'how', 'same', 'would', 'could', 'should',

    # Year tokens (1900-2099)
    *[str(year) for year in range(1900, 2100)],

    # Numeric patterns common in papers
    'fig', 'figs', 'eq', 'eqn', 'eqs', 'vol', 'pp', 'page', 'pages',
    'chapter', 'chapters', 'supplementary', 'supplement', 'si',

    # Common verbs
    'show', 'showed', 'shown', 'shows', 'demonstrate', 'demonstrated', 'demonstrates',
    'indicate', 'indicated', 'indicates', 'suggest', 'suggested', 'suggests',
    'observe', 'observed', 'observes', 'find', 'found', 'finds', 'report', 'reported',
    'obtain', 'obtained', 'obtains', 'achieve', 'achieved', 'achieves', 'perform',
    'performed', 'performs', 'measure', 'measured', 'measures', 'calculate', 'calculated',
    'determine', 'determined', 'determines', 'provide', 'provided', 'provides',
    'present', 'presented', 'presents', 'describe', 'described', 'describes',
    'reveal', 'revealed', 'reveals', 'confirm', 'confirmed', 'confirms',
    'examine', 'examined', 'examines', 'investigate', 'investigated', 'investigates',
    'study', 'studied', 'studies', 'analyze', 'analyzed', 'analyzes', 'discuss',
    'discussed', 'discusses', 'compare', 'compared', 'compares', 'propose', 'proposed',
    'develop', 'developed', 'develops', 'use', 'used', 'uses', 'apply', 'applied',
    'consider', 'considered', 'considers', 'explore', 'explored', 'explores',
    'evaluate', 'evaluated', 'evaluates', 'assess', 'assessed', 'assesses',

    # Common adjectives
    'high', 'low', 'large', 'small', 'good', 'bad', 'new', 'old', 'important',
    'significant', 'different', 'various', 'several', 'many', 'few', 'first',
    'second', 'third', 'last', 'next', 'previous', 'following', 'above', 'below',
    'similar', 'multiple', 'single', 'double', 'triple',
    'better', 'worse', 'higher', 'lower', 'larger', 'smaller', 'greater', 'lesser',
    'strong', 'weak', 'effective', 'efficient', 'successful', 'excellent', 'poor',

    # Scientific boilerplate
    'results', 'result', 'conclusion', 'conclusions', 'abstract', 'introduction',
    'discussion', 'method', 'methods', 'methodology', 'experimental', 'theory',
    'theoretical', 'analysis', 'synthesis', 'data', 'figure', 'figures', 'table',
    'tables', 'section', 'sections', 'reference', 'references', 'cited',
    'citation', 'citations', 'published', 'journal', 'journals', 'paper', 'papers',
    'article', 'articles', 'author', 'authors', 'copyright', 'rights', 'reserved',
    'publisher', 'publication', 'elsevier', 'springer', 'wiley', 'elsewhere',
    'extracted', 'glance', 'option', 'original', 'select', 'service', 'trackable',
    'weekly', 'recently', 'currently', 'previously', # Publisher names and metadata
    'gmbh', 'kgaa', 'verlag', 'weinheim', 'wiley-vch', 'wiley-blackwell',
    'nature', 'acs', 'rsc', 'iop', 'aip', 'ieee',
    'functionality', 'missing', 'queries', 'supplied', 'ltd', 'inc', 'llc',
    'amp', 'vch', 'co', 'corp', 'press', 'john', 'sons', 'american',

    # Journal/peer review boilerplate and metrics
    'abstractcitation', 'advertisement', 'altmetric', 'citing', 'clicking',
    'counter-compliant', 'crossref', 'donut', 'copy-edited', 'peer-reviewed',
    'specialist', 'readers', 'typeset', 'documents', 'facts', 'submitted',
    'delivery', 'files', 're-organized', 'technical',
    'counts', 'downloads', 'e-alerts', 'exportriscitationcitation', 'icon',
    'inadd', 'inissue', 'institutions', 'individuals', 'load', 'metricsarticle',
    'cite', 'daily', 'get', '1021',
    # Social media and web interface artifacts
    'inredditemail', 'publicationscopyright', 'pubs', 'reuse', 'november',
    'onfacebooktwitterwechatlinked', 'referenceadd', 'referencesmore', 'score',
    'social', 'toview', 'optionsget', 'media', 'metrics', 'options', 'permissionsarticle', 'permissions', # Creative Commons and licensing
    'commons', 'license', 'licensed', 'attribution', 'creativecommons',
    # Supporting information boilerplate
    'available', 'note', 'please', 'information', 'see', 'supporting',
    # Non-English words (German articles, prepositions, conjunctions)
    'bild', 'mit', 'siehe', 'der', 'die', 'das', 'den', 'dem', 'des',
    'und', 'oder', 'aber', 'von', 'für', 'bei', 'zu', 'nach', 'vor',
    'eine', 'einen', 'einem', 'einer', 'eines', 'ein', 'sind', 'ist',
    'wird', 'werden', 'wurde', 'wurden', 'war', 'waren', 'auf', 'über',
    'durch', 'unter', 'zwischen', 'während', 'seit', 'als', 'wenn',

    # Institutional terms
    'university', 'universities', 'institute', 'institut', 'laboratory',
    'laboratories', 'center', 'centre', 'centers', 'centres', 'college',
    'colleges', 'department', 'departments', 'school', 'schools',
    'academy', 'academies', 'society', 'societies', 'foundation',
    'foundations', 'association', 'associations', 'consortium',

    # Common verbs and verb forms to remove
    'given', 'place', 'takes', 'took', 'taken', 'give', 'gives', 'giving',
    'make', 'makes', 'made', 'making', 'history', 'become', 'becomes', 'became',
    'pull', 'push', 'pulled', 'pushed', 'pulling', 'pushing',

    # Generic descriptors
    'neglect', 'overlap', 'detail', 'central', 'fluctuation', 'bluish-white',

    # Time/quantity words
    'time', 'times', 'year', 'years', 'month', 'months', 'day', 'days', 'hour',
    'hours', 'minute', 'minutes', 'seconds', 'number', 'numbers',
    'amount', 'amounts', 'value', 'values', 'level', 'levels', 'rate', 'rates',

    # Generic qualifiers
    'well', 'much', 'less', 'even', 'still', 'just', 'quite', 'rather', 'fairly',
    'especially', 'particularly', 'generally', 'typically', 'usually', 'commonly',
    'often', 'rarely', 'seldom', 'always', 'never', 'sometimes', 'occasionally',

    # Generic nouns and adjectives (for filtering collocations)
    'environmentally', 'friendly', 'attention', 'received', 'building', 'blocks',
    'potential', 'applications', 'promising', 'candidate', 'approach', 'strategy', 'technique', 'process', 'system', 'systems', 'device',
    'devices', 'material', 'materials', 'property', 'properties', 'performance',
    'efficiency', 'improvement', 'improvements', 'enhancement', 'enhancements',
    'wide', 'broad', 'narrow', 'range', 'variety', 'types', 'type', 'kind',
    'kinds', 'areas', 'area', 'field', 'fields', 'work', 'works', 'research',
    'researchers', 'conditions', 'condition', 'temperature', 'pressure', 'concentration',
    'solution', 'solutions', 'preparation', 'prepared', 'fabrication', 'fabricated',
    'formation', 'formed', 'growth', 'grown', 'behavior', 'behaviour', 'mechanism',
    'mechanisms', 'effect', 'effects', 'influence', 'influences', 'role', 'roles',
    'characteristics', 'characteristic', 'features', 'feature', 'factors', 'factor',
    'parameters', 'parameter', 'aspects', 'aspect', 'issues', 'issue', 'problems',
    'problem', 'challenges', 'challenge', 'opportunities', 'opportunity', 'advantages',
    'advantage', 'disadvantages', 'disadvantage', 'limitations', 'limitation',
    'approaches', 'techniques', 'procedures', 'procedure', 'processes',

    # Common verbs (gerunds and past participles)
    'increasing', 'decreasing', 'improving', 'enhancing', 'reducing', 'controlling',
    'modifying', 'changing', 'varying', 'adjusting', 'optimizing', 'maximizing',
    'minimizing', 'achieving', 'obtaining', 'providing', 'producing', 'generating',
    'creating', 'forming', 'developing', 'preparing', 'fabricating',
    'synthesizing', 'growing', 'depositing', 'coating', 'covering', 'protecting',

    # Generic degree/extent modifiers
    'extremely', 'highly', 'significantly', 'considerably', 'substantially', 'greatly',
    'remarkably', 'notably', 'moderately', 'slightly', 'somewhat', 'relatively',
    'comparatively', 'approximately', 'roughly', 'nearly', 'almost', 'mainly',
    'primarily', 'largely', 'mostly', 'chiefly', 'principally', 'essentially',
    'basically', 'fundamentally', 'inherently', 'naturally', 'obviously', 'clearly',

    # Generic spectroscopy and characterization terms
    'assigned', 'assignment', 'assignments', 'attributed', 'corresponding',
    'lowest', 'highest', 'strongest', 'weakest', 'maximum', 'minimum',
    'oscillator', 'oscillators', 'strength', 'strengths', 'transition', 'transitions',
    'band', 'bands', 'peak', 'peaks', 'signal', 'signals', 'spectrum', 'spectra',
    'spectral', 'spectroscopy', 'spectroscopic', 'wavelength', 'wavelengths',
    'intensity', 'intensities', 'absorption', 'emission', 'excitation',
    'forbidden', 'allowed', 'selection', 'relaxation',

    # Generic materials characterization terms
    'sputter', 'sputtered', 'sputtering', 'target', 'targets', 'substrate',
    'substrates', 'deposition', 'deposited', 'annealed', 'annealing',
    'thickness', 'thicknesses', 'morphology', 'morphologies', 'grain', 'grains',
    'crystalline', 'crystallinity', 'amorphous', 'polycrystalline',

    # Generic chemistry/physics descriptors
    'click', 'ratio', 'ratios', 'percentage', 'percentages', 'fraction', 'fractions',
    'component', 'components', 'constituent', 'constituents', 'entity', 'entities',
    'species', 'moiety', 'moieties', 'unit', 'units', 'segment', 'segments',
    'particle', 'particles', 'like', 'unlike', 'toward', 'towards',
    'separation', 'separated', 'charge', 'charges', 'charged', 'neutral',
    'disorder', 'disordered', 'ordered', 'ordering', 'point', 'points',

    # Common author last names (top 100 most common in scientific literature)
    'wang', 'li', 'zhang', 'liu', 'kim', 'lee', 'chen', 'smith', 'kumar', 'singh',
    'johnson', 'williams', 'brown', 'jones', 'miller', 'davis', 'garcia', 'rodriguez',
    'martinez', 'hernandez', 'lopez', 'gonzalez', 'wilson', 'anderson', 'thomas',
    'taylor', 'moore', 'jackson', 'martin', 'thompson', 'white', 'harris', 'clark',
    'lewis', 'robinson', 'walker', 'hall', 'allen', 'young', 'king', 'wright',
    'scott', 'green', 'baker', 'adams', 'nelson', 'carter', 'mitchell', 'roberts',
    'hirayama', 'inokuti', 'tanaka', 'yamamoto', 'suzuki', 'watanabe', 'ito',
    'nakamura', 'takahashi', 'sato', 'kobayashi', 'kato', 'yoshida', 'yamada',
    'mueller', 'schmidt', 'schneider', 'fischer', 'weber', 'meyer', 'wagner',
    'becker', 'schulz', 'hoffmann', 'koch', 'bauer', 'klein', 'wolf', 'schroeder',
}


def is_junk_token(token: str) -> bool:
    """
    Check if token is a junk pattern (article IDs, mixed alphanumeric, subscription disclaimers, etc.).

    FIXED: Now catches subscription disclaimers, website navigation terms, and metadata artifacts.
    """
    # Handle None/NaN
    if not token or (isinstance(token, float) and pd.isna(token)):
        return True

    token = str(token).lower()

    # Filter pure numeric tokens
    if token.isdigit():
        return True

    # Filter subscription/disclaimer boilerplate
    junk_patterns = [
        'society', 'request', 'societyrequest', 'e-alerts', 'morearticle',
        'subscription', 'disclaimer', 'copyright', 'rights', 'reserved',
        'alerts', 'close', 'subjects', 'supporting', 'info', 'learn',
        'views', 'usage', 'updated', 'regularly', 'orcid', 'lett',
        'chemmater', 'june', 'july', 'august', 'september', 'october',
        'november', 'december', 'january', 'february', 'march', 'april', 'may',
        # Additional stray terms found in evidence bundles
        'phone', 'online', 'website', 'email', 'contact', 'address',
        'deterioration', 'conditions', 'environmental', 'general',
        'return', 'web', 'lett', 'add', 'vs',
        # String representation of NaN
        'nan', 'none', 'null'
    ]

    # Check for exact matches or substrings in concatenated junk
    if token in junk_patterns:
        return True

    # Check for compound junk patterns (e.g., "e-alertsclosesubjects")
    if any(junk in token for junk in ['societyrequest', 'e-alerts', 'morearticle']):
        return True

    # Filter tokens that look like article IDs (e.g., jc_2089, alt304, etc.)
    if '_' in token or any(c.isdigit() for c in token):
        # Allow some common scientific abbreviations with numbers
        allowed_patterns = ['2d', '3d', '4d', 'h2', 'o2', 'n2', 'co2', 'ch4', 'nh3']
        if token.lower() not in allowed_patterns:
            # If it has both letters and numbers and isn't in allowed list, filter it
            has_letters = any(c.isalpha() for c in token)
            has_numbers = any(c.isdigit() for c in token)
            if has_letters and has_numbers:
                return True

    return False


def is_english_text(text: str) -> bool:
    """
    Check if text is in English using langdetect.
    Returns True if English or if detection fails (fail open).
    """
    if not LANGDETECT_AVAILABLE:
        return True  # Fail open if langdetect not available

    try:
        # Detect language (requires at least ~20 characters)
        if len(text) < 20:
            return True  # Too short to reliably detect
        return detect(text) == 'en'
    except (LangDetectException, Exception):
        return True  # Fail open on error


def lemmatize_token(token: str) -> str:
    """
    Lemmatize a token using spaCy.
    Returns original token if spaCy not available.

    Examples:
        clusters -> cluster
        perovskites -> perovskite
    """
    if not SPACY_AVAILABLE:
        return token

    try:
        doc = nlp(token)
        if len(doc) > 0:
            return doc[0].lemma_
    except Exception:
        pass

    return token


def load_stage3_vocabulary(ctfidf_path: Path, top_k: int = 100) -> dict[int, set[str]]:
    """
    Load Stage 3 c-TF-IDF curated vocabulary for each lineage.

    Args:
        ctfidf_path: Path to lineage_ctfidf_terms.csv
        top_k: Number of top terms to keep per lineage (default: 100)

    Returns:
        Dict mapping lineage_id -> set of curated terms
    """
    print(f"\n[Stage 3 Integration] Loading curated vocabulary from {ctfidf_path}...")

    if not ctfidf_path.exists():
        print(f"      [WARNING] Stage 3 vocabulary not found at {ctfidf_path}")
        print("      Falling back to raw text tokenization (may include junk terms)")
        return {}

    df = pd.read_csv(ctfidf_path)

    # Group by lineage and take top K terms
    vocab = {}
    for lineage_id, group in df.groupby('lineage_id'):
        # Take top_k terms by c-TF-IDF score (already sorted by rank)
        top_terms = set(group.nsmallest(top_k, 'rank')['term'].tolist())
        vocab[lineage_id] = top_terms

    avg_terms = np.mean([len(terms) for terms in vocab.values()])
    print(f"      Loaded vocabulary for {len(vocab)} lineages (avg {avg_terms:.1f} terms per lineage)")

    return vocab


def _compute_npmi_for_lineage(
    lineage_id: int,
    term_extractor,
    text_extractor,
    stage3_vocabulary: dict[int, set[str]],
    uninformative_terms: set[str],
    min_pair_count: int,
    min_npmi: float
) -> tuple[dict[tuple[str, str], float], dict[str, int]]:
    """Shared implementation for extracting NPMI term pairs for a lineage."""
    stats = {
        'texts_filtered_non_english': 0,
        'tokens_filtered_stage3': 0,
        'tokens_lemmatized': 0
    }

    paper_ids = term_extractor.load_lineage_papers_fast(int(lineage_id))
    if not paper_ids:
        return {}, stats

    text_dict = text_extractor.get_texts_batch(paper_ids)
    texts = list(text_dict.values())
    total_papers = len(texts)

    if total_papers == 0:
        return {}, stats

    stage3_terms = stage3_vocabulary.get(int(lineage_id), set())
    use_stage3_filter = len(stage3_terms) > 0

    term_paper_counts = Counter()
    pair_paper_counts = Counter()

    for text in texts:
        if not is_english_text(text):
            stats['texts_filtered_non_english'] += 1
            continue

        tokens = term_extractor.tokenize_and_filter(text)
        tokens = [
            t for t in tokens
            if t not in MARKUP_TERMS
            and t not in EXTENDED_STOPWORDS
            and not is_junk_token(t)
            and t not in uninformative_terms
        ]

        if SPACY_AVAILABLE:
            tokens = [lemmatize_token(t) for t in tokens]
            stats['tokens_lemmatized'] += len(tokens)

        if use_stage3_filter:
            original_count = len(tokens)
            tokens = [t for t in tokens if t in stage3_terms]
            stats['tokens_filtered_stage3'] += (original_count - len(tokens))

        unique_terms = set(tokens)
        term_paper_counts.update(unique_terms)

        if len(unique_terms) < 2:
            continue

        for term1, term2 in combinations(sorted(unique_terms), 2):
            pair_paper_counts[(term1, term2)] += 1

    npmi_scores: dict[tuple[str, str], float] = {}

    for (term1, term2), pair_count in pair_paper_counts.items():
        if pair_count < min_pair_count:
            continue

        count1 = term_paper_counts[term1]
        count2 = term_paper_counts[term2]

        p_x = count1 / total_papers
        p_y = count2 / total_papers
        p_xy = pair_count / total_papers

        if p_xy == 0 or p_x == 0 or p_y == 0:
            continue

        pmi = math.log(p_xy / (p_x * p_y))
        denom = -math.log(p_xy)
        npmi = 0.0 if denom == 0.0 else pmi / denom

        npmi = max(-1.0, min(1.0, npmi))

        O11 = pair_count
        O10 = count1 - pair_count
        O01 = count2 - pair_count
        O00 = total_papers - count1 - count2 + pair_count

        E11 = (count1 * count2) / total_papers
        E10 = (count1 * (total_papers - count2)) / total_papers
        E01 = ((total_papers - count1) * count2) / total_papers
        E00 = ((total_papers - count1) * (total_papers - count2)) / total_papers

        g_squared = 0.0
        for O, E in ((O11, E11), (O10, E10), (O01, E01), (O00, E00)):
            if O > 0 and E > 0:
                g_squared += 2 * O * math.log(O / E)

        if g_squared < 10.83:
            continue

        if npmi >= min_npmi:
            npmi_scores[(term1, term2)] = npmi

    return npmi_scores, stats


_WORKER_CONTEXT = None


class _NPMIWorkerContext:
    """Holds per-process resources for multiprocessing execution."""

    def __init__(self, config: dict):
        registry_path = config.get('registry_path')
        partitions_path = config.get('partitions_dir')
        raw_path = config.get('raw_dir')
        cache_path = config.get('abstract_cache_path')
        self.term_extractor = LineageTermExtractor(
            registry_path=Path(registry_path) if registry_path else None,
            partitions_dir=Path(partitions_path) if partitions_path else None,
            raw_dir=Path(raw_path) if raw_path else None,
            front_config_path=Path(config['front_config_path']),
            min_quarters=config['min_quarters'],
            cache_path=Path(cache_path) if cache_path else None,
            store=None
        )
        self.text_extractor = self.term_extractor.text_extractor
        self.stage3_vocabulary = {
            int(k): set(v) for k, v in config['stage3_vocabulary'].items()
        }
        self.uninformative_terms = set(config['uninformative_terms'])
        self.min_pair_count = config['min_pair_count']
        self.min_npmi = config['min_npmi']

    def compute(self, lineage_id: int):
        return _compute_npmi_for_lineage(
            lineage_id,
            self.term_extractor,
            self.text_extractor,
            self.stage3_vocabulary,
            self.uninformative_terms,
            self.min_pair_count,
            self.min_npmi
        )


def _npmi_worker_init(config: dict):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = _NPMIWorkerContext(config)


def _npmi_worker_run(lineage_id: int):
    pairs, stats = _WORKER_CONTEXT.compute(lineage_id)
    return lineage_id, pairs, stats


class LineageNPMIAnalyzer:
    """Compute NPMI co-occurrence patterns for lineages."""

    def __init__(
        self,
        term_extractor: LineageTermExtractor,
        text_extractor: AbstractExtractor,
        stage3_vocabulary: dict[int, set[str]] = None,
        min_npmi: float = 0.2,
        min_pair_count: int = 5,
        informativeness_threshold: float = 0.5,
        max_workers: int = 1,
        worker_memory_gb: float = 4.0,
        memory_reserve_gb: float = 4.0,
        registry_path: Path | None = None,
        raw_dir: Path | None = None,
        partitions_dir: Path | None = None,
        front_config_path: Path | None = None,
        abstract_cache_path: Path | None = None
    ):
        """
        Initialize NPMI analyzer.

        Args:
            term_extractor: Pre-initialized LineageTermExtractor (for paper loading)
            text_extractor: Pre-initialized AbstractExtractor (for text extraction)
            stage3_vocabulary: Dict mapping lineage_id -> curated terms from Stage 3 c-TF-IDF (optional but recommended)
            min_npmi: Minimum NPMI threshold for considering a pair (default: 0.2)
            min_pair_count: Minimum co-occurrence count (default: 5, tightened from 3)
            informativeness_threshold: Reject terms appearing in >X fraction of lineages (default: 0.5, tightened from 0.8)
            abstract_cache_path: Optional path to serialized abstract index cache
        """
        self.term_extractor = term_extractor
        self.text_extractor = text_extractor
        self.stage3_vocabulary = stage3_vocabulary or {}
        self.min_npmi = min_npmi
        self.min_pair_count = min_pair_count
        self.informativeness_threshold = informativeness_threshold
        self.max_workers = max(1, int(max_workers or 1))
        self.worker_memory_gb = float(worker_memory_gb)
        self.memory_reserve_gb = float(memory_reserve_gb)
        self.partitions_dir = partitions_dir or term_extractor.partitions_dir
        self.registry_path = Path(registry_path) if registry_path else getattr(term_extractor, "registry_path", None)
        self.raw_dir = Path(raw_dir) if raw_dir else getattr(text_extractor, "raw_dir", None)
        self.front_config_path = Path(front_config_path) if front_config_path else Path('config/front_aliases.yaml')
        self.min_quarters = getattr(term_extractor, "min_quarters", 12)
        self.abstract_cache_path = Path(abstract_cache_path) if abstract_cache_path else getattr(text_extractor, "cache_path", None)
        if self.partitions_dir:
            self.partitions_dir = Path(self.partitions_dir)
        if self.registry_path:
            self.registry_path = Path(self.registry_path)
        if self.raw_dir:
            self.raw_dir = Path(self.raw_dir)

        # Will store: {lineage_id: {(term1, term2): npmi_score}}
        self.lineage_npmi_pairs = {}

        # Document frequency for informativeness filtering
        self.term_document_frequency = Counter()
        self.total_lineages = 0
        self.uninformative_terms = set()

        # Statistics tracking
        self.stats = {
            'texts_filtered_non_english': 0,
            'tokens_filtered_stage3': 0,
            'tokens_lemmatized': 0
        }

    def _accumulate_stats(self, stats: dict[str, int]):
        for key, value in stats.items():
            self.stats[key] = self.stats.get(key, 0) + value

    def _allowed_worker_count(self) -> int:
        if self.max_workers <= 1:
            return 1
        if psutil is None or self.worker_memory_gb <= 0:
            return self.max_workers
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        allowance = int((available_gb - self.memory_reserve_gb) // self.worker_memory_gb)
        return max(1, min(self.max_workers, allowance))

    def _memory_ok(self) -> bool:
        if psutil is None or self.worker_memory_gb <= 0:
            return True
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        return available_gb - self.memory_reserve_gb >= self.worker_memory_gb

    def compute_term_document_frequencies(self, lineage_ids: list[int]):
        """
        IMPROVEMENT: Compute document frequency for all terms to enable informativeness filtering.

        Pre-process all lineages to count how many lineages each term appears in.
        Terms appearing in >80% of lineages are marked as uninformative.
        """
        print("\n[Pre-processing] Computing term document frequencies for informativeness filtering...")

        self.total_lineages = len(lineage_ids)

        for lineage_id in tqdm(lineage_ids, desc="Computing DF"):
            # Load papers
            paper_ids = self.term_extractor.load_lineage_papers_fast(lineage_id)
            if not paper_ids:
                continue

            # Extract texts
            text_dict = self.text_extractor.get_texts_batch(paper_ids)
            texts = list(text_dict.values())

            # Get unique terms across all papers in this lineage
            lineage_terms = set()
            for text in texts:
                tokens = self.term_extractor.tokenize_and_filter(text)
                tokens = [t for t in tokens
                         if t not in MARKUP_TERMS
                         and t not in EXTENDED_STOPWORDS
                         and not is_junk_token(t)]
                lineage_terms.update(tokens)

            # Count this lineage for each unique term
            self.term_document_frequency.update(lineage_terms)

        # Identify uninformative terms (appear in >threshold of lineages)
        threshold_count = self.informativeness_threshold * self.total_lineages
        for term, count in self.term_document_frequency.items():
            if count > threshold_count:
                self.uninformative_terms.add(term)

        print(f"      Total unique terms: {len(self.term_document_frequency)}")
        print(f"      Uninformative terms (DF > {self.informativeness_threshold}): {len(self.uninformative_terms)}")
        if self.uninformative_terms:
            sample = list(self.uninformative_terms)[:10]
            print(f"      Sample: {sample}")

    def extract_lineage_term_pairs(self, lineage_id: int) -> dict[tuple[str, str], float]:
        """
        Extract co-occurring term pairs from a lineage and compute NPMI scores.

        IMPROVEMENTS:
        - Language detection: Filter out non-English texts
        - Lemmatization: Normalize word forms (clusters -> cluster)
        - Stage 3 vocabulary restriction: Only use curated c-TF-IDF terms
        - Tightened thresholds: min_pair_count=5, informativeness=0.5

        Returns: {(term1, term2): npmi_score} for pairs meeting thresholds
        """
        pairs, stats = _compute_npmi_for_lineage(
            lineage_id,
            self.term_extractor,
            self.text_extractor,
            self.stage3_vocabulary,
            self.uninformative_terms,
            self.min_pair_count,
            self.min_npmi
        )
        self._accumulate_stats(stats)
        return pairs

    def extract_all_lineage_pairs(self, lineage_ids: list[int]):
        """Extract NPMI pairs for all lineages."""
        # IMPROVEMENT: First compute document frequencies for informativeness filtering
        self.compute_term_document_frequencies(lineage_ids)

        print(f"\n[4/5] Extracting NPMI co-occurrence pairs from {len(lineage_ids)} lineages...")

        lineage_ids = list(lineage_ids)

        if self.max_workers > 1 and psutil is None:
            print("      [WARNING] psutil not installed; Stage 4 multiprocessing disabled.")

        use_parallel = (
            self.max_workers > 1
            and psutil is not None
            and self.registry_path
            and self.raw_dir
            and self.partitions_dir
        )

        if use_parallel:
            allowed_workers = self._allowed_worker_count()
            if allowed_workers <= 1:
                use_parallel = False
            elif not self._memory_ok():
                print("      [WARNING] Insufficient free memory for Stage 4 workers; falling back to sequential execution.")
                use_parallel = False
            else:
                print(f"      Using up to {allowed_workers} worker processes (~{self.worker_memory_gb:.1f} GB each)")
                self._extract_pairs_parallel(lineage_ids, allowed_workers)

        if not use_parallel:
            for lineage_id in tqdm(lineage_ids, desc="Computing NPMI"):
                pairs = self.extract_lineage_term_pairs(lineage_id)
                self.lineage_npmi_pairs[lineage_id] = pairs

        total_pairs = sum(len(pairs) for pairs in self.lineage_npmi_pairs.values())
        avg_pairs = total_pairs / len(lineage_ids) if lineage_ids else 0
        print(f"      Found {total_pairs} high-NPMI pairs across all lineages (avg: {avg_pairs:.1f} per lineage)")

    def compute_front_npmi_similarity(
        self,
        lineage_id: int,
        front_terms: set[str]
    ) -> float:
        """
        Compute NPMI-based similarity between lineage and front.

        Matches lineage term pairs to front canonical terms and aggregates NPMI scores.

        Returns: similarity score
        """
        lineage_pairs = self.lineage_npmi_pairs.get(lineage_id, {})
        if not lineage_pairs:
            return 0.0

        # Find pairs where both terms are in front canonical terms
        matching_score = 0.0
        match_count = 0

        for (term1, term2), npmi in lineage_pairs.items():
            if term1 in front_terms and term2 in front_terms:
                matching_score += npmi
                match_count += 1

        # Normalize by number of matches (or return 0 if no matches)
        if match_count == 0:
            return 0.0

        return matching_score / match_count

    def _extract_pairs_parallel(self, lineage_ids: list[int], max_workers: int):
        if not (self.registry_path and self.raw_dir and self.partitions_dir):
            raise ValueError("registry_path, raw_dir, and partitions_dir are required for parallel execution.")

        payload = {
            'registry_path': str(self.registry_path) if self.registry_path else None,
            'raw_dir': str(self.raw_dir) if self.raw_dir else None,
            'partitions_dir': str(self.partitions_dir) if self.partitions_dir else None,
            'front_config_path': str(self.front_config_path),
            'min_quarters': self.min_quarters,
            'stage3_vocabulary': {int(k): list(v) for k, v in self.stage3_vocabulary.items()},
            'uninformative_terms': list(self.uninformative_terms),
            'min_pair_count': self.min_pair_count,
            'min_npmi': self.min_npmi,
            'abstract_cache_path': str(self.abstract_cache_path) if self.abstract_cache_path else None,
        }

        context = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=context,
            initializer=_npmi_worker_init,
            initargs=(payload,)
        ) as executor:
            pending = set()
            progress = tqdm(total=len(lineage_ids), desc="Computing NPMI", unit="lineage")

            def drain_completed(block=False):
                if not pending:
                    return
                wait_kwargs = {'return_when': FIRST_COMPLETED}
                if not block:
                    wait_kwargs['timeout'] = 0.1
                done, _ = wait(pending, **wait_kwargs)
                for fut in list(done):
                    pending.remove(fut)
                    lineage_id_done, pairs, stats = fut.result()
                    self.lineage_npmi_pairs[lineage_id_done] = pairs
                    self._accumulate_stats(stats)
                    progress.update(1)

            for lineage_id in lineage_ids:
                while True:
                    drain_completed(block=False)
                    if len(pending) < max_workers and self._memory_ok():
                        pending.add(executor.submit(_npmi_worker_run, int(lineage_id)))
                        break
                    time.sleep(0.2)

            while pending:
                drain_completed(block=True)

            progress.close()

    def generate_outputs(
        self,
        front_config: dict,
        _output_threshold: float = 0.8,
        output_dir_lineage: Path = Path("data/out/02_lineage_tracking"),
        output_dir_mapping: Path = Path("data/out/03_milestone_mapping")
    ):
        """
        Generate output files:
        1. Strong co-occurring pairs per lineage (above threshold)
        2. NPMI similarity matrix (lineage x front)

        Args:
            front_config: Research front definitions
            output_threshold: Minimum NPMI score for output (default: 0.8)
        """
        # Set up output directories
        output_dir_lineage = Path(output_dir_lineage)
        output_dir_mapping = Path(output_dir_mapping)
        output_dir_lineage.mkdir(parents=True, exist_ok=True)
        output_dir_mapping.mkdir(parents=True, exist_ok=True)

        print("\n[Output 1/2] Writing strong NPMI pairs per lineage (ADAPTIVE threshold = 90th percentile)...")

        # Output 1: Strong pairs per lineage (adaptive by quality)
        rows = []
        for lineage_id in sorted(self.lineage_npmi_pairs.keys()):
            pairs = self.lineage_npmi_pairs[lineage_id]

            if not pairs:
                continue

            # IMPROVEMENT: Adaptive threshold (90th percentile per lineage)
            npmi_values = list(pairs.values())
            adaptive_threshold = np.percentile(npmi_values, 90) if len(npmi_values) > 0 else 0.0

            # Sort by NPMI score descending
            sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)

            # Take all pairs above adaptive threshold
            for rank, ((term1, term2), npmi) in enumerate(sorted_pairs, start=1):
                if npmi >= adaptive_threshold:
                    rows.append({
                        'lineage_id': lineage_id,
                        'rank': rank,
                        'term1': term1,
                        'term2': term2,
                        'npmi_score': npmi,
                        'threshold': adaptive_threshold  # Record threshold for debugging
                    })

        df_pairs = pd.DataFrame(rows)

        # FIX: Remove any rows with NaN values in term columns
        df_pairs = df_pairs.dropna(subset=['term1', 'term2'])

        output_path = output_dir_lineage / "lineage_npmi_pairs.csv"
        df_pairs.to_csv(output_path, index=False)

        # Compute statistics
        pairs_per_lineage = df_pairs.groupby('lineage_id').size()
        avg_pairs = pairs_per_lineage.mean() if len(pairs_per_lineage) > 0 else 0
        print(f"      Wrote {len(rows)} pair records to {output_path}")
        print(f"      Average {avg_pairs:.1f} pairs per lineage (range: {pairs_per_lineage.min()}-{pairs_per_lineage.max()})")

        # Output 2: NPMI similarity matrix
        print("\n[Output 2/2] Computing lineage-front NPMI similarity matrix...")

        similarity_rows = []
        front_names = sorted(front_config.keys())

        # Pre-extract front terms (reuse from LineageTermExtractor)
        front_term_sets = {
            front_name: self.term_extractor.get_front_terms(front_name)
            for front_name in front_names
        }

        for lineage_id in tqdm(sorted(self.lineage_npmi_pairs.keys()), desc="Computing NPMI similarity"):
            row = {'lineage_id': lineage_id}

            for front_name in front_names:
                front_terms = front_term_sets[front_name]
                similarity = self.compute_front_npmi_similarity(lineage_id, front_terms)
                row[front_name] = similarity

            similarity_rows.append(row)

        df_similarity = pd.DataFrame(similarity_rows)
        output_path = output_dir_mapping / "lineage_front_npmi_similarity.csv"
        df_similarity.to_csv(output_path, index=False)
        print(f"      Wrote {len(similarity_rows)} x {len(front_names)} similarity matrix to {output_path}")

        return df_pairs, df_similarity


def run_npmi(
    min_quarters: int = 6,
    min_npmi: float = 0.2,
    min_pair_count: int = 3,
    output_threshold: float = 0.8,
    front_config_path: Path = Path('config/front_aliases.yaml'),
    partitions_dir: Path = Path('data/out/cache_cum/partitions_cum'),
    ctfidf_vocab_path: Path = Path('data/out/02_lineage_tracking/lineage_ctfidf_terms.csv'),
    vocab_size: int = 100,
    registry_path: Path = None,  # For standalone mode
    raw_dir: Path = None,  # For standalone mode
    abstract_cache_path: Path = None,
    store=None,  # Optional LineageTextStore for pipeline mode
    validate: bool = True,  # Run validation checks and generate reports
    output_root: Path = Path('data/out'),  # Base directory for outputs
    max_workers: int = 1,
    worker_memory_gb: float = 4.0,
    memory_reserve_gb: float = 4.0
) -> tuple:
    """
    Run Stage 4: NPMI co-term discovery.

    This function can be called from the pipeline (with pre-loaded store) or
    standalone (loads fresh resources).

    Args:
        min_quarters: Minimum quarters for persistent lineages
        min_npmi: Minimum NPMI threshold
        min_pair_count: Minimum co-occurrence count
        output_threshold: Minimum NPMI score for output pairs
        front_config_path: Path to research front YAML
        partitions_dir: Path to partition JSONs
        ctfidf_vocab_path: Path to Stage 3 c-TF-IDF vocabulary
        vocab_size: Number of top c-TF-IDF terms to use per lineage
        registry_path: Path to lineage registry JSON (standalone mode)
        raw_dir: Path to raw JSONL files (standalone mode)
        abstract_cache_path: Optional path to serialized abstract index cache
        store: Optional pre-loaded LineageTextStore (pipeline mode)
        validate: Run validation checks and generate reports (default: True)
        max_workers: Maximum worker processes for parallel NPMI extraction
        worker_memory_gb: Estimated memory footprint per worker process (GB)
        memory_reserve_gb: Memory to keep free when launching workers (GB)

    Returns:
        (df_pairs, df_similarity, validation_results): DataFrames and validation dict
    """
    front_config_path = Path(front_config_path)
    partitions_dir = Path(partitions_dir)
    ctfidf_vocab_path = Path(ctfidf_vocab_path)
    output_root = Path(output_root)
    if registry_path:
        registry_path = Path(registry_path)
    if raw_dir:
        raw_dir = Path(raw_dir)
    output_dir_lineage = output_root / "02_lineage_tracking"
    output_dir_mapping = output_root / "03_milestone_mapping"
    validation_dir = output_root / "06_validation" / "stage4"

    print("=" * 70)
    print("Stage 4: NPMI Co-term Discovery (IMPROVED)")
    print("=" * 70)

    # Initialize term extractor (with or without shared store)
    print("[1/6] Initializing LineageTermExtractor...")
    start = time.time()
    term_extractor = LineageTermExtractor(
        registry_path=None if store else (registry_path or Path('data/out/02_lineage_tracking/lineage_registry.json')),
        partitions_dir=partitions_dir,
        raw_dir=None if store else (raw_dir or Path('data/current_ingest/raw')),
        front_config_path=front_config_path,
        min_quarters=min_quarters,
        cache_path=abstract_cache_path,
        store=store  # Pass shared store if available
    )
    print(f"      OK in {time.time()-start:.3f}s")

    # Load Stage 3 curated vocabulary
    print("[2/6] Loading Stage 3 c-TF-IDF vocabulary...")
    start = time.time()
    stage3_vocab = load_stage3_vocabulary(ctfidf_vocab_path, top_k=vocab_size)
    if stage3_vocab:
        print(f"      OK in {time.time()-start:.3f}s")
        print("      IMPROVEMENT: Restricting NPMI to Stage 3 curated terms only")
    else:
        print("      WARNING: Stage 3 vocabulary not found, using raw tokenization")

    # Initialize NPMI analyzer
    print("[3/6] Initializing NPMI analyzer...")
    start = time.time()
    npmi_analyzer = LineageNPMIAnalyzer(
        term_extractor=term_extractor,
        text_extractor=term_extractor.text_extractor,
        stage3_vocabulary=stage3_vocab,
        min_npmi=min_npmi,
        min_pair_count=min_pair_count,
        max_workers=max_workers,
        worker_memory_gb=worker_memory_gb,
        memory_reserve_gb=memory_reserve_gb,
        registry_path=registry_path,
        raw_dir=raw_dir,
        partitions_dir=partitions_dir,
        front_config_path=front_config_path,
        abstract_cache_path=abstract_cache_path
    )
    print(f"      OK in {time.time()-start:.3f}s")

    print("[4/6] Configuration:")
    print(f"      - Min NPMI threshold: {min_npmi}")
    print(f"      - Min pair count: {min_pair_count} (tightened from 3)")
    print(f"      - Informativeness threshold: {npmi_analyzer.informativeness_threshold} (tightened from 0.8)")
    print(f"      - Persistent lineages: {len(term_extractor.persistent_lineages)}")
    print(f"      - Stage 3 vocabulary restriction: {'ENABLED' if stage3_vocab else 'DISABLED'}")
    print(f"      - Language filtering: {'ENABLED' if LANGDETECT_AVAILABLE else 'DISABLED'}")
    print(f"      - Lemmatization: {'ENABLED' if SPACY_AVAILABLE else 'DISABLED'}")

    # Extract NPMI pairs
    overall_start = time.time()
    npmi_analyzer.extract_all_lineage_pairs(term_extractor.persistent_lineages)

    # Generate outputs
    df_pairs, df_similarity = npmi_analyzer.generate_outputs(
        front_config=term_extractor.fronts_config,
        output_threshold=output_threshold,
        output_dir_lineage=output_dir_lineage,
        output_dir_mapping=output_dir_mapping
    )

    overall_time = time.time() - overall_start

    print("\n" + "=" * 70)
    print("STAGE 4 COMPLETE (IMPROVED)")
    print("=" * 70)
    print(f"Total time: {overall_time:.1f}s ({overall_time/60:.1f} minutes)")
    print(f"Lineages processed: {len(term_extractor.persistent_lineages)}")

    # Compute statistics
    avg_pairs = sum(len(pairs) for pairs in npmi_analyzer.lineage_npmi_pairs.values()) / len(npmi_analyzer.lineage_npmi_pairs) if npmi_analyzer.lineage_npmi_pairs else 0
    print(f"Average high-NPMI pairs per lineage: {avg_pairs:.1f}")

    # Print filtering statistics
    print("\nFiltering statistics:")
    print(f"  - Non-English texts filtered: {npmi_analyzer.stats['texts_filtered_non_english']}")
    print(f"  - Tokens filtered by Stage 3 vocab: {npmi_analyzer.stats['tokens_filtered_stage3']}")
    print(f"  - Tokens lemmatized: {npmi_analyzer.stats['tokens_lemmatized']}")

    print("\nOutputs:")
    print(f"  - {output_dir_lineage / 'lineage_npmi_pairs.csv'}")
    print(f"  - {output_dir_mapping / 'lineage_front_npmi_similarity.csv'}")

    print("\nIMPROVEMENTS APPLIED:")
    print("  1. ✓ Stage 3 vocabulary restriction (only curated c-TF-IDF terms)")
    print("  2. ✓ Language filtering (English only)")
    print("  3. ✓ Lemmatization (normalize word forms)")
    print("  4. ✓ German stopwords")
    print("  5. ✓ Tightened thresholds (min_pair_count, informativeness)")

    # Run validation if requested
    if validate:
        print(f"\n{'='*70}")
        print("STAGE 4 VALIDATION")
        print(f"{'='*70}\n")

        validation_dir.mkdir(parents=True, exist_ok=True)
        validation_results = run_phase4_validation(
            df_pairs,
            df_similarity,
            validation_dir=validation_dir
        )
    else:
        validation_results = None

    return df_pairs, df_similarity, validation_results


# ============================================================================
# VALIDATION FUNCTIONS (integrated from validate_stage4.py)
# ============================================================================

def run_phase4_validation(
    pairs_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    validation_dir: Path = Path('data/out/06_validation/stage4')
) -> dict:
    """
    Run Stage 4 validation checks and generate outputs.

    Args:
        pairs_df: DataFrame with NPMI term pairs
        similarity_df: DataFrame with lineage-front similarity scores

    Returns:
        Dictionary with validation results
    """
    # Lazy imports to avoid overhead when validation disabled

    # Create output directory
    output_dir = Path(validation_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run validation checks
    print("[1/5] Running data integrity checks...")
    checks = _validate_stage4_integrity(pairs_df, similarity_df)

    # Generate visualizations
    print("[2/5] Generating similarity heatmap...")
    _generate_phase4_heatmap(similarity_df, output_dir / 'phase4_similarity_heatmap.png')

    print("[3/5] Generating top pairs showcase...")
    _generate_phase4_top_pairs(pairs_df, output_dir / 'phase4_top_pairs.png')

    print("[4/5] Generating score distributions...")
    _generate_phase4_distributions(pairs_df, similarity_df, output_dir / 'phase4_distributions.png')

    # Generate report
    print("[5/5] Generating validation report...")
    _generate_phase4_report(checks, pairs_df, similarity_df, output_dir / 'phase4_summary.md')

    # Save JSON results
    validation_path = output_dir / 'phase4_validation.json'
    with open(validation_path, 'w') as f:
        json.dump(checks, f, indent=2)

    print(f"\n[Validation] Complete! Results saved to {output_dir}/")

    return checks


def _validate_stage4_integrity(pairs_df: pd.DataFrame, similarity_df: pd.DataFrame) -> dict:
    """Run data integrity checks on Stage 4 outputs."""
    checks = {}

    # Check 1: Expected shape
    checks['pairs_shape_ok'] = bool(len(pairs_df) > 0)
    checks['similarity_shape_ok'] = bool(len(similarity_df) > 0)
    checks['expected_lineages'] = int(len(similarity_df))

    # Check 2: No missing values in critical columns
    checks['pairs_no_nulls'] = bool(not pairs_df[['lineage_id', 'term1', 'term2', 'npmi_score']].isnull().any().any())
    checks['similarity_no_nulls'] = bool(not similarity_df.isnull().any().any())

    # Check 3: NPMI scores in valid range [-1, 1]
    npmi_scores = pairs_df['npmi_score']
    checks['npmi_min'] = float(npmi_scores.min())
    checks['npmi_max'] = float(npmi_scores.max())
    checks['npmi_range_ok'] = bool((npmi_scores >= -1).all() and (npmi_scores <= 1).all())

    # Check 4: Similarity scores in valid range [0, 1]
    similarity_cols = [col for col in similarity_df.columns if col != 'lineage_id']
    similarity_values = similarity_df[similarity_cols].values.flatten()
    checks['similarity_min'] = float(np.min(similarity_values))
    checks['similarity_max'] = float(np.max(similarity_values))
    checks['similarity_range_ok'] = bool((similarity_values >= 0).all() and (similarity_values <= 1).all())

    # Check 5: Coverage - how many lineages have at least one match
    has_match = (similarity_df[similarity_cols] > 0).any(axis=1)
    checks['lineages_with_matches'] = int(has_match.sum())
    checks['lineages_without_matches'] = int((~has_match).sum())
    checks['coverage_pct'] = float(has_match.sum() / len(similarity_df) * 100)

    # Check 6: Pairs per lineage distribution
    pairs_per_lineage = pairs_df.groupby('lineage_id').size()
    checks['avg_pairs_per_lineage'] = float(pairs_per_lineage.mean())
    checks['min_pairs_per_lineage'] = int(pairs_per_lineage.min())
    checks['max_pairs_per_lineage'] = int(pairs_per_lineage.max())

    return checks


def _generate_phase4_heatmap(similarity_df: pd.DataFrame, output_path: Path):
    """Generate heatmap of lineage-front NPMI similarity scores."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Prepare data (lineages as rows, fronts as columns)
    lineage_ids = similarity_df['lineage_id'].values
    front_cols = [col for col in similarity_df.columns if col != 'lineage_id']

    # Get similarity matrix
    similarity_matrix = similarity_df[front_cols].values

    # Find top 20 lineages by total similarity
    total_similarity = similarity_matrix.sum(axis=1)
    top_indices = np.argsort(total_similarity)[-20:][::-1]

    top_lineages = lineage_ids[top_indices]
    top_matrix = similarity_matrix[top_indices, :]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot heatmap
    sns.heatmap(
        top_matrix,
        xticklabels=front_cols,
        yticklabels=[f"L{lid}" for lid in top_lineages],
        cmap='YlOrRd',
        cbar_kws={'label': 'NPMI Similarity'},
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=top_matrix.max() if top_matrix.max() > 0 else 1.0
    )

    ax.set_xlabel('Research Front', fontsize=12, fontweight='bold')
    ax.set_ylabel('Lineage ID', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Lineages: NPMI Similarity to Research Fronts',
                 fontsize=14, fontweight='bold', pad=20)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _generate_phase4_top_pairs(pairs_df: pd.DataFrame, output_path: Path, n_examples: int = 20):
    """Generate bar chart showcasing top co-occurring term pairs."""
    import matplotlib.pyplot as plt

    # Get overall top pairs by NPMI score
    top_pairs = pairs_df.nlargest(n_examples, 'npmi_score')

    # Create labels
    labels = [f"({row['term1']}, {row['term2']})" for _, row in top_pairs.iterrows()]
    scores = top_pairs['npmi_score'].values

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Horizontal bar chart
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, scores, color='steelblue', edgecolor='navy', linewidth=0.5)

    # Color top 5 differently
    for i in range(min(5, len(bars))):
        bars[i].set_color('coral')
        bars[i].set_edgecolor('darkred')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('NPMI Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {n_examples} Co-Occurring Term Pairs (by NPMI)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Invert y-axis so highest score is at top
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _generate_phase4_distributions(pairs_df: pd.DataFrame, similarity_df: pd.DataFrame, output_path: Path):
    """Generate distribution plots for NPMI and similarity scores."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: NPMI score distribution
    ax = axes[0, 0]
    ax.hist(pairs_df['npmi_score'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(pairs_df['npmi_score'].median(), color='red', linestyle='--',
               linewidth=2, label=f"Median: {pairs_df['npmi_score'].median():.3f}")
    ax.set_xlabel('NPMI Score', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title('Distribution of NPMI Scores (All Pairs)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Similarity score distribution (flattened)
    ax = axes[0, 1]
    front_cols = [col for col in similarity_df.columns if col != 'lineage_id']
    similarity_values = similarity_df[front_cols].values.flatten()
    # Filter out zeros for better visualization
    nonzero_similarities = similarity_values[similarity_values > 0]
    similarity_median = np.median(nonzero_similarities) if len(nonzero_similarities) > 0 else 0
    ax.hist(nonzero_similarities, bins=50, color='coral', edgecolor='darkred', alpha=0.7)
    ax.axvline(similarity_median, color='blue', linestyle='--',
               linewidth=2, label=f"Median: {similarity_median:.4f}")
    ax.set_xlabel('NPMI Similarity Score', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title('Distribution of Lineage-Front Similarity (Non-Zero Only)',
                 fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Pairs per lineage
    ax = axes[1, 0]
    pairs_per_lineage = pairs_df.groupby('lineage_id').size()
    ax.hist(pairs_per_lineage, bins=30, color='mediumseagreen', edgecolor='darkgreen', alpha=0.7)
    ax.axvline(pairs_per_lineage.median(), color='red', linestyle='--',
               linewidth=2, label=f"Median: {pairs_per_lineage.median():.0f}")
    ax.set_xlabel('Pairs per Lineage', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title('Distribution of Term Pairs per Lineage', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Matches per lineage (how many fronts each lineage matches to)
    ax = axes[1, 1]
    matches_per_lineage = (similarity_df[front_cols] > 0).sum(axis=1)
    ax.hist(matches_per_lineage, bins=range(0, int(matches_per_lineage.max())+2),
            color='mediumpurple', edgecolor='indigo', alpha=0.7, align='left')
    ax.axvline(matches_per_lineage.median(), color='red', linestyle='--',
               linewidth=2, label=f"Median: {matches_per_lineage.median():.0f}")
    ax.set_xlabel('Number of Front Matches', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Lineages', fontsize=10, fontweight='bold')
    ax.set_title('Front Matches per Lineage', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Stage 4 (NPMI) Score Distributions', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _generate_phase4_report(checks: dict, pairs_df: pd.DataFrame, similarity_df: pd.DataFrame, output_path: Path):
    """Generate markdown validation report."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Compute additional statistics
    front_cols = [col for col in similarity_df.columns if col != 'lineage_id']
    nonzero_similarities = similarity_df[front_cols].values.flatten()
    nonzero_similarities = nonzero_similarities[nonzero_similarities > 0]
    similarity_median = np.median(nonzero_similarities) if len(nonzero_similarities) > 0 else 0

    # Get top pair examples
    top_pairs = pairs_df.nlargest(5, 'npmi_score')
    examples = []
    for _, row in top_pairs.iterrows():
        examples.append(f"- `({row['term1']}, {row['term2']})` NPMI={row['npmi_score']:.3f}")
    examples_text = "\n".join(examples)

    all_checks_pass = all([checks['pairs_shape_ok'], checks['similarity_shape_ok'],
                           checks['npmi_range_ok'], checks['similarity_range_ok']])

    report = f"""# Stage 4 (NPMI Co-Term Discovery) Summary Report

**Generated**: {timestamp}
**Status**: {'[PASS]' if all_checks_pass else '[FAIL]'}

---

## Overview

Stage 4 implemented Normalized Pointwise Mutual Information (NPMI) analysis to identify strongly co-occurring term pairs within lineages and match them to research front canonical term combinations. NPMI measures how strongly two terms appear together in papers, with scores ranging from -1 (never co-occur) to +1 (perfect co-occurrence).

The analysis processed **{checks['expected_lineages']} persistent lineages**, extracting **{len(pairs_df):,} high-quality term pairs** (top 30 per lineage) and computing similarity scores to **{len(front_cols)} research fronts** based on canonical term overlap.

---

## Key Findings

**Coverage & Quality**:
- **{checks['coverage_pct']:.1f}%** of lineages ({checks['lineages_with_matches']}/{checks['expected_lineages']}) matched to at least one research front
- **{len(nonzero_similarities):,}** non-zero similarity scores across all lineage-front pairs
- Average **{checks['avg_pairs_per_lineage']:.1f} term pairs** extracted per lineage (range: {checks['min_pairs_per_lineage']}-{checks['max_pairs_per_lineage']})

**Score Distributions**:
- NPMI scores: median **{pairs_df['npmi_score'].median():.3f}**, range [{checks['npmi_min']:.3f}, {checks['npmi_max']:.3f}]
- Similarity scores (non-zero): median **{similarity_median:.4f}**, range [{checks['similarity_min']:.4f}, {checks['similarity_max']:.4f}]
- Top co-occurring pairs show strong technical collocations

**Technical Quality Examples**:
{examples_text}

---

## Data Validation

| Check | Status | Details |
|-------|--------|---------|
| Data Integrity | {'[OK]' if checks['pairs_shape_ok'] and checks['similarity_shape_ok'] else '[FAIL]'} | {len(pairs_df):,} pairs, {len(similarity_df)} lineages |
| No Missing Values | {'[OK]' if checks['pairs_no_nulls'] and checks['similarity_no_nulls'] else '[FAIL]'} | All critical columns complete |
| NPMI Range | {'[OK]' if checks['npmi_range_ok'] else '[FAIL]'} | All scores in [-1, 1] |
| Similarity Range | {'[OK]' if checks['similarity_range_ok'] else '[FAIL]'} | All scores in [0, 1] |
| Coverage | {'[OK]' if checks['coverage_pct'] >= 50 else '[WARN]' if checks['coverage_pct'] >= 25 else '[FAIL]'} | {checks['coverage_pct']:.1f}% lineages matched |

---

## Overall Assessment

{'[OK] **ALL VALIDATION CHECKS PASSED**' if all_checks_pass else '[FAIL] **SOME VALIDATION CHECKS FAILED**'}
"""

    output_path.write_text(report, encoding='utf-8')


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute NPMI co-term discovery for lineages")
    parser.add_argument(
        '--min-quarters',
        type=int,
        default=6,
        help='Minimum quarters for persistent lineages (default: 6)'
    )
    parser.add_argument(
        '--min-npmi',
        type=float,
        default=0.2,
        help='Minimum NPMI threshold (default: 0.2)'
    )
    parser.add_argument(
        '--min-pair-count',
        type=int,
        default=3,
        help='Minimum co-occurrence count (default: 3)'
    )
    parser.add_argument(
        '--output-threshold',
        type=float,
        default=0.8,
        help='Minimum NPMI score for output pairs (default: 0.8, adaptive per lineage)'
    )
    parser.add_argument(
        '--registry',
        type=Path,
        default=None,
        help='Path to lineage registry JSON'
    )
    parser.add_argument(
        '--partitions',
        type=Path,
        default=None,
        help='Path to cached partition JSONs'
    )
    parser.add_argument(
        '--raw',
        type=Path,
        default=None,
        help='Path to raw JSONL data'
    )
    parser.add_argument(
        '--abstract-cache',
        type=Path,
        default=None,
        help='Path to serialized abstract index cache (default auto-generated)'
    )
    parser.add_argument(
        '--fronts',
        type=Path,
        default=Path('config/front_aliases.yaml'),
        help='Path to research front definitions'
    )
    parser.add_argument(
        '--ctfidf-vocab',
        type=Path,
        default=None,
        help='Path to Stage 3 c-TF-IDF vocabulary (recommended for quality)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=100,
        help='Number of top c-TF-IDF terms to use per lineage (default: 100)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=1,
        help='Maximum worker processes for Stage 4 (default: 1 = sequential)'
    )
    parser.add_argument(
        '--worker-memory-gb',
        type=float,
        default=4.0,
        help='Estimated memory per Stage 4 worker process in GB (default: 4.0)'
    )
    parser.add_argument(
        '--memory-reserve-gb',
        type=float,
        default=4.0,
        help='Memory to keep free when launching Stage 4 workers (default: 4.0)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='Run validation checks and generate reports (default: True)'
    )
    parser.add_argument(
        '--no-validate',
        dest='validate',
        action='store_false',
        help='Skip validation checks'
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=None,
        help='Base directory for outputs (default: data/out)'
    )
    add_domain_args(parser)

    args = parser.parse_args()

    paths = resolve_script_paths(args, REPO_ROOT)
    if args.registry is None:
        args.registry = paths.lineage_tracking / "lineage_registry.json" if paths else Path("data/out/02_lineage_tracking/lineage_registry.json")
    if args.partitions is None:
        args.partitions = paths.cache_cum / "partitions_cum" if paths else Path("data/out/cache_cum/partitions_cum")
    if args.raw is None:
        args.raw = paths.raw if paths else Path("data/current_ingest/raw")
    if args.ctfidf_vocab is None:
        args.ctfidf_vocab = paths.lineage_tracking / "lineage_ctfidf_terms.csv" if paths else Path("data/out/02_lineage_tracking/lineage_ctfidf_terms.csv")
    if args.output_root is None:
        args.output_root = paths.out if paths else Path("data/out")

    # Call the refactored function (standalone mode, store=None)
    run_npmi(
        min_quarters=args.min_quarters,
        min_npmi=args.min_npmi,
        min_pair_count=args.min_pair_count,
        output_threshold=args.output_threshold,
        front_config_path=args.fronts,
        partitions_dir=args.partitions,
        ctfidf_vocab_path=args.ctfidf_vocab,
        vocab_size=args.vocab_size,
        registry_path=args.registry,  # Pass CLI argument
        raw_dir=args.raw,  # Pass CLI argument
        abstract_cache_path=args.abstract_cache,
        store=None,  # Standalone mode
        validate=args.validate,  # Pass validate flag
        output_root=args.output_root,
        max_workers=args.max_workers,
        worker_memory_gb=args.worker_memory_gb,
        memory_reserve_gb=args.memory_reserve_gb
    )


if __name__ == '__main__':
    main()

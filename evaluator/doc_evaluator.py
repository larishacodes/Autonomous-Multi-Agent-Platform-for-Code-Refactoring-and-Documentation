# evaluator/doc_evaluator.py
#
# Block 6b: Documentation Evaluator
# Evaluates the quality of generated Javadoc documentation using:
#   - Coverage    : % of methods that have documentation
#   - Completeness: Are parameters, return values, and overview present?
#   - Length      : Is the documentation an appropriate length?
#   - BLEU-4      : n-gram overlap with reference (if available)
#   - ROUGE-L     : Longest common subsequence with reference (if available)
#   - Confidence  : Combined weighted score

import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import BLEU/ROUGE — provide fallbacks if not installed
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# Coverage
# ═══════════════════════════════════════════════════════════════════════════════
def compute_coverage(documentation: str, parsed_code: dict) -> Dict[str, Any]:
    """
    Compute what percentage of Java methods are documented.
    Checks if each method name appears in the documentation.
    """
    functions = parsed_code.get("functions", [])
    if not functions:
        return {"score": 1.0, "covered": 0, "total": 0, "missing": []}

    covered = []
    missing = []
    doc_lower = documentation.lower()

    for func in functions:
        name = func.get("name", "")
        if name.lower() in doc_lower:
            covered.append(name)
        else:
            missing.append(name)

    score = len(covered) / len(functions) if functions else 0.0

    return {
        "score":              round(score, 3),
        "covered":            len(covered),
        "total":              len(functions),
        "covered_functions":  covered,
        "missing_functions":  missing,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Completeness
# ═══════════════════════════════════════════════════════════════════════════════
def compute_completeness(documentation: str, parsed_code: dict) -> Dict[str, Any]:
    """
    Check if documentation includes key Java doc sections:
    - Overview/description
    - Parameters mentioned (@param or parameter table)
    - Return values mentioned (@return or return table)
    - Method-level sections (### headers)
    - Javadoc indicators
    """
    doc_lower = documentation.lower()

    checks = {
        "has_overview":       any(kw in doc_lower for kw in
                                  ["overview", "description", "this module", "this class", "##"]),
        "has_parameters":     any(kw in doc_lower for kw in
                                  ["@param", "param", "parameter", "parameters", "arguments"]),
        "has_return_info":    any(kw in doc_lower for kw in
                                  ["@return", "return", "returns", "output", "result"]),
        "has_method_docs":    "###" in documentation,
        "has_javadoc_style":  any(kw in doc_lower for kw in
                                  ["@param", "@return", "@throws", "/**", "javadoc",
                                   "lines of code", "parameters |"]),
    }

    score = sum(checks.values()) / len(checks)

    return {
        "score":   round(score, 3),
        "checks":  checks,
        "passed":  sum(checks.values()),
        "total":   len(checks),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BLEU-4
# ═══════════════════════════════════════════════════════════════════════════════
def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Compute BLEU-4 score between reference and generated documentation.
    Falls back to word overlap if NLTK not available.
    """
    if not reference:
        return 0.0

    if BLEU_AVAILABLE:
        try:
            ref_tokens = reference.lower().split()
            hyp_tokens = hypothesis.lower().split()
            smoothing  = SmoothingFunction().method1
            return round(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing), 3)
        except Exception as e:
            logger.debug(f"BLEU calculation failed: {e}")

    # Fallback: simple word overlap
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())
    if not ref_words:
        return 0.0
    return round(len(ref_words & hyp_words) / len(ref_words), 3)


# ═══════════════════════════════════════════════════════════════════════════════
# ROUGE-L
# ═══════════════════════════════════════════════════════════════════════════════
def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """
    Compute ROUGE-L score between reference and generated documentation.
    Falls back to LCS-based estimate if rouge_score not available.
    """
    if not reference:
        return 0.0

    if ROUGE_AVAILABLE:
        try:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            result = scorer.score(reference, hypothesis)
            return round(result["rougeL"].fmeasure, 3)
        except Exception as e:
            logger.debug(f"ROUGE-L calculation failed: {e}")

    # Fallback: character-level estimate
    a = reference.lower()
    b = hypothesis.lower()
    if not a or not b:
        return 0.0
    common = sum(1 for c in b if c in a)
    return round(min((2 * common) / (len(a) + len(b)), 1.0), 3)


# ═══════════════════════════════════════════════════════════════════════════════
# Length and readability
# ═══════════════════════════════════════════════════════════════════════════════
def compute_length_score(documentation: str, parsed_code: dict) -> Dict[str, Any]:
    """
    Check if documentation length is appropriate for the number of methods.
    Target: 20–100 words per method.
    """
    word_count = len(documentation.split())
    line_count = len(documentation.splitlines())
    func_count = len(parsed_code.get("functions", []))

    expected_min = max(20, func_count * 20)
    expected_max = max(200, func_count * 100)

    if expected_min <= word_count <= expected_max:
        score = 1.0
    elif word_count < expected_min:
        score = word_count / expected_min
    else:
        score = expected_max / word_count

    return {
        "score":        round(score, 3),
        "word_count":   word_count,
        "line_count":   line_count,
        "expected_min": expected_min,
        "expected_max": expected_max,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Confidence score
# ═══════════════════════════════════════════════════════════════════════════════
def compute_doc_confidence(coverage: Dict, completeness: Dict,
                           length: Dict, bleu: float, rouge_l: float,
                           has_reference: bool) -> Dict[str, Any]:
    """
    Combine all doc metrics into a single confidence score.
    """
    components = {
        "coverage":     coverage["score"],
        "completeness": completeness["score"],
        "length":       length["score"],
    }
    weights = {
        "coverage":     0.35,
        "completeness": 0.35,
        "length":       0.30,
    }

    if has_reference:
        components["bleu"]    = bleu
        components["rouge_l"] = rouge_l
        weights = {
            "coverage":     0.20,
            "completeness": 0.20,
            "length":       0.15,
            "bleu":         0.20,
            "rouge_l":      0.25,
        }

    score = max(0.0, min(1.0, sum(weights[k] * components[k] for k in components)))

    if score >= 0.85:   status = "✅ EXCELLENT"
    elif score >= 0.75: status = "✅ GOOD"
    elif score >= 0.65: status = "⚠ MODERATE"
    elif score >= 0.50: status = "⚠ MARGINAL"
    else:               status = "❌ POOR"

    return {
        "score":      round(score, 3),
        "status":     status,
        "components": {k: round(v, 3) for k, v in components.items()},
        "weights":    weights,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation function
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_documentation(documentation: str, parsed_code: dict,
                            reference_doc: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate the quality of generated Java documentation.

    Args:
        documentation: The generated documentation string
        parsed_code:   The parsed AST structure of the Java code
        reference_doc: Optional reference documentation for BLEU/ROUGE

    Returns:
        Full evaluation results dict
    """
    has_reference = reference_doc is not None and len(reference_doc.strip()) > 0

    coverage     = compute_coverage(documentation, parsed_code)
    completeness = compute_completeness(documentation, parsed_code)
    length       = compute_length_score(documentation, parsed_code)
    bleu         = compute_bleu(reference_doc or "", documentation) if has_reference else 0.0
    rouge_l      = compute_rouge_l(reference_doc or "", documentation) if has_reference else 0.0
    confidence   = compute_doc_confidence(coverage, completeness, length,
                                          bleu, rouge_l, has_reference)

    return {
        "coverage":      coverage,
        "completeness":  completeness,
        "length":        length,
        "bleu":          bleu,
        "rouge_l":       rouge_l,
        "has_reference": has_reference,
        "confidence":    confidence,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Report saver
# ═══════════════════════════════════════════════════════════════════════════════
def save_doc_evaluation_report(analysis: Dict[str, Any], documentation: str,
                                output_file: str = "doc_evaluation_report.txt") -> None:
    """Save documentation evaluation to a readable report file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    lines += ["=" * 60, "JAVA DOCUMENTATION QUALITY EVALUATION REPORT",
              "=" * 60, f"Generated: {timestamp}", ""]

    cov = analysis["coverage"]
    lines += ["METHOD COVERAGE", "-" * 40,
              f"Score  : {cov['score']}/1.0",
              f"Covered: {cov['covered']}/{cov['total']} methods"]
    if cov.get("missing_functions"):
        lines.append(f"Missing: {', '.join(cov['missing_functions'])}")
    lines.append("")

    comp = analysis["completeness"]
    lines += ["COMPLETENESS", "-" * 40,
              f"Score : {comp['score']}/1.0  ({comp['passed']}/{comp['total']} checks passed)"]
    for check, passed in comp["checks"].items():
        lines.append(f"  {'✅' if passed else '❌'} {check.replace('_', ' ').title()}")
    lines.append("")

    ln = analysis["length"]
    lines += ["LENGTH & READABILITY", "-" * 40,
              f"Score      : {ln['score']}/1.0",
              f"Word count : {ln['word_count']}  (expected {ln['expected_min']}–{ln['expected_max']})",
              f"Line count : {ln['line_count']}", ""]

    if analysis["has_reference"]:
        lines += ["BLEU & ROUGE (vs reference)", "-" * 40,
                  f"BLEU-4  : {analysis['bleu']}",
                  f"ROUGE-L : {analysis['rouge_l']}", ""]

    conf = analysis["confidence"]
    lines += ["OVERALL CONFIDENCE", "-" * 40,
              f"Score  : {conf['score']}/1.0",
              f"Status : {conf['status']}", "", "Component Scores:"]
    for name, score in conf["components"].items():
        weight  = conf["weights"][name] * 100
        bar     = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        lines.append(f"  {name.replace('_',' ').title():<15} {bar}  {score:.3f}  ({weight:.0f}%)")
    lines.append("")

    lines += ["=" * 60, "VERDICT", "=" * 60]
    score = conf["score"]
    if score >= 0.75:
        lines.append("✅ Documentation ACCEPTED — Good to excellent quality.")
    elif score >= 0.65:
        lines.append("⚠  Documentation CONDITIONALLY ACCEPTED — Review recommended.")
    else:
        lines.append("❌ Documentation REJECTED — Insufficient quality.")
    lines.append("")

    lines += ["=" * 60, "DOCUMENTATION PREVIEW (first 20 lines)", "=" * 60]
    for i, line in enumerate(documentation.splitlines()[:20], 1):
        lines.append(f"  {i:2d}: {line}")
    if len(documentation.splitlines()) > 20:
        lines.append(f"  ... ({len(documentation.splitlines())} lines total)")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Doc evaluation report saved: {output_file}")
    except Exception as e:
        logger.error(f"Could not save doc report: {e}")

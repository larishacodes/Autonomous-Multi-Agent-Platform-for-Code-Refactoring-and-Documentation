# parser/java_parser.py
#
# Block 2: Java Source Code Parser
#
# Primary:  javalang  — pure Python Java 8 parser (accurate AST-based metrics)
# Fallback: regex     — used if javalang is unavailable or file uses Java 9+ syntax
#
# HONEST LIMITATIONS:
#   - javalang supports Java 8 syntax only.
#   - Java 9+ features trigger regex fallback automatically.
#   - Regex param counting is approximate for deeply nested generics.
#   - Responsibility count is heuristic, not static analysis.
#
# ENRICHMENT (NEW — fills all fields required by state.py FunctionUnit + ClassUnit)
#   - params         : list[str]  (param names, not just count)
#   - docstring      : str | None (Javadoc comment immediately preceding declaration)
#   - start_line     : int        (1-based, inclusive)
#   - end_line       : int        (1-based, inclusive, brace-matched)
#   - calls          : list[str]  (method names directly invoked in body)
#   - called_by      : list[str]  (reverse: which methods call this one)
#   - cohesion_score : float      (TCC proxy in [0,1]; 1.0 = fully cohesive)
#   - superclass     : str | None
#   - interfaces     : list[str]
#   - is_abstract    : bool
#   - lcom           : float      (LCOM4-normalised: connected_components / max(1, methods))
#   - instability    : float      (Ce / (Ce + Ca), Robert C. Martin)
#   - file_path      : str        (passed in by pipeline; empty string if not provided)

import re
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

try:
    import javalang
    JAVALANG_AVAILABLE = True
except ImportError:
    JAVALANG_AVAILABLE = False
    logger.warning("[JavaParser] javalang not installed — pip install javalang")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def _strip_strings(code: str) -> str:
    return re.sub(r'"[^"\n]*"', '""', code)


def _extract_method_body(source_code: str, start_line: int) -> list[str]:
    lines = source_code.splitlines()
    brace_count = 0
    body = []
    started = False
    for idx in range(max(0, start_line - 1), len(lines)):
        line = lines[idx]
        body.append(line)
        opens  = line.count("{")
        closes = line.count("}")
        brace_count += opens - closes
        if opens > 0:
            started = True
        if started and brace_count <= 0:
            break
    return body


def _count_nesting_depth(lines: list[str]) -> int:
    depth = max_depth = 0
    for line in lines:
        depth += line.count("{") - line.count("}")
        max_depth = max(max_depth, depth)
    return max(0, max_depth)


def _count_conditionals(lines: list[str]) -> int:
    code = _strip_strings("\n".join(lines))
    count  = len(re.findall(r'\bif\b',              code))
    count += len(re.findall(r'\bcase\b',            code))
    count += len(re.findall(r'\belse\b(?!\s*if\b)', code))
    return count


def _count_loops(lines: list[str]) -> int:
    code = _strip_strings("\n".join(lines))
    return len(re.findall(r'\b(?:for|while|do)\b', code))


def _estimate_responsibilities(lines: list[str]) -> int:
    code = _strip_strings("\n".join(lines)).lower()
    resp = set()
    if re.search(r'system\.out|logger\.|log\.', code):        resp.add("io")
    if re.search(r'\bfile\b|\bstream\b|\bdatabase\b|\bconnection\b|\brepository\b', code): resp.add("persistence")
    if re.search(r'\bfor\b|\bwhile\b|\bdo\b', code):          resp.add("iteration")
    if re.search(r'\bif\b|\bswitch\b', code):                 resp.add("decision")
    if re.search(r'\bnew\s+\w+\s*\(', code):                  resp.add("creation")
    if re.search(r'\bthrow\b|\bcatch\b|\bexception\b', code): resp.add("error")
    return max(1, len(resp))


def _count_params_safe(params_str: str) -> int:
    if not params_str.strip():
        return 0
    cleaned = re.sub(r'<[^>]*>', '', params_str)
    return len([p for p in cleaned.split(",") if p.strip()])


# ─────────────────────────────────────────────────────────────────────────────
# NEW helpers
# ─────────────────────────────────────────────────────────────────────────────

def _end_line(source_code: str, start_line: int) -> int:
    """
    Return the 1-based line number of the closing brace of a method/class
    that opens at start_line.  Falls back to start_line if braces never close.
    """
    body = _extract_method_body(source_code, start_line)
    return start_line + len(body) - 1


def _extract_javadoc(source_code: str, declaration_line: int) -> str | None:
    """
    Walk backward from declaration_line to find a /** ... */ Javadoc block.

    Rules (mirrors javadoc tool behaviour):
    - Only a /** block (not /* or //) counts as Javadoc.
    - The block must end on the line immediately before the declaration
      (ignoring blank lines and @annotation lines between them).
    - Returns the stripped comment text, or None if absent.
    """
    lines = source_code.splitlines()
    idx = declaration_line - 2     # 0-based index of the line above declaration

    # Skip blank lines and annotation lines between comment and declaration
    while idx >= 0 and (
        not lines[idx].strip()
        or lines[idx].strip().startswith("@")
    ):
        idx -= 1

    if idx < 0 or not lines[idx].strip().endswith("*/"):
        return None

    # Collect backward until /**
    end_idx = idx
    while idx >= 0 and "/**" not in lines[idx]:
        idx -= 1

    if idx < 0 or "/**" not in lines[idx]:
        return None

    block = lines[idx : end_idx + 1]
    # Strip comment markers and leading * characters
    cleaned = []
    for line in block:
        stripped = line.strip().lstrip("/*").strip()
        if stripped and not stripped.startswith("@"):
            cleaned.append(stripped)
    return " ".join(cleaned) if cleaned else None


def _extract_calls_from_body(body_lines: list[str]) -> list[str]:
    """
    Extract method call names from a method body using a simple regex.

    Pattern: identifier followed by '(' that is NOT a keyword or constructor.
    Returns deduplicated list preserving first-occurrence order.

    Research: method-method call relations are the primary edge type used
    in LCOM4 computation and call-graph construction.  (Hitz & Montazeri,
    1995; jastg project, 2024).
    """
    code = _strip_strings("\n".join(body_lines))
    _SKIP = {
        "if", "for", "while", "switch", "catch", "return", "new",
        "super", "this", "assert", "synchronized", "throw",
    }
    raw_calls = re.findall(r'\b([a-zA-Z_]\w*)\s*\(', code)
    seen: set[str] = set()
    result: list[str] = []
    for name in raw_calls:
        if name not in _SKIP and name[0].islower() and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _build_called_by(functions: list[dict]) -> None:
    """
    Populate called_by in-place by inverting each function's calls list.

    After this pass, function F's called_by contains the names of all
    functions that list F in their calls.

    Research: reverse call-graph edges are required by PlannerAgent's
    context-aware priority (blast-radius escalation) and by the Supervisor's
    prompt builder (caller context injection).
    """
    name_to_fn = {f["name"]: f for f in functions}
    for fn in functions:
        for callee_name in fn.get("calls", []):
            if callee_name in name_to_fn:
                callee = name_to_fn[callee_name]
                if fn["name"] not in callee["called_by"]:
                    callee["called_by"].append(fn["name"])


def _cohesion_score(method_names: list[str], fields: list[str], body_map: dict[str, list[str]]) -> float:
    """
    Compute a Tight Class Cohesion (TCC) proxy in [0.0, 1.0].

    TCC = pairs of methods sharing ≥1 field access / total method pairs.

    Research: TCC is a normalised cohesion metric closely related to LCOM4.
    For static analysis without bytecode, method-attribute access is the
    most reliable signal (Hitz & Montazeri 1995; Chidamber & Kemerer 1994).
    TCC = 1.0 means all method pairs share a field → maximally cohesive.
    TCC = 0.0 means no method pair shares any field → class should be split.

    Falls back to 1.0 if there are fewer than 2 methods or no fields
    (avoids division by zero and unfair penalisation of tiny classes).
    """
    if len(method_names) < 2 or not fields:
        return 1.0

    # Which fields does each method access?
    method_fields: dict[str, set[str]] = {}
    for method in method_names:
        body = body_map.get(method, [])
        body_text = _strip_strings("\n".join(body))
        accessed = {f for f in fields if re.search(r'\b' + re.escape(f) + r'\b', body_text)}
        method_fields[method] = accessed

    shared_pairs = 0
    total_pairs  = 0
    for i, m1 in enumerate(method_names):
        for m2 in method_names[i + 1:]:
            total_pairs += 1
            if method_fields[m1] & method_fields[m2]:
                shared_pairs += 1

    return round(shared_pairs / total_pairs, 4) if total_pairs > 0 else 1.0


def _lcom4_normalised(
    method_names: list[str],
    fields: list[str],
    body_map: dict[str, list[str]],
    calls_map: dict[str, list[str]],
) -> float:
    """
    Compute a normalised LCOM4 value in [0.0, 1.0].

    LCOM4 = number of connected components in the method graph, where two
    methods are connected if:
      (a) they both access the same field, OR
      (b) one calls the other.

    Normalised LCOM4 = (components - 1) / max(1, methods - 1)
    → 0.0 means fully cohesive (1 component, ideal)
    → 1.0 means fully fragmented (every method is isolated)

    Research: LCOM4=1 is cohesive; LCOM4≥2 means class should
    be split into that many smaller classes.  Normalising makes it
    comparable across classes of different sizes.
    """
    if len(method_names) < 2:
        return 0.0

    # Build adjacency list
    adj: dict[str, set[str]] = defaultdict(set)

    # Field-sharing edges
    method_fields: dict[str, set[str]] = {}
    for method in method_names:
        body = body_map.get(method, [])
        body_text = _strip_strings("\n".join(body))
        method_fields[method] = {
            f for f in fields
            if re.search(r'\b' + re.escape(f) + r'\b', body_text)
        }
    for i, m1 in enumerate(method_names):
        for m2 in method_names[i + 1:]:
            if method_fields[m1] & method_fields[m2]:
                adj[m1].add(m2)
                adj[m2].add(m1)

    # Method-call edges
    for method in method_names:
        for callee in calls_map.get(method, []):
            if callee in set(method_names):
                adj[method].add(callee)
                adj[callee].add(method)

    # Count connected components via BFS
    visited: set[str] = set()
    components = 0
    for start in method_names:
        if start not in visited:
            components += 1
            queue = [start]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                queue.extend(adj[node] - visited)

    n = len(method_names)
    return round((components - 1) / max(1, n - 1), 4)


def _instability(class_name: str, all_classes: list[dict]) -> float:
    """
    Compute Martin's instability metric: I = Ce / (Ce + Ca)

    Ce (efferent coupling) = number of other class names this class references
                             in its methods' call lists and field types.
    Ca (afferent coupling) = number of other classes that reference this class.

    Range [0.0, 1.0]:
      0.0 = maximally stable (nobody depends on others; everyone depends on it)
      1.0 = maximally unstable (depends on many; nobody depends on it)

    Research: Instability I = Ce / (Ce + Ca) is Robert C. Martin's
    package metric (Agile Software Development, 2002); the same formula applies
    at the class level.  Planner uses this to decide whether to defer
    refactoring of stable classes.

    Implementation: we use the class name set as a proxy for the type system.
    If a method body mentions another class name (capitalised identifier),
    that counts as an efferent dependency.  Afferent coupling is the inverse.
    """
    other_names = {c["name"] for c in all_classes if c["name"] != class_name}

    # Collect all body text for this class
    target = next((c for c in all_classes if c["name"] == class_name), None)
    if not target:
        return 0.0

    all_body_text = " ".join(target.get("_body_text", ""))

    # Ce: capitalised names from other classes mentioned in this class's bodies
    ce = sum(1 for n in other_names if re.search(r'\b' + re.escape(n) + r'\b', all_body_text))

    # Ca: other classes that mention this class name in their bodies
    ca = sum(
        1 for c in all_classes
        if c["name"] != class_name
        and re.search(r'\b' + re.escape(class_name) + r'\b', " ".join(c.get("_body_text", "")))
    )

    total = ce + ca
    return round(ce / total, 4) if total > 0 else 0.0


def _extract_fields_javalang(class_node) -> list[str]:
    """Extract declared field names from a javalang ClassDeclaration node."""
    fields = []
    for _, member in class_node:
        if isinstance(member, javalang.tree.FieldDeclaration):
            for declarator in (member.declarators or []):
                if hasattr(declarator, "name"):
                    fields.append(declarator.name)
    return fields


# ─────────────────────────────────────────────────────────────────────────────
# JavaParser
# ─────────────────────────────────────────────────────────────────────────────

class JavaParser:
    """
    Parses a Java source file and returns a structured dict consumed by the
    rest of the pipeline.

    Output keys
    -----------
    parse_success  bool
    parser_used    "javalang" | "regex"
    language       "java"
    total_loc      int
    file_path      str   (echo of the argument passed to parse())
    classes        list[dict]
    functions      list[dict]

    Per-function dict keys  (all fields required by state.FunctionUnit)
    ----------------------
    name, params, docstring, file_path, start_line, end_line,
    calls, called_by, complexity, loc, cohesion_score,
    param_count, return_type, modifiers,
    nesting_depth, conditional_count, loop_count,
    responsibility_count, parser_used

    Per-class dict keys  (all fields required by state.ClassUnit)
    -------------------
    name, docstring, file_path, superclass, interfaces, is_abstract,
    method_count, loc, lcom, instability, methods, parser_used
    """

    def parse(self, source_code: str, file_path: str = "") -> dict:
        if JAVALANG_AVAILABLE:
            result = self._parse_with_javalang(source_code, file_path)
            if result["parse_success"]:
                return result
            logger.warning("[JavaParser] javalang failed — falling back to regex.")
        return self._parse_with_regex(source_code, file_path)

    # ── javalang (primary) ───────────────────────────────────────────────────

    def _parse_with_javalang(self, source_code: str, file_path: str) -> dict:
        try:
            tree = javalang.parse.parse(source_code)
        except Exception as e:
            return {
                "parse_success": False, "error": str(e),
                "functions": [], "classes": [], "total_loc": 0,
                "language": "java", "parser_used": "javalang", "file_path": file_path,
            }

        lines     = source_code.splitlines()
        functions: list[dict] = []
        classes:   list[dict] = []

        # ── First pass: collect classes with body text for coupling metrics ──
        class_nodes = []
        for _, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration):
                lineno = node.position.line if node.position else 1
                body   = _extract_method_body(source_code, lineno)
                class_nodes.append((node, lineno, body))

        # Temporary class records needed for instability computation
        temp_classes: list[dict] = []
        for node, lineno, body in class_nodes:
            temp_classes.append({
                "name":        node.name,
                "_body_text":  body,
            })

        # ── Second pass: build full class + function dicts ────────────────────
        for node, lineno, body in class_nodes:
            end      = _end_line(source_code, lineno)
            doc      = _extract_javadoc(source_code, lineno)
            fields   = _extract_fields_javalang(node)
            modifiers = list(node.modifiers) if node.modifiers else []

            # Collect method-level data for cohesion metrics
            method_names:    list[str]             = []
            body_map:        dict[str, list[str]]  = {}
            calls_map:       dict[str, list[str]]  = {}
            method_dicts:    list[dict]             = []

            for _, member in node:
                if not isinstance(member, javalang.tree.MethodDeclaration):
                    continue

                m_lineno  = member.position.line if member.position else lineno
                m_body    = _extract_method_body(source_code, m_lineno)
                m_end     = _end_line(source_code, m_lineno)
                m_doc     = _extract_javadoc(source_code, m_lineno)
                m_calls   = _extract_calls_from_body(m_body)
                m_params  = [p.name for p in (member.parameters or [])]

                if member.return_type is None:
                    return_type = "void"
                elif hasattr(member.return_type, "name"):
                    return_type = member.return_type.name or "void"
                else:
                    rt_str = str(member.return_type)
                    m = re.search(r"name='(\w+)'", rt_str)
                    return_type = m.group(1) if m else rt_str

                method_names.append(member.name)
                body_map[member.name]  = m_body
                calls_map[member.name] = m_calls

                method_dicts.append({
                    "name":                 member.name,
                    "params":               m_params,
                    "docstring":            m_doc,
                    "file_path":            file_path,
                    "start_line":           m_lineno,
                    "end_line":             m_end,
                    "calls":                m_calls,
                    "called_by":            [],          # populated in third pass
                    "complexity":           _count_conditionals(m_body) + _count_loops(m_body) + 1,
                    "loc":                  len(m_body),
                    "cohesion_score":       1.0,         # set per-class below
                    "param_count":          len(m_params),
                    "return_type":          return_type,
                    "modifiers":            sorted(member.modifiers) if member.modifiers else [],
                    "nesting_depth":        _count_nesting_depth(m_body),
                    "conditional_count":    _count_conditionals(m_body),
                    "loop_count":           _count_loops(m_body),
                    "responsibility_count": _estimate_responsibilities(m_body),
                    "parser_used":          "javalang",
                })

            # Compute class-level cohesion score (TCC proxy) and distribute
            tcc  = _cohesion_score(method_names, fields, body_map)
            lcom = _lcom4_normalised(method_names, fields, body_map, calls_map)
            for md in method_dicts:
                md["cohesion_score"] = tcc

            # Compute instability
            inst = _instability(node.name, temp_classes)

            # Resolve superclass name
            superclass = None
            if node.extends:
                superclass = node.extends.name if hasattr(node.extends, "name") else str(node.extends)

            interfaces = []
            if node.implements:
                for iface in node.implements:
                    interfaces.append(iface.name if hasattr(iface, "name") else str(iface))

            classes.append({
                "name":         node.name,
                "docstring":    doc,
                "file_path":    file_path,
                "superclass":   superclass,
                "interfaces":   interfaces,
                "is_abstract":  "abstract" in modifiers,
                "method_count": len(method_dicts),
                "loc":          len(body),
                "lcom":         lcom,
                "instability":  inst,
                "methods":      method_dicts,
                "parser_used":  "javalang",
            })
            functions.extend(method_dicts)

        # ── Third pass: build called_by (reverse call graph) ─────────────────
        _build_called_by(functions)

        return {
            "parse_success": True,
            "parser_used":   "javalang",
            "language":      "java",
            "total_loc":     len(lines),
            "file_path":     file_path,
            "classes":       classes,
            "functions":     functions,
        }

    # ── regex fallback ───────────────────────────────────────────────────────

    def _parse_with_regex(self, source_code: str, file_path: str) -> dict:
        """Regex fallback for Java 9+ or when javalang is not installed."""
        lines     = source_code.splitlines()
        functions: list[dict] = []
        classes:   list[dict] = []

        class_pattern = re.compile(
            r'(?:(?:public|private|protected|abstract|final)\s+)*class\s+(\w+)'
            r'(?:\s+extends\s+(\w+))?'
            r'(?:\s+implements\s+([\w,\s]+))?'
        )
        method_pattern = re.compile(
            r'(?:(?:public|private|protected|static|final|synchronized|abstract|native)\s+)*'
            r'([\w<>\[\]]+)\s+(\w+)\s*\(([^)]*)\)'
            r'(?:\s*throws\s+[\w,\s]+)?\s*\{'
        )

        WINDOW  = 8
        windows = [
            " ".join(l.strip() for l in lines[i : i + WINDOW])
            for i in range(len(lines))
        ]

        seen: set[str] = set()
        # Temporary class body accumulator for coupling
        temp_classes: list[dict] = []

        for i, line in enumerate(lines, 1):

            # ── Class detection ───────────────────────────────────────────
            cm = class_pattern.search(line)
            if cm and "class" in line and "=" not in line:
                name   = cm.group(1)
                sc     = cm.group(2)
                ifaces_raw = cm.group(3)
                ifaces = [x.strip() for x in ifaces_raw.split(",")] if ifaces_raw else []

                body       = _extract_method_body(source_code, i)
                end        = i + len(body) - 1
                doc        = _extract_javadoc(source_code, i)
                is_abstract = "abstract" in line

                method_count = len(re.findall(
                    r'(?:public|private|protected)\s+\w[\w<>\[\]]*\s+\w+\s*\(',
                    "\n".join(body),
                ))

                temp_classes.append({"name": name, "_body_text": body})

                classes.append({
                    "name":         name,
                    "docstring":    doc,
                    "file_path":    file_path,
                    "superclass":   sc,
                    "interfaces":   ifaces,
                    "is_abstract":  is_abstract,
                    "method_count": method_count,
                    "loc":          len(body),
                    "lcom":         0.0,        # requires field analysis; not available in regex
                    "instability":  0.0,        # computed post-loop below
                    "methods":      [],
                    "parser_used":  "regex",
                })
                continue

            # ── Method detection ──────────────────────────────────────────
            window = windows[i - 1]
            mm     = method_pattern.search(window)
            if not mm:
                continue

            return_type = mm.group(1)
            method_name = mm.group(2)
            params_str  = mm.group(3)

            anchor = " ".join(l.strip() for l in lines[i - 1 : min(i + 1, len(lines))])
            if method_name not in anchor:
                continue

            _SKIP = {"if", "for", "while", "switch", "catch", "else",
                     "return", "new", "class", "interface", "enum"}
            if method_name.lower() in _SKIP:
                continue
            if return_type[0].isupper() and return_type == method_name:
                continue
            if method_name in seen:
                continue
            seen.add(method_name)

            body  = _extract_method_body(source_code, i)
            end   = i + len(body) - 1
            doc   = _extract_javadoc(source_code, i)
            calls = _extract_calls_from_body(body)

            # Parse param names (best-effort from regex)
            params: list[str] = []
            cleaned = re.sub(r'<[^>]*>', '', params_str)
            for part in cleaned.split(","):
                tokens = part.strip().split()
                if len(tokens) >= 2:
                    params.append(tokens[-1])

            functions.append({
                "name":                 method_name,
                "params":               params,
                "docstring":            doc,
                "file_path":            file_path,
                "start_line":           i,
                "end_line":             end,
                "calls":                calls,
                "called_by":            [],
                "complexity":           _count_conditionals(body) + _count_loops(body) + 1,
                "loc":                  len(body),
                "cohesion_score":       1.0,    # cannot compute without field list in regex
                "param_count":          _count_params_safe(params_str),
                "return_type":          return_type,
                "modifiers":            [],
                "nesting_depth":        _count_nesting_depth(body),
                "conditional_count":    _count_conditionals(body),
                "loop_count":           _count_loops(body),
                "responsibility_count": _estimate_responsibilities(body),
                "parser_used":          "regex",
            })

        # Third pass: reverse call graph
        _build_called_by(functions)

        # Instability for regex-parsed classes (best-effort)
        for cls in classes:
            cls["instability"] = _instability(cls["name"], temp_classes)

        return {
            "parse_success": True,
            "parser_used":   "regex",
            "language":      "java",
            "total_loc":     len(lines),
            "file_path":     file_path,
            "classes":       classes,
            "functions":     functions,
        }
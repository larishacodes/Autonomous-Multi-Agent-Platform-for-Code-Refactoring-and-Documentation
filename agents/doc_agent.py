# agents/doc_agent.py
#
# Block 5b: Doc Agent
# Fine-tuned CodeT5+ 770M with LoRA adapter for Java documentation generation.
#
# Receives the documentation prompt from the Prompting Engine
# and returns generated documentation in Markdown format.

import json
import shutil
import tempfile
import logging
import re
from pathlib import Path
from typing import Optional

import torch
import safetensors.torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import PeftModel

logger = logging.getLogger(__name__)

BASE_MODEL_NAME = "Salesforce/codet5p-770m"


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer loader (added_tokens bug patch)
# ═══════════════════════════════════════════════════════════════════════════════
def _load_tokenizer(source_dir: Path) -> AutoTokenizer:
    """Load tokenizer with added_tokens format patch for codet5p-770m."""
    tmp = Path(tempfile.mkdtemp(prefix="tok_doc_"))
    try:
        for pattern in ["tokenizer*", "special_tokens_map.json", "vocab.json", "merges.txt"]:
            for f in source_dir.glob(pattern):
                shutil.copy(f, tmp / f.name)

        tok_json = tmp / "tokenizer.json"
        if tok_json.exists():
            with open(tok_json, encoding="utf-8") as fh:
                data = json.load(fh)
            fixed = []
            for entry in data.get("added_tokens", []):
                if isinstance(entry, str):
                    fixed.append({"id": 0, "content": entry, "single_word": False,
                                  "lstrip": False, "rstrip": False,
                                  "normalized": False, "special": False})
                else:
                    fixed.append(entry)
            data["added_tokens"] = fixed
            with open(tok_json, "w", encoding="utf-8") as fh:
                json.dump(data, fh)

        return AutoTokenizer.from_pretrained(str(tmp), use_fast=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Doc Agent class
# ═══════════════════════════════════════════════════════════════════════════════
class DocAgent:
    """
    Doc Agent — fine-tuned CodeT5+ 770M with LoRA adapter for docstring generation.
    Trained on CodeSearchNet Java dataset.
    Receives prompt from the Prompting Engine and returns documentation.
    Falls back to template-based documentation if model is unavailable.
    """

    def __init__(self, adapter_path: str, max_input_length: int = 512,
                 max_output_length: int = 256):
        self.adapter_path = Path(adapter_path)
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False

    def load(self) -> bool:
        """Load base model + LoRA adapter. Returns True if successful."""
        if self._loaded:
            return True

        if not self.adapter_path.exists():
            logger.error(f"Doc adapter not found: {self.adapter_path}")
            logger.error("Copy your trained adapter files to models/doc_agent_final/")
            return False

        try:
            logger.debug(f"[DocAgent] Loading base model {BASE_MODEL_NAME} ...")
            logging.getLogger("transformers.safetensors_conversion").setLevel(logging.ERROR)

            self.tokenizer = _load_tokenizer(self.adapter_path)
            base_model = T5ForConditionalGeneration.from_pretrained(
                BASE_MODEL_NAME, use_safetensors=False
            )
            base_model = base_model.to(self.device)

            logger.debug(f"[DocAgent] Loading LoRA adapter from {self.adapter_path} ...")
            self.model = PeftModel.from_pretrained(base_model, str(self.adapter_path),
                                                   is_trainable=False)

            # Key-remapping fix
            weights_file = self.adapter_path / "adapter_model.safetensors"
            if weights_file.exists():
                state_dict = safetensors.torch.load_file(str(weights_file))
                fixed = {}
                for k, v in state_dict.items():
                    nk = k.replace("base_model.model.base_model.model.", "base_model.model.")
                    nk = nk.replace(".lora_A.weight", ".lora_A.default.weight")
                    nk = nk.replace(".lora_B.weight", ".lora_B.default.weight")
                    fixed[nk] = v
                result  = self.model.load_state_dict(fixed, strict=False)
                matched = len(fixed) - len(result.unexpected_keys)
                logger.info(f"[DocAgent] ✅ LoRA loaded ({matched}/{len(fixed)} keys)")

            self.model.eval()
            self._loaded = True
            logger.debug("[DocAgent] ✅ Ready.")
            return True

        except Exception as e:
            logger.error(f"[DocAgent] Load failed: {e}")
            return False

    def run(self, prompt: str, parsed_code: dict) -> dict:
        """
        Run the doc agent.

        Args:
            prompt:       The full documentation prompt from the Prompting Engine
                          (includes function list and code)
            parsed_code:  The parsed AST structure (used for fallback)

        Returns:
            dict with keys:
              - documentation: str  (Markdown-formatted documentation)
              - used_model: bool
        """
        if not self._loaded:
            loaded = self.load()
            if not loaded:
                logger.warning("[DocAgent] Model unavailable — using template fallback.")
                return self._fallback(parsed_code)

        try:
            # The Doc Agent generates per-function docstrings
            # Run once per function for best quality
            functions = parsed_code.get("functions", [])
            if not functions:
                return self._fallback(parsed_code)

            doc_sections = []

            for func in functions:
                func_prompt = self._build_function_prompt(func, prompt)
                inputs = self.tokenizer(
                    func_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_input_length,
                    padding=True,
                    return_attention_mask=True,
                ).to(self.device)

                logger.debug(f"[DocAgent] Generating docs for: {func['name']} ...")
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=self.max_output_length,
                        num_beams=2,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        length_penalty=1.0,
                    )

                raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                docstring = self._clean_docstring(raw)
                doc_sections.append((func["name"], docstring))

            # Format as Markdown documentation
            documentation = self._format_as_markdown(doc_sections, parsed_code)
            logger.debug("[DocAgent] ✅ Documentation generated.")
            return {"documentation": documentation, "used_model": True}

        except Exception as e:
            logger.error(f"[DocAgent] Inference error: {e}")
            return self._fallback(parsed_code)

    # ─── Helpers ────────────────────────────────────────────────────────────

    def _build_function_prompt(self, func: dict, full_prompt: str) -> str:
        """
        Build a per-function prompt in CodeSearchNet Java style.
        The doc agent is trained on CodeSearchNet Java pairs:
        Java method code → summary description.
        """
        func_name   = func.get("name", "unknown")
        return_type = func.get("return_type", "void")
        params      = func.get("param_count", 0)
        # Matches CodeSearchNet Java training format
        return (f"Generate documentation for Java method: "
                f"{return_type} {func_name} with {params} parameters")

    def _clean_docstring(self, raw: str) -> str:
        """Clean up model output to extract the docstring text."""
        # Remove common artifacts
        for marker in ("Generate documentation for Java method:", "Docstring:", "```java", "```python", "```"):
            if marker in raw:
                raw = raw.split(marker, 1)[-1]
        raw = re.sub(r"```\w*", "", raw).strip()

        # Return cleaned text
        return raw.strip() if raw.strip() else "No description available."

    def _format_as_markdown(self, doc_sections: list, parsed_code: dict) -> str:
        """Format the generated docstrings as a Markdown document."""
        lines = []
        lines.append("# Code Documentation")
        lines.append("")
        lines.append("## Overview")
        lines.append("")
        lines.append(f"This module contains {len(parsed_code.get('functions', []))} function(s) "
                     f"and {len(parsed_code.get('classes', []))} class(es).")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Functions")
        lines.append("")

        functions = parsed_code.get("functions", [])
        for func, (func_name, docstring) in zip(functions, doc_sections):
            param_count = func.get("param_count", 0)
            loc         = func.get("loc", 0)
            lineno      = func.get("lineno", "?")

            lines.append(f"### `{func_name}`")
            lines.append("")
            lines.append(f"**Description:** {docstring}")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Line number | {lineno} |")
            lines.append(f"| Lines of code | {loc} |")
            lines.append(f"| Parameters | {param_count} |")
            lines.append("")

        return "\n".join(lines)

    def _fallback(self, parsed_code: dict) -> dict:
        """
        Template-based documentation fallback.
        Generates structured docs from the parsed AST — no model required.
        """
        lines = []
        lines.append("# Code Documentation")
        lines.append("")
        lines.append("## Overview")
        lines.append("")
        functions = parsed_code.get("functions", [])
        classes   = parsed_code.get("classes", [])
        lines.append(f"This module contains {len(functions)} function(s) "
                     f"and {len(classes)} class(es).")
        lines.append("")
        lines.append("> *Note: Generated using template fallback. "
                     "Install the Doc Agent model for AI-generated descriptions.*")
        lines.append("")
        lines.append("---")
        lines.append("")

        if functions:
            lines.append("## Functions")
            lines.append("")
            for func in functions:
                name   = func.get("name", "unknown")
                params = func.get("param_count", 0)
                loc    = func.get("loc", 0)
                lineno = func.get("lineno", "?")
                depth  = func.get("nesting_depth", 0)
                resp   = func.get("responsibility_count", 1)

                lines.append(f"### `{name}`")
                lines.append("")
                lines.append(f"**Description:** Function `{name}` — auto-extracted from source.")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                lines.append(f"| Line number | {lineno} |")
                lines.append(f"| Lines of code | {loc} |")
                lines.append(f"| Parameters | {params} |")
                lines.append(f"| Nesting depth | {depth} |")
                lines.append(f"| Responsibility count | {resp} |")
                lines.append("")

        if classes:
            lines.append("## Classes")
            lines.append("")
            for cls in classes:
                name    = cls.get("name", "unknown")
                methods = cls.get("method_count", 0)
                loc     = cls.get("loc", 0)
                lineno  = cls.get("lineno", "?")

                lines.append(f"### `{name}`")
                lines.append("")
                lines.append(f"**Description:** Class `{name}` — auto-extracted from source.")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                lines.append(f"| Line number | {lineno} |")
                lines.append(f"| Lines of code | {loc} |")
                lines.append(f"| Methods | {methods} |")
                lines.append("")

        return {"documentation": "\n".join(lines), "used_model": False}

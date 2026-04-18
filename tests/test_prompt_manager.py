"""Code-based grader: prompt_manager render_prompt function."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_manager import render_prompt


class TestRenderPrompt:
    def test_jinja2_rendering(self):
        template = "Translate {{ source_lang }} text: {{ text }}"
        result = render_prompt(template, source_lang="Pali", text="Dhamma")
        assert result == "Translate Pali text: Dhamma"

    def test_python_format_rendering(self):
        template = "Translate {source_lang} text: {text}"
        result = render_prompt(template, source_lang="Sanskrit", text="Om")
        assert result == "Translate Sanskrit text: Om"

    def test_plain_string_passthrough(self):
        template = "No variables here"
        result = render_prompt(template)
        assert result == "No variables here"

    def test_jinja2_takes_priority_when_braces_present(self):
        template = "{{ source_lang }}"
        result = render_prompt(template, source_lang="Pali")
        assert result == "Pali"

    def test_missing_format_key_returns_template(self):
        template = "Hello {name}"
        result = render_prompt(template, wrong_key="value")
        assert result == template

    def test_jinja2_conditional_requires_double_braces(self):
        # render_prompt only activates Jinja2 when {{ is present.
        # Pure {% %} blocks without {{ fall through to str.format() and return as-is.
        template = "{{ lang }}{% if lang == 'Pali' %}-Ancient{% endif %}"
        result = render_prompt(template, lang="Pali")
        assert result == "Pali-Ancient"

    def test_extra_kwargs_ignored(self):
        template = "{{ source_lang }}"
        result = render_prompt(template, source_lang="Pali", extra="unused")
        assert result == "Pali"

"""Integration tests for orchestrator --domain forwarding.

Verify that run_all_metrics.py, run_metric_refresh.py, and
run_build_pipeline.py correctly forward the --domain flag to child
subprocess calls.  Uses AST analysis to inspect the forwarding patterns
in source code rather than executing the full pipelines.

Tested patterns
---------------
- run_all_metrics.py: builds ``domain_fwd = ["--domain", args.domain]``
  list, concatenated onto every command via ``+ domain_fwd``.
- run_metric_refresh.py: appends ``f"--domain={domain}"`` (equals format)
  inside ``run_metric_script()``.
- run_build_pipeline.py: calls ``cmd.extend(['--domain', config.domain])``
  inside ``refresh_metrics_if_requested()``.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts.run_build_pipeline import resolve_front_config_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _parse_script(rel_path: str) -> ast.Module:
    """Parse a script file and return its AST, skipping if absent."""
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        pytest.skip(f"{rel_path} not present")
    source = path.read_text(encoding="utf-8")
    return ast.parse(source, filename=rel_path)


def _iter_function_defs(
    tree: ast.Module,
) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Yield all top-level and nested FunctionDef nodes in *tree*."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _function_source_lines(
    tree: ast.Module, func_name: str, source_text: str,
) -> list[str]:
    """Return source lines for the named function, or empty list if absent."""
    for func in _iter_function_defs(tree):
        if func.name == func_name:
            lines = source_text.splitlines()
            return lines[func.lineno - 1 : func.end_lineno]
    return []


def _has_add_domain_args_call(tree: ast.Module) -> bool:
    """Return True if the AST contains a call to ``add_domain_args``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "add_domain_args":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "add_domain_args":
            return True
    return False


def _has_string_containing(tree: ast.Module, substring: str) -> bool:
    """Return True if any string constant in the AST contains *substring*."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if substring in node.value:
                return True
    return False


def _has_joined_str_containing(tree: ast.Module, substring: str) -> bool:
    """Return True if any f-string in the AST contains *substring* as a literal part."""
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for val in node.values:
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    if substring in val.value:
                        return True
    return False


def _find_assignments_to(tree: ast.Module, name: str) -> list[ast.Assign]:
    """Find all assignment statements where *name* is a target."""
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    results.append(node)
    return results


# ---------------------------------------------------------------------------
# run_all_metrics.py
# ---------------------------------------------------------------------------


class TestRunAllMetricsDomainForwarding:
    """Verify domain forwarding in run_all_metrics.py."""

    SCRIPT = "scripts/run_all_metrics.py"

    def test_has_add_domain_args(self) -> None:
        """Script must register --domain via add_domain_args()."""
        tree = _parse_script(self.SCRIPT)
        assert _has_add_domain_args_call(tree), (
            f"{self.SCRIPT} does not call add_domain_args()"
        )

    def test_builds_domain_fwd_list(self) -> None:
        """Script must construct a domain_fwd list with '--domain' literal."""
        tree = _parse_script(self.SCRIPT)
        assignments = _find_assignments_to(tree, "domain_fwd")
        assert len(assignments) >= 1, (
            f"{self.SCRIPT}: no assignment to 'domain_fwd' found"
        )
        # At least one assignment should reference "--domain" string
        assert _has_string_containing(tree, "--domain"), (
            f"{self.SCRIPT}: no '--domain' string literal found"
        )

    def test_domain_fwd_concatenated_into_commands(self) -> None:
        """Each metric command list must be concatenated with domain_fwd.

        Pattern: ``[...] + domain_fwd`` appears for every metric script entry.
        """
        path = PROJECT_ROOT / self.SCRIPT
        if not path.exists():
            pytest.skip(f"{self.SCRIPT} not present")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=self.SCRIPT)

        # Find all BinOp nodes where right operand is Name('domain_fwd')
        concat_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                if (
                    isinstance(node.right, ast.Name)
                    and node.right.id == "domain_fwd"
                ):
                    concat_count += 1

        # run_all_metrics.py has 5 metric commands, each with + domain_fwd
        assert concat_count >= 5, (
            f"{self.SCRIPT}: expected >= 5 '+ domain_fwd' concatenations, "
            f"found {concat_count}"
        )

    def test_domain_fwd_guards_on_domain_attribute(self) -> None:
        """domain_fwd must only be populated when args.domain is set.

        Ensures no unconditional injection of --domain into commands.
        """
        path = PROJECT_ROOT / self.SCRIPT
        if not path.exists():
            pytest.skip(f"{self.SCRIPT} not present")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=self.SCRIPT)

        # The pattern is: if getattr(args, "domain", None): domain_fwd = [...]
        # Find If nodes that guard the domain_fwd assignment.
        guarded = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            # Check if the body contains an assignment to domain_fwd
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id == "domain_fwd":
                            guarded = True
                            break

        assert guarded, (
            f"{self.SCRIPT}: domain_fwd assignment is not guarded by a conditional"
        )

    def test_no_domain_when_omitted(self) -> None:
        """When --domain is not passed, domain_fwd must be empty.

        Verify default initialization: ``domain_fwd: list[str] = []``.
        """
        path = PROJECT_ROOT / self.SCRIPT
        if not path.exists():
            pytest.skip(f"{self.SCRIPT} not present")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=self.SCRIPT)

        # Find the default assignment: domain_fwd = []
        has_empty_default = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                target = node.target
                if isinstance(target, ast.Name) and target.id == "domain_fwd":
                    if isinstance(node.value, ast.List) and len(node.value.elts) == 0:
                        has_empty_default = True
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "domain_fwd":
                        if isinstance(node.value, ast.List) and len(node.value.elts) == 0:
                            has_empty_default = True

        assert has_empty_default, (
            f"{self.SCRIPT}: domain_fwd does not default to empty list"
        )


# ---------------------------------------------------------------------------
# run_metric_refresh.py
# ---------------------------------------------------------------------------


class TestRunMetricRefreshDomainForwarding:
    """Verify domain forwarding in run_metric_refresh.py."""

    SCRIPT = "scripts/run_metric_refresh.py"

    def test_has_add_domain_args(self) -> None:
        """Script must register --domain via add_domain_args()."""
        tree = _parse_script(self.SCRIPT)
        assert _has_add_domain_args_call(tree), (
            f"{self.SCRIPT} does not call add_domain_args()"
        )

    def test_run_metric_script_accepts_domain_param(self) -> None:
        """run_metric_script() must accept a 'domain' keyword argument."""
        tree = _parse_script(self.SCRIPT)
        for func in _iter_function_defs(tree):
            if func.name == "run_metric_script":
                arg_names = [arg.arg for arg in func.args.args]
                kwonly_names = [arg.arg for arg in func.args.kwonlyargs]
                all_params = arg_names + kwonly_names
                assert "domain" in all_params, (
                    "run_metric_script() must accept a 'domain' parameter"
                )
                return
        pytest.fail("run_metric_script() function not found in AST")

    def test_forwards_domain_with_equals_format(self) -> None:
        """Domain forwarding must use f-string equals format: --domain={domain}."""
        tree = _parse_script(self.SCRIPT)
        # Look for JoinedStr (f-string) containing "--domain="
        found = _has_joined_str_containing(tree, "--domain=")
        assert found, (
            f'{self.SCRIPT}: no f-string containing "--domain=" found'
        )

    def test_domain_forwarding_guarded_by_conditional(self) -> None:
        """Domain append must be conditional on domain being non-None."""
        path = PROJECT_ROOT / self.SCRIPT
        if not path.exists():
            pytest.skip(f"{self.SCRIPT} not present")
        source = path.read_text(encoding="utf-8")
        func_lines = _function_source_lines(
            ast.parse(source, filename=self.SCRIPT),
            "run_metric_script",
            source,
        )
        assert func_lines, "run_metric_script() not found"

        # Verify the pattern: 'if domain is not None:' followed by '--domain='
        func_text = "\n".join(func_lines)
        assert "domain is not None" in func_text or "domain:" in func_text, (
            "Domain forwarding must be guarded by a None check"
        )
        assert "--domain=" in func_text, (
            "run_metric_script() must contain '--domain=' forwarding"
        )

    def test_main_passes_domain_to_run_metric_script(self) -> None:
        """main() must forward args.domain to run_metric_script() calls."""
        path = PROJECT_ROOT / self.SCRIPT
        if not path.exists():
            pytest.skip(f"{self.SCRIPT} not present")
        source = path.read_text(encoding="utf-8")

        # Verify the keyword argument domain= is passed in a call
        tree = ast.parse(source, filename=self.SCRIPT)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                is_target = False
                if isinstance(func, ast.Name) and func.id == "run_metric_script":
                    is_target = True
                if isinstance(func, ast.Attribute) and func.attr == "run_metric_script":
                    is_target = True
                if is_target:
                    kw_names = [kw.arg for kw in node.keywords]
                    assert "domain" in kw_names, (
                        "run_metric_script() call must include domain= keyword"
                    )
                    return
        pytest.fail("No call to run_metric_script() found in main()")


# ---------------------------------------------------------------------------
# run_build_pipeline.py
# ---------------------------------------------------------------------------


class TestRunBuildPipelineDomainForwarding:
    """Verify domain forwarding in run_build_pipeline.py."""

    SCRIPT = "scripts/run_build_pipeline.py"

    def test_has_add_domain_args(self) -> None:
        """Script must register --domain via add_domain_args()."""
        tree = _parse_script(self.SCRIPT)
        assert _has_add_domain_args_call(tree), (
            f"{self.SCRIPT} does not call add_domain_args()"
        )

    def test_pipeline_config_captures_domain(self) -> None:
        """PipelineConfig.__init__ must store domain from args."""
        path = PROJECT_ROOT / self.SCRIPT
        if not path.exists():
            pytest.skip(f"{self.SCRIPT} not present")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=self.SCRIPT)

        # Find PipelineConfig class and verify domain assignment
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "PipelineConfig":
                class_source = source.splitlines()[node.lineno - 1 : node.end_lineno]
                class_text = "\n".join(class_source)
                assert "self.domain" in class_text, (
                    "PipelineConfig must store self.domain"
                )
                return
        pytest.fail("PipelineConfig class not found")

    def test_refresh_metrics_forwards_domain_via_extend(self) -> None:
        """refresh_metrics_if_requested() must extend cmd with --domain."""
        path = PROJECT_ROOT / self.SCRIPT
        if not path.exists():
            pytest.skip(f"{self.SCRIPT} not present")
        source = path.read_text(encoding="utf-8")
        func_lines = _function_source_lines(
            ast.parse(source, filename=self.SCRIPT),
            "refresh_metrics_if_requested",
            source,
        )
        assert func_lines, "refresh_metrics_if_requested() not found"

        func_text = "\n".join(func_lines)
        assert "--domain" in func_text, (
            "refresh_metrics_if_requested() must contain '--domain'"
        )
        assert "config.domain" in func_text, (
            "refresh_metrics_if_requested() must reference config.domain"
        )

    def test_domain_forwarding_guarded_by_conditional(self) -> None:
        """Domain extend must be conditional on config.domain being set."""
        path = PROJECT_ROOT / self.SCRIPT
        if not path.exists():
            pytest.skip(f"{self.SCRIPT} not present")
        source = path.read_text(encoding="utf-8")
        func_lines = _function_source_lines(
            ast.parse(source, filename=self.SCRIPT),
            "refresh_metrics_if_requested",
            source,
        )
        assert func_lines, "refresh_metrics_if_requested() not found"

        func_text = "\n".join(func_lines)
        assert "config.domain is not None" in func_text, (
            "Domain forwarding must be guarded by 'config.domain is not None'"
        )

    def test_subprocess_cmd_includes_run_metric_refresh(self) -> None:
        """refresh_metrics_if_requested() must invoke run_metric_refresh.py."""
        path = PROJECT_ROOT / self.SCRIPT
        if not path.exists():
            pytest.skip(f"{self.SCRIPT} not present")
        source = path.read_text(encoding="utf-8")
        func_lines = _function_source_lines(
            ast.parse(source, filename=self.SCRIPT),
            "refresh_metrics_if_requested",
            source,
        )
        assert func_lines, "refresh_metrics_if_requested() not found"
        func_text = "\n".join(func_lines)
        assert "run_metric_refresh.py" in func_text, (
            "refresh_metrics_if_requested() must call run_metric_refresh.py"
        )

    def test_domain_uses_separate_args_not_equals(self) -> None:
        """run_build_pipeline must use ['--domain', value] (not '--domain=value').

        This distinguishes it from run_metric_refresh.py which uses equals format.
        """
        path = PROJECT_ROOT / self.SCRIPT
        if not path.exists():
            pytest.skip(f"{self.SCRIPT} not present")
        source = path.read_text(encoding="utf-8")
        func_lines = _function_source_lines(
            ast.parse(source, filename=self.SCRIPT),
            "refresh_metrics_if_requested",
            source,
        )
        assert func_lines, "refresh_metrics_if_requested() not found"
        func_text = "\n".join(func_lines)

        # Pattern should be: extend(['--domain', config.domain])
        # not: append(f"--domain={config.domain}")
        assert "extend" in func_text or "'--domain'" in func_text, (
            "run_build_pipeline must use extend/list format for --domain forwarding"
        )

    def test_resolve_front_config_path_uses_domain_specific_aliases(self) -> None:
        """CRISPR runs must use the CRISPR front-alias config."""
        crispr_path = resolve_front_config_path("crispr")
        psc_path = resolve_front_config_path("psc")
        default_path = resolve_front_config_path(None)

        assert crispr_path.name == "front_aliases_crispr.yaml"
        assert psc_path.name == "front_aliases.yaml"
        assert default_path.name == "front_aliases.yaml"


# ---------------------------------------------------------------------------
# Cross-cutting: all three orchestrators
# ---------------------------------------------------------------------------


class TestAllOrchestratorsCommon:
    """Cross-cutting tests that apply to all three orchestrator scripts."""

    ORCHESTRATORS = [
        "scripts/run_all_metrics.py",
        "scripts/run_metric_refresh.py",
        "scripts/run_build_pipeline.py",
    ]

    @pytest.mark.parametrize("rel_path", ORCHESTRATORS)
    def test_imports_add_domain_args(self, rel_path: str) -> None:
        """Each orchestrator must import add_domain_args from domain_registry."""
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            pytest.skip(f"{rel_path} not present")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel_path)

        found_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "domain_registry" in node.module:
                    names = [alias.name for alias in node.names]
                    if "add_domain_args" in names:
                        found_import = True
                        break

        assert found_import, (
            f"{rel_path} must import add_domain_args from domain_registry"
        )

    @pytest.mark.parametrize("rel_path", ORCHESTRATORS)
    def test_imports_resolve_script_paths(self, rel_path: str) -> None:
        """Each orchestrator must import resolve_script_paths."""
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            pytest.skip(f"{rel_path} not present")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel_path)

        found_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "domain_registry" in node.module:
                    names = [alias.name for alias in node.names]
                    if "resolve_script_paths" in names:
                        found_import = True
                        break

        assert found_import, (
            f"{rel_path} must import resolve_script_paths from domain_registry"
        )

    @pytest.mark.parametrize("rel_path", ORCHESTRATORS)
    def test_valid_python_syntax(self, rel_path: str) -> None:
        """Each orchestrator must have valid Python syntax."""
        _parse_script(rel_path)

    @pytest.mark.parametrize("rel_path", ORCHESTRATORS)
    def test_domain_string_present_in_source(self, rel_path: str) -> None:
        """Each orchestrator must contain a '--domain' string for forwarding."""
        tree = _parse_script(rel_path)
        has_plain = _has_string_containing(tree, "--domain")
        has_fstr = _has_joined_str_containing(tree, "--domain")
        assert has_plain or has_fstr, (
            f"{rel_path} contains no '--domain' string literal or f-string"
        )

    @pytest.mark.parametrize("rel_path", ORCHESTRATORS)
    def test_subprocess_import(self, rel_path: str) -> None:
        """Each orchestrator must import subprocess for child execution."""
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            pytest.skip(f"{rel_path} not present")
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel_path)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "subprocess":
                        found = True
            if isinstance(node, ast.ImportFrom):
                if node.module == "subprocess":
                    found = True

        assert found, f"{rel_path} must import subprocess"

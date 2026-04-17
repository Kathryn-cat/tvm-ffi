# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Testing utilities for printer ↔ parser round-trip checks.

The core invariant this module validates is::

    pyast.parse(pyast.to_python(orig), lang_modules=...) == orig
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tvm_ffi import Object, pyast, structural_equal
from tvm_ffi.structural import get_first_structural_mismatch


_INDENT = "    "


def assert_roundtrip(
    orig: Object,
    lang_modules: dict[str, Any],
    *,
    var_factory: Callable[[str, Any], Any] | None = None,
    check_reprint: bool = True,
    verbose: bool = False,
    file: Any = None,
) -> Object:
    """Assert ``orig`` round-trips through ``to_python`` → ``IRParser.parse``.

    Parameters
    ----------
    orig
        The original trait-decorated IR object.
    lang_modules
        Registry passed to :class:`pyast.IRParser`.
    var_factory
        Optional ``(name, ty) -> Var`` factory passed to
        :class:`pyast.IRParser`. Mandatory when ``orig`` contains custom
        ``Var`` subclasses — ``var_factory`` **must** return the same class
        the originals use, otherwise the final ``==`` fails because
        ``py_class`` equality is exact-class.
    check_reprint
        When ``True`` (default) also asserts the printed text is stable
        under a second round-trip (``to_python(parsed) == to_python(orig)``).
    verbose
        When ``True``, prints a success diagnostic on stdout (or
        ``file``) showing the orig/parsed reprs and the intermediate
        printed text — using the same shape as the failure diff so you
        can compare success and failure traces side by side. Off by
        default so test suites stay quiet.
    file
        Stream to write the verbose diagnostic to. Defaults to
        :data:`sys.stdout`. Ignored when ``verbose=False``.

    Returns
    -------
    parsed
        The parsed IR (useful when the test wants to do further
        per-field assertions).

    Raises
    ------
    AssertionError
        With a diff-style message on any mismatch.
    """
    text = pyast.to_python(orig)
    parser = pyast.IRParser(lang_modules=lang_modules, var_factory=var_factory)
    try:
        parsed_list = parser.parse(text)
    except Exception as exc:
        raise AssertionError(
            f"parse raised {type(exc).__name__}: {exc}\n"
            f"while parsing:\n{_indent(text)}",
        ) from exc

    if not isinstance(parsed_list, list) or len(parsed_list) != 1:
        raise AssertionError(
            f"expected top-level parse to yield a single IR in a list, "
            f"got {type(parsed_list).__name__!r} of size "
            f"{len(parsed_list) if hasattr(parsed_list, '__len__') else '?'}:\n"
            f"  value = {parsed_list!r}\n"
            f"  text was:\n{_indent(text)}",
        )
    parsed = parsed_list[0]

    if not structural_equal(orig, parsed):
        raise AssertionError(_format_ir_diff(orig, parsed, text))

    if check_reprint:
        text2 = pyast.to_python(parsed)
        if text != text2:
            raise AssertionError(
                "print → parse → print is not stable:\n"
                f"  first print:\n{_indent(text)}\n"
                f"  second print:\n{_indent(text2)}",
            )

    if verbose:
        import sys  # noqa: PLC0415

        out = file if file is not None else sys.stdout
        print(_format_ir_success(orig, parsed, text), file=out)

    return parsed


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------


def _indent(text: str, prefix: str = _INDENT) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _path_str(path: Any) -> str:
    """Render an :class:`AccessPath` as a compact dotted path (``.a.b[0]``).

    Falls back to ``repr`` when the object isn't an AccessPath.
    """
    to_steps = getattr(path, "to_steps", None)
    if not callable(to_steps):
        return repr(path)
    try:
        steps = to_steps()
    except Exception:
        return repr(path)
    out: list[str] = []
    for step in steps:
        kind = getattr(step, "kind", None)
        key = getattr(step, "key", None)
        # kind 0 = attr, 1 = array_item, 2 = map_item (per tvm_ffi reflection).
        if kind == 1:
            out.append(f"[{key}]")
        elif kind == 2:
            out.append(f"[{key!r}]")
        else:
            out.append(f".{key}")
    return "".join(out) if out else "(root)"


def _format_ir_diff(orig: Any, parsed: Any, text: str) -> str:
    """Produce a readable structural-diff message between two IR objects."""
    # Use :func:`get_first_structural_mismatch` — locates the first
    # divergence as a pair of AccessPaths into orig / parsed.
    try:
        mismatch = get_first_structural_mismatch(orig, parsed)
    except Exception:
        mismatch = None

    header = "round-trip structural mismatch:"
    body: list[str] = []
    if mismatch is not None:
        body.append(
            f"  first mismatch at: orig@{_path_str(mismatch[0])}"
            f"  parsed@{_path_str(mismatch[1])}",
        )
    body.append(f"  orig  = {orig!r}")
    body.append(f"  parsed = {parsed!r}")
    body.append("")
    body.append(f"intermediate text:\n{_indent(text)}")
    # Also try reprinting both — the text is often the clearest diff signal.
    try:
        text_parsed = pyast.to_python(parsed)
        if text_parsed != text:
            body.append(f"\nreprint of parsed (DIFFERS from intermediate):\n{_indent(text_parsed)}")
    except Exception as exc:  # pylint: disable=broad-except
        body.append(f"\n(reprint of parsed failed: {type(exc).__name__}: {exc})")
    return header + "\n" + "\n".join(body)


def _format_ir_success(orig: Any, parsed: Any, text: str) -> str:
    """Produce a readable success message for a passing round-trip check."""
    header = "round-trip OK:"
    body: list[str] = []
    body.append(f"  orig   = {orig!r}")
    body.append(f"  parsed = {parsed!r}")
    body.append("")
    body.append(f"intermediate text:\n{_indent(text)}")
    return header + "\n" + "\n".join(body)

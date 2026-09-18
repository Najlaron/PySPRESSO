from __future__ import annotations

import base64
import json
import mimetypes
import re
import uuid
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

import pandas as pd


REPORT_SCHEMA_VERSION = 1
DEFAULT_REPORT_FILE_NAME = "report"

_REPORT_LOCK = RLock()


# =============================================================================
# PUBLIC API
# =============================================================================


def configure_report(
    state: Any,
    *,
    workflow_name: str | None = None,
    subtitle: str = "Workflow report",
    logo_path: str | Path | None = None,
    report_directory: str | Path | None = None,
    report_file_name: str | None = None,
) -> str:
    """
    Create/update the report header.

    This function does not inspect analytical outputs.

    Parameters
    ----------
    state
        WorkflowState-like object.

    workflow_name
        Name displayed at the top of the report.

    subtitle
        Small subtitle displayed below the workflow name.

    logo_path
        Optional path to a logo/image. The image is embedded directly in HTML.

    report_directory
        Explicit report directory. If not provided, state.main_folder is used.

    report_file_name
        File name/stem. ".html" is added automatically.

    Returns
    -------
    str
        Path to the HTML report.
    """

    with _REPORT_LOCK:
        paths = _report_paths(
            state,
            report_directory=report_directory,
            report_file_name=report_file_name,
        )

        manifest = _load_or_create_manifest(
            paths,
            workflow_name=workflow_name,
            subtitle=subtitle,
        )

        if workflow_name:
            manifest["workflow_name"] = str(
                workflow_name
            )

        if subtitle is not None:
            manifest["subtitle"] = str(
                subtitle
            )

        if logo_path is not None:
            manifest["logo"] = (
                _image_data_uri_from_path(
                    logo_path
                )
            )

        _persist(
            paths,
            manifest,
        )

        _store_paths_on_state(
            state,
            paths,
        )

        return str(
            paths["html"]
        )


def begin_step(
    state: Any,
    *,
    operation_name: str,
    parameters: Mapping[str, Any] | None = None,
    operation_id: str | None = None,
    step_id: str | None = None,
    workflow_name: str | None = None,
    subtitle: str = "Workflow report",
    logo_path: str | Path | None = None,
    report_directory: str | Path | None = None,
    report_file_name: str | None = None,
) -> str:
    """
    Create a new report block for one execution of a workflow step.

    Re-running the same step creates another independent report block.
    """

    if (
        not operation_name
        or not str(
            operation_name
        ).strip()
    ):
        raise ValueError(
            "operation_name must be a non-empty string."
        )

    with _REPORT_LOCK:
        paths = _report_paths(
            state,
            report_directory=report_directory,
            report_file_name=report_file_name,
        )

        manifest = _load_or_create_manifest(
            paths,
            workflow_name=workflow_name,
            subtitle=subtitle,
        )

        if workflow_name:
            manifest["workflow_name"] = str(
                workflow_name
            )

        if logo_path is not None:
            manifest["logo"] = (
                _image_data_uri_from_path(
                    logo_path
                )
            )

        block_id = uuid.uuid4().hex

        previous_runs = 0

        if step_id is not None:
            previous_runs = sum(
                1
                for block in manifest["blocks"]
                if block.get("step_id")
                == step_id
            )

        block = {
            "id": block_id,
            "step_id": step_id,
            "operation_id": operation_id,
            "operation_name": str(
                operation_name
            ),
            "run_number": previous_runs + 1,
            "parameters": _json_safe(
                dict(
                    parameters or {}
                )
            ),
            "status": "running",
            "started_at": _utc_now_iso(),
            "finished_at": None,
            "items": [],
        }

        manifest["blocks"].append(
            block
        )

        _persist(
            paths,
            manifest,
        )

        _store_paths_on_state(
            state,
            paths,
        )

        # Runtime-only information.
        setattr(
            state,
            "_html_report_active_block_id",
            block_id,
        )

        return block_id


def add_text(
    state: Any,
    text: Any,
    *,
    title: str | None = None,
    preformatted: bool = False,
) -> None:
    """
    Add text to the currently running operation block.
    """

    item = {
        "type": "text",
        "title": (
            None
            if title is None
            else str(title)
        ),
        "text": (
            ""
            if text is None
            else str(text)
        ),
        "preformatted": bool(
            preformatted
        ),
    }

    _append_item(
        state,
        item,
    )


def add_table(
    state: Any,
    table: Any,
    *,
    title: str | None = None,
    include_index: bool = False,
    max_rows: int | None = None,
) -> None:
    """
    Add a table explicitly supplied by an operation.

    Accepted inputs:
        pandas.DataFrame
        pandas.Series
        dict / Mapping
        rectangular sequence

    max_rows=None:
        include the complete table

    max_rows=30:
        explicitly include only the first 30 rows

    Nothing is discovered automatically.
    """

    item = _table_to_item(
        table,
        title=title,
        include_index=include_index,
        max_rows=max_rows,
    )

    _append_item(
        state,
        item,
    )


def add_figure(
    state: Any,
    figure: Any,
    *,
    title: str | None = None,
    caption: str | None = None,
    alt: str = "Plot",
    dpi: int = 160,
) -> None:
    """
    Add a figure/plot explicitly supplied by an operation.

    figure can be an existing image path or a matplotlib-like Figure.

    Images are embedded directly into report.html.
    """

    data_uri = _figure_to_data_uri(
        figure,
        dpi=dpi,
    )

    item = {
        "type": "figure",
        "title": (
            None
            if title is None
            else str(title)
        ),
        "caption": (
            None
            if caption is None
            else str(caption)
        ),
        "alt": str(
            alt or "Plot"
        ),
        "src": data_uri,
    }

    _append_item(
        state,
        item,
    )


def finish_step(
    state: Any,
    *,
    status: str = "success",
) -> None:
    """
    Close the active operation block.
    """

    normalized = (
        str(status)
        .strip()
        .lower()
    )

    aliases = {
        "done": "success",
        "ok": "success",
        "success": "success",
        "failed": "error",
        "failure": "error",
        "error": "error",
        "blocked": "blocked",
        "cancelled": "cancelled",
        "canceled": "cancelled",
    }

    if normalized not in aliases:
        raise ValueError(
            "status must be success, error, blocked or cancelled."
        )

    normalized = aliases[
        normalized
    ]

    with _REPORT_LOCK:
        paths, manifest, block = (
            _load_active_block(
                state
            )
        )

        block["status"] = normalized
        block["finished_at"] = (
            _utc_now_iso()
        )

        _persist(
            paths,
            manifest,
        )

        setattr(
            state,
            "_html_report_active_block_id",
            None,
        )


def get_report_path(
    state: Any,
    *,
    report_directory: str | Path | None = None,
    report_file_name: str | None = None,
) -> str:
    """
    Return the deterministic HTML report path.
    """

    paths = _report_paths(
        state,
        report_directory=report_directory,
        report_file_name=report_file_name,
    )

    return str(
        paths["html"]
    )


# =============================================================================
# PATHS / MANIFEST
# =============================================================================


def _render_time(
    value: Any,
) -> str:
    if not value:
        return ""

    value = str(
        value
    )

    return (
        '<time class="local-time" '
        f'datetime="{escape(value, quote=True)}">'
        f"{escape(value)}"
        "</time>"
    )


def _utc_now_iso() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .replace(
            microsecond=0
        )
        .isoformat()
    )


def _safe_stem(
    value: Any,
) -> str:
    raw = str(
        value
        or DEFAULT_REPORT_FILE_NAME
    ).strip()

    raw = Path(
        raw
    ).stem

    raw = re.sub(
        r"[^\w\-. ]+",
        "_",
        raw,
        flags=re.UNICODE,
    )

    raw = re.sub(
        r"\s+",
        "_",
        raw,
    )

    raw = raw.strip(
        "._"
    )

    return (
        raw
        or DEFAULT_REPORT_FILE_NAME
    )


def _report_paths(
    state: Any,
    *,
    report_directory: str | Path | None = None,
    report_file_name: str | None = None,
) -> dict[str, Path]:

    # During one step execution use the exact paths created by begin_step().
    if (
        report_directory is None
        and report_file_name is None
        and getattr(
            state,
            "_html_report_html_path",
            None,
        )
    ):
        html_path = Path(
            state._html_report_html_path
        )

        manifest_path = Path(
            state._html_report_manifest_path
        )

        return {
            "directory": html_path.parent,
            "html": html_path,
            "manifest": manifest_path,
        }

    directory_value = (
        report_directory
    )

    if directory_value is None:
        directory_value = getattr(
            state,
            "main_folder",
            None,
        )

    if (
        directory_value is None
        or str(
            directory_value
        ).strip()
        == ""
    ):
        raise ValueError(
            "No HTML report directory is available. "
            "Set state.main_folder or pass report_directory."
        )

    directory = Path(
        directory_value
    ).expanduser()

    # Keep workflow paths compatible with the existing frontend.
    # "Demo Workflow" and "outputs/Demo Workflow" resolve to the same place.
    if not directory.is_absolute():
        parts = directory.parts

        if (
            not parts
            or parts[0] != "outputs"
        ):
            directory = (
                Path("outputs")
                / directory
            )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    configured_name = (
        report_file_name
    )

    if configured_name is None:
        configured_name = getattr(
            state,
            "report_file_name",
            None,
        )

    stem = _safe_stem(
        configured_name
    )

    html_path = (
        directory
        / f"{stem}.html"
    )

    manifest_path = (
        directory
        / f"{stem}.report.json"
    )

    return {
        "directory": directory,
        "html": html_path,
        "manifest": manifest_path,
    }


def _default_manifest(
    *,
    workflow_name: str | None,
    subtitle: str,
) -> dict[str, Any]:
    now = _utc_now_iso()

    return {
        "schema_version":
            REPORT_SCHEMA_VERSION,
        "workflow_name":
            str(
                workflow_name
                or "PySPRESSO workflow"
            ),
        "subtitle":
            str(
                subtitle
                or ""
            ),
        "logo":
            None,
        "created_at":
            now,
        "updated_at":
            now,
        "blocks":
            [],
    }


def _load_or_create_manifest(
    paths: Mapping[str, Path],
    *,
    workflow_name: str | None,
    subtitle: str,
) -> dict[str, Any]:

    manifest_path = paths[
        "manifest"
    ]

    if manifest_path.is_file():
        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            manifest = json.load(
                handle
            )

        if not isinstance(
            manifest,
            dict,
        ):
            raise ValueError(
                "HTML report manifest is malformed."
            )

        manifest.setdefault(
            "schema_version",
            REPORT_SCHEMA_VERSION,
        )

        manifest.setdefault(
            "workflow_name",
            workflow_name
            or "PySPRESSO workflow",
        )

        manifest.setdefault(
            "subtitle",
            subtitle or "",
        )

        manifest.setdefault(
            "logo",
            None,
        )

        manifest.setdefault(
            "created_at",
            _utc_now_iso(),
        )

        manifest.setdefault(
            "updated_at",
            _utc_now_iso(),
        )

        manifest.setdefault(
            "blocks",
            [],
        )

        return manifest

    return _default_manifest(
        workflow_name=workflow_name,
        subtitle=subtitle,
    )


def _persist(
    paths: Mapping[str, Path],
    manifest: dict[str, Any],
) -> None:

    manifest["updated_at"] = (
        _utc_now_iso()
    )

    manifest_text = json.dumps(
        _json_safe(
            manifest
        ),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

    html_text = _render_html(
        manifest
    )

    _atomic_write_text(
        paths["manifest"],
        manifest_text,
    )

    _atomic_write_text(
        paths["html"],
        html_text,
    )


def _atomic_write_text(
    path: Path,
    text: str,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        path.with_name(
            path.name + ".tmp"
        )
    )

    temporary.write_text(
        text,
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def _store_paths_on_state(
    state: Any,
    paths: Mapping[str, Path],
) -> None:

    # Runtime paths used by add_text/add_table/add_figure.
    setattr(
        state,
        "_html_report_html_path",
        str(
            paths["html"]
        ),
    )

    setattr(
        state,
        "_html_report_manifest_path",
        str(
            paths["manifest"]
        ),
    )

    # Optional public paths.
    try:
        setattr(
            state,
            "report_html_path",
            str(
                paths["html"]
            ),
        )

        setattr(
            state,
            "report_manifest_path",
            str(
                paths["manifest"]
            ),
        )

    except Exception:
        pass


# =============================================================================
# ACTIVE BLOCK
# =============================================================================


def _load_active_block(
    state: Any,
) -> tuple[
    dict[str, Path],
    dict[str, Any],
    dict[str, Any],
]:

    block_id = getattr(
        state,
        "_html_report_active_block_id",
        None,
    )

    if not block_id:
        raise RuntimeError(
            "No active HTML report block. "
            "begin_step() must be called before "
            "add_text(), add_table(), add_figure() "
            "or finish_step()."
        )

    paths = _report_paths(
        state
    )

    if not paths[
        "manifest"
    ].is_file():
        raise RuntimeError(
            "HTML report manifest does not exist: "
            f"{paths['manifest']}"
        )

    with paths[
        "manifest"
    ].open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    for block in reversed(
        manifest.get(
            "blocks",
            [],
        )
    ):
        if (
            block.get("id")
            == block_id
        ):
            return (
                paths,
                manifest,
                block,
            )

    raise RuntimeError(
        "Active HTML report block "
        f"'{block_id}' was not found."
    )


def _append_item(
    state: Any,
    item: dict[str, Any],
) -> None:

    with _REPORT_LOCK:
        paths, manifest, block = (
            _load_active_block(
                state
            )
        )

        block.setdefault(
            "items",
            [],
        ).append(
            _json_safe(
                item
            )
        )

        _persist(
            paths,
            manifest,
        )


# =============================================================================
# CONTENT SERIALIZATION
# =============================================================================


def _json_safe(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        (str, bool, int),
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if pd.isna(
            value
        ):
            return None

        if value in (
            float("inf"),
            float("-inf"),
        ):
            return None

        return value

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key):
                _json_safe(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            _json_safe(item)
            for item
            in value
        ]

    try:
        if pd.isna(
            value
        ):
            return None

    except Exception:
        pass

    if hasattr(
        value,
        "item",
    ):
        try:
            return _json_safe(
                value.item()
            )

        except Exception:
            pass

    if hasattr(
        value,
        "isoformat",
    ):
        try:
            return value.isoformat()

        except Exception:
            pass

    return str(
        value
    )


def _table_to_item(
    table: Any,
    *,
    title: str | None,
    include_index: bool,
    max_rows: int | None,
) -> dict[str, Any]:

    if isinstance(
        table,
        pd.Series,
    ):
        frame = table.to_frame()

    elif isinstance(
        table,
        pd.DataFrame,
    ):
        frame = table.copy()

    elif isinstance(
        table,
        Mapping,
    ):
        frame = pd.DataFrame(
            [
                (key, value)
                for key, value
                in table.items()
            ],
            columns=[
                "Key",
                "Value",
            ],
        )

    elif (
        isinstance(
            table,
            Sequence,
        )
        and not isinstance(
            table,
            (
                str,
                bytes,
                bytearray,
            ),
        )
    ):
        frame = pd.DataFrame(
            table
        )

    else:
        raise TypeError(
            "table must be a pandas DataFrame/Series, "
            "mapping or rectangular sequence."
        )

    total_rows = len(
        frame
    )

    if max_rows is not None:
        max_rows = int(
            max_rows
        )

        if max_rows < 1:
            raise ValueError(
                "max_rows must be at least 1 or None."
            )

        frame = frame.head(
            max_rows
        )

    if include_index:
        frame = frame.reset_index()

    columns = [
        str(column)
        for column
        in frame.columns
    ]

    rows = [
        [
            _json_safe(
                value
            )
            for value
            in row
        ]
        for row
        in frame.itertuples(
            index=False,
            name=None,
        )
    ]

    return {
        "type": "table",
        "title": (
            None
            if title is None
            else str(title)
        ),
        "columns": columns,
        "rows": rows,
        "shown_rows": len(
            frame
        ),
        "total_rows": total_rows,
    }


# =============================================================================
# FIGURES / LOGO
# =============================================================================


def _image_data_uri_from_path(
    path_value: str | Path,
) -> str:

    path = Path(
        path_value
    ).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            "Image file does not exist: "
            f"{path}"
        )

    mime_type, _ = (
        mimetypes.guess_type(
            path.name
        )
    )

    if (
        not mime_type
        or not mime_type.startswith(
            "image/"
        )
    ):
        raise ValueError(
            "Unsupported image type for HTML report: "
            f"{path.suffix}"
        )

    encoded = (
        base64.b64encode(
            path.read_bytes()
        )
        .decode(
            "ascii"
        )
    )

    return (
        f"data:{mime_type};base64,"
        f"{encoded}"
    )


def _figure_to_data_uri(
    figure: Any,
    *,
    dpi: int,
) -> str:

    # Existing saved plot.
    if isinstance(
        figure,
        (str, Path),
    ):
        return (
            _image_data_uri_from_path(
                figure
            )
        )

    # Raw PNG bytes.
    if isinstance(
        figure,
        (bytes, bytearray),
    ):
        encoded = (
            base64.b64encode(
                bytes(
                    figure
                )
            )
            .decode(
                "ascii"
            )
        )

        return (
            "data:image/png;base64,"
            + encoded
        )

    # Matplotlib-like Figure.
    if hasattr(
        figure,
        "savefig",
    ):
        buffer = BytesIO()

        figure.savefig(
            buffer,
            format="png",
            dpi=int(
                dpi
            ),
            bbox_inches="tight",
        )

        encoded = (
            base64.b64encode(
                buffer.getvalue()
            )
            .decode(
                "ascii"
            )
        )

        return (
            "data:image/png;base64,"
            + encoded
        )

    raise TypeError(
        "figure must be an image path, image bytes "
        "or an object exposing savefig()."
    )


# =============================================================================
# HTML RENDERING
# =============================================================================

def _render_report_script() -> str:
    return """
<script>

/* Convert UTC timestamps stored by the backend
   to the local timezone of the browser. */
document
    .querySelectorAll("time.local-time")
    .forEach((element) => {

        const value =
            element.getAttribute("datetime");

        const date =
            new Date(value);

        if (!Number.isNaN(date.getTime())) {
            element.textContent =
                date.toLocaleString(
                    undefined,
                    {
                        year: "numeric",
                        month: "2-digit",
                        day: "2-digit",
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit"
                    }
                );
        }
    });


const reportBlocks =
    Array.from(
        document.querySelectorAll(
            "details.report-block"
        )
    );


const tocLinks =
    Array.from(
        document.querySelectorAll(
            ".toc-link"
        )
    );


/* Expand all workflow steps. */
document
    .getElementById("expand-all")
    ?.addEventListener(
        "click",
        () => {

            reportBlocks.forEach(
                (block) => {
                    block.open = true;
                }
            );
        }
    );


/* Collapse all workflow steps. */
document
    .getElementById("collapse-all")
    ?.addEventListener(
        "click",
        () => {

            reportBlocks.forEach(
                (block) => {
                    block.open = false;
                }
            );
        }
    );


/* Clicking a TOC link opens the target block
   and scrolls directly to it. */
tocLinks.forEach(
    (link) => {

        link.addEventListener(
            "click",
            (event) => {

                const targetId =
                    link.dataset.stepTarget;

                const target =
                    document.getElementById(
                        targetId
                    );

                if (!target) {
                    return;
                }

                event.preventDefault();

                target.open = true;

                target.scrollIntoView(
                    {
                        behavior: "smooth",
                        block: "start"
                    }
                );

                tocLinks.forEach(
                    (item) => {
                        item.classList.remove(
                            "active"
                        );
                    }
                );

                link.classList.add(
                    "active"
                );

                if (history.replaceState) {
                    history.replaceState(
                        null,
                        "",
                        "#" + targetId
                    );
                }
            }
        );
    }
);


/* Highlight the TOC item corresponding to the
   workflow step currently in view. */
if ("IntersectionObserver" in window) {

    const observer =
        new IntersectionObserver(
            (entries) => {

                const visible =
                    entries
                        .filter(
                            (entry) =>
                                entry.isIntersecting
                        )
                        .sort(
                            (a, b) =>
                                a.boundingClientRect.top
                                - b.boundingClientRect.top
                        );

                if (visible.length === 0) {
                    return;
                }

                const activeId =
                    visible[0].target.id;

                tocLinks.forEach(
                    (link) => {

                        link.classList.toggle(
                            "active",
                            link.dataset.stepTarget
                            === activeId
                        );
                    }
                );
            },
            {
                rootMargin:
                    "-10% 0px -78% 0px",

                threshold:
                    0
            }
        );

    reportBlocks.forEach(
        (block) => {
            observer.observe(block);
        }
    );
}


/* Open the requested block if the report URL
   contains an anchor. */
const initiallyRequestedId =
    decodeURIComponent(
        window.location.hash.replace(
            /^#/,
            ""
        )
    );

if (initiallyRequestedId) {

    const requestedBlock =
        document.getElementById(
            initiallyRequestedId
        );

    if (
        requestedBlock
        &&
        requestedBlock.matches(
            "details.report-block"
        )
    ) {
        requestedBlock.open = true;
    }
}


/* Printing should always include all report
   content even when blocks are collapsed. */
let printOpenState = null;

window.addEventListener(
    "beforeprint",
    () => {

        printOpenState =
            reportBlocks.map(
                (block) => block.open
            );

        reportBlocks.forEach(
            (block) => {
                block.open = true;
            }
        );
    }
);


window.addEventListener(
    "afterprint",
    () => {

        if (!printOpenState) {
            return;
        }

        reportBlocks.forEach(
            (block, index) => {

                block.open =
                    Boolean(
                        printOpenState[index]
                    );
            }
        );

        printOpenState = null;
    }
);

</script>
"""

def _render_html(
    manifest: Mapping[str, Any],
) -> str:

    workflow_name = escape(
        str(
            manifest.get(
                "workflow_name"
            )
            or "PySPRESSO workflow"
        )
    )

    subtitle = escape(
        str(
            manifest.get(
                "subtitle"
            )
            or ""
        )
    )

    logo = manifest.get(
        "logo"
    )

    logo_html = ""

    if logo:
        logo_html = (
            '<img class="brand-logo" src="'
            + escape(
                str(logo),
                quote=True,
            )
            + '" alt="PySPRESSO logo">'
        )

    blocks = [
        block
        for block
        in manifest.get(
            "blocks",
            [],
        )
        if isinstance(
            block,
            Mapping,
        )
    ]

    blocks_html = "\n".join(
        _render_block(
            block,
            block_index=index,
        )
        for index, block
        in enumerate(
            blocks,
            start=1,
        )
    )

    toc_html = _render_toc(blocks)

    script_html = _render_report_script()

    return f"""<!doctype html>

<html lang="en">

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
>

<title>
    {workflow_name} | PySPRESSO report
</title>

<style>

:root {{
    --espresso: #713105;
    --espresso-dark: #4b2105;
    --coffee: #9a5b2d;
    --latte: #c99b72;
    --crema: #f4eadf;
    --foam: #fffaf6;
    --paper: #ffffff;
    --ink: #2c1b12;
    --muted: #7b675a;
    --line: #e2d1c2;

    --success: #58744b;
    --error: #a84a38;
    --blocked: #8a6a2f;
    --running: #6f5d9c;

    --shadow:
        0 10px 28px
        rgba(76, 39, 15, 0.09);
}}

* {{
    box-sizing:
        border-box;
}}

html {{
    scroll-behavior:
        smooth;
}}

body {{
    margin: 0;

    background:
        var(--crema);

    color:
        var(--ink);

    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    line-height:
        1.55;
}}

.report-shell {{
    width:
        min(
            1480px,
            calc(100% - 32px)
        );

    margin:
        32px auto 56px;
}}


/* ------------------------------------------------------------------
   HEADER
   ------------------------------------------------------------------ */

.report-header {{
    position:
        relative;

    overflow:
        hidden;

    display:
        flex;

    align-items:
        center;

    gap:
        22px;

    padding:
        28px 32px;

    border-radius:
        22px;

    background:
        linear-gradient(
            135deg,
            var(--espresso-dark),
            var(--espresso)
        );

    color:
        white;

    box-shadow:
        var(--shadow);
}}

.report-header::after {{
    content:
        "";

    position:
        absolute;

    right:
        -90px;

    top:
        -110px;

    width:
        280px;

    height:
        280px;

    border-radius:
        50%;

    background:
        rgba(
            255,
            255,
            255,
            0.06
        );
}}

.brand-logo {{
    position:
        relative;

    z-index:
        1;

    width:
        78px;

    height:
        78px;

    object-fit:
        contain;

    background:
        rgba(
            255,
            255,
            255,
            0.96
        );

    border-radius:
        18px;

    padding:
        8px;
}}

.brand-copy {{
    position:
        relative;

    z-index:
        1;
}}

.eyebrow {{
    margin:
        0 0 4px;

    font-size:
        0.78rem;

    font-weight:
        800;

    letter-spacing:
        0.16em;

    text-transform:
        uppercase;

    color:
        #f0d7c2;
}}

.report-header h1 {{
    margin:
        0;

    font-size:
        clamp(
            1.8rem,
            4vw,
            3rem
        );

    line-height:
        1.1;
}}

.subtitle {{
    margin:
        8px 0 0;

    color:
        #f4e4d6;
}}

.report-meta {{
    display:
        flex;

    justify-content:
        space-between;

    gap:
        16px;

    margin:
        13px 5px 25px;

    color:
        var(--muted);

    font-size:
        0.86rem;
}}


/* ------------------------------------------------------------------
   MAIN LAYOUT / TABLE OF CONTENTS
   ------------------------------------------------------------------ */

.report-layout {{
    display:
        grid;

    grid-template-columns:
        270px minmax(0, 1fr);

    gap:
        24px;

    align-items:
        start;
}}

.report-sidebar {{
    position:
        sticky;

    top:
        18px;

    min-width:
        0;
}}

.toc-card {{
    max-height:
        calc(100vh - 36px);

    overflow-y:
        auto;

    padding:
        16px;

    border:
        1px solid var(--line);

    border-radius:
        16px;

    background:
        var(--paper);

    box-shadow:
        var(--shadow);
}}

.toc-title {{
    margin-bottom:
        11px;

    color:
        var(--espresso-dark);

    font-size:
        0.8rem;

    font-weight:
        850;

    letter-spacing:
        0.1em;

    text-transform:
        uppercase;
}}

.toc-actions {{
    display:
        grid;

    grid-template-columns:
        1fr 1fr;

    gap:
        7px;

    margin-bottom:
        13px;
}}

.toc-actions button {{
    appearance:
        none;

    border:
        1px solid var(--line);

    border-radius:
        8px;

    padding:
        7px 8px;

    background:
        var(--foam);

    color:
        var(--espresso);

    font:
        inherit;

    font-size:
        0.72rem;

    font-weight:
        750;

    cursor:
        pointer;
}}

.toc-actions button:hover {{
    background:
        #f7e9dc;
}}

.toc-nav {{
    display:
        flex;

    flex-direction:
        column;

    gap:
        5px;
}}

.toc-link {{
    display:
        grid;

    grid-template-columns:
        25px minmax(0, 1fr);

    gap:
        8px;

    align-items:
        center;

    padding:
        8px 9px;

    border-radius:
        9px;

    color:
        var(--ink);

    text-decoration:
        none;
}}

.toc-link:hover,
.toc-link.active {{
    background:
        #f8eadf;
}}

.toc-number {{
    display:
        inline-flex;

    align-items:
        center;

    justify-content:
        center;

    width:
        24px;

    height:
        24px;

    border-radius:
        999px;

    background:
        #f2dfcf;

    color:
        var(--espresso);

    font-size:
        0.7rem;

    font-weight:
        850;
}}

.toc-label-wrap {{
    min-width:
        0;
}}

.toc-label {{
    display:
        block;

    overflow:
        hidden;

    text-overflow:
        ellipsis;

    white-space:
        nowrap;

    color:
        var(--espresso-dark);

    font-size:
        0.82rem;

    font-weight:
        700;
}}

.toc-meta {{
    display:
        block;

    margin-top:
        2px;

    color:
        var(--muted);

    font-size:
        0.66rem;

    text-transform:
        uppercase;

    letter-spacing:
        0.04em;
}}

.toc-empty {{
    color:
        var(--muted);

    font-size:
        0.82rem;
}}

.report-content {{
    min-width:
        0;
}}


/* ------------------------------------------------------------------
   WORKFLOW STEP BLOCK
   ------------------------------------------------------------------ */

.report-block {{
    margin:
        0 0 22px;

    border:
        1px solid var(--line);

    border-radius:
        18px;

    background:
        var(--paper);

    box-shadow:
        var(--shadow);

    overflow:
        hidden;

    scroll-margin-top:
        22px;
}}

.block-header {{
    display:
        grid;

    grid-template-columns:
        auto minmax(0, 1fr) auto;

    gap:
        14px;

    align-items:
        flex-start;

    padding:
        20px 24px;

    background:
        linear-gradient(
            180deg,
            #fffdfb,
            #faf3ed
        );

    cursor:
        pointer;

    list-style:
        none;

    user-select:
        none;
}}

.block-header::-webkit-details-marker {{
    display:
        none;
}}

.block-header::marker {{
    content:
        "";
}}

.block-header::before {{
    content:
        "▼";

    margin-top:
        4px;

    color:
        var(--espresso);

    font-size:
        0.78rem;

    transform-origin:
        center;

    transition:
        transform 0.18s ease;
}}

.report-block:not([open])
> .block-header::before {{
    transform:
        rotate(-90deg);
}}

.block-header:hover {{
    background:
        linear-gradient(
            180deg,
            #fffaf6,
            #f8eee5
        );
}}

.report-block[open]
> .block-header {{
    border-bottom:
        1px solid var(--line);
}}

.block-title-wrap {{
    min-width:
        0;
}}

.block-kicker {{
    margin-bottom:
        4px;

    font-size:
        0.76rem;

    font-weight:
        800;

    color:
        var(--coffee);

    text-transform:
        uppercase;

    letter-spacing:
        0.1em;
}}

.block-header h2 {{
    margin:
        0;

    color:
        var(--espresso-dark);

    font-size:
        1.35rem;
}}

.block-time {{
    margin-top:
        5px;

    font-size:
        0.82rem;

    color:
        var(--muted);
}}

.status {{
    flex:
        0 0 auto;

    display:
        inline-flex;

    align-items:
        center;

    border-radius:
        999px;

    padding:
        6px 10px;

    font-size:
        0.77rem;

    font-weight:
        800;

    text-transform:
        uppercase;

    letter-spacing:
        0.06em;

    color:
        white;
}}

.status-running {{
    background:
        var(--running);
}}

.status-success {{
    background:
        var(--success);
}}

.status-error {{
    background:
        var(--error);
}}

.status-blocked {{
    background:
        var(--blocked);
}}

.status-cancelled {{
    background:
        var(--muted);
}}

.block-body {{
    padding-bottom:
        1px;
}}


/* ------------------------------------------------------------------
   PARAMETERS
   ------------------------------------------------------------------ */

.parameters {{
    padding:
        18px 24px 2px;
}}

.parameters summary {{
    cursor:
        pointer;

    color:
        var(--espresso);

    font-weight:
        750;

    user-select:
        none;
}}

.parameter-grid {{
    display:
        grid;

    grid-template-columns:
        minmax(170px, 0.32fr)
        minmax(0, 1fr);

    margin-top:
        12px;

    border:
        1px solid var(--line);

    border-radius:
        12px;

    overflow:
        hidden;
}}

.parameter-key,
.parameter-value {{
    padding:
        9px 11px;

    border-bottom:
        1px solid var(--line);
}}

.parameter-key {{
    background:
        #fbf5ef;

    color:
        var(--espresso-dark);

    font-weight:
        700;
}}

.parameter-value {{
    min-width:
        0;

    overflow-wrap:
        anywhere;
}}

pre {{
    margin:
        0;

    white-space:
        pre-wrap;

    overflow-wrap:
        anywhere;

    font:
        0.88rem/1.45
        ui-monospace,
        SFMono-Regular,
        Menlo,
        Consolas,
        monospace;
}}


/* ------------------------------------------------------------------
   STEP CONTENT
   ------------------------------------------------------------------ */

.block-content {{
    padding:
        6px 24px 24px;
}}

.report-item {{
    margin-top:
        20px;
}}

.report-item h3 {{
    margin:
        0 0 9px;

    color:
        var(--espresso);

    font-size:
        1.02rem;
}}

.text-item p {{
    margin:
        0;

    white-space:
        pre-wrap;
}}

.table-wrap {{
    width:
        100%;

    overflow-x:
        auto;

    border:
        1px solid var(--line);

    border-radius:
        12px;
}}

table {{
    width:
        100%;

    border-collapse:
        collapse;

    font-size:
        0.9rem;
}}

th {{
    background:
        var(--espresso);

    color:
        white;

    text-align:
        left;
}}

th,
td {{
    padding:
        8px 10px;

    border-bottom:
        1px solid var(--line);

    vertical-align:
        top;

    white-space:
        nowrap;
}}

tbody tr:nth-child(even) {{
    background:
        #fcf8f4;
}}

.table-note {{
    margin-top:
        7px;

    color:
        var(--muted);

    font-size:
        0.8rem;
}}

.figure-card {{
    margin:
        0;

    padding:
        14px;

    border:
        1px solid var(--line);

    border-radius:
        14px;

    background:
        var(--foam);
}}

.figure-card img {{
    display:
        block;

    max-width:
        100%;

    height:
        auto;

    margin:
        0 auto;

    border-radius:
        8px;
}}

.figure-card figcaption {{
    margin-top:
        10px;

    color:
        var(--muted);

    font-size:
        0.88rem;
}}

.report-footer {{
    margin-top:
        28px;

    text-align:
        center;

    color:
        var(--muted);

    font-size:
        0.8rem;
}}


/* ------------------------------------------------------------------
   RESPONSIVE
   ------------------------------------------------------------------ */

@media (max-width: 1050px) {{

    .report-layout {{
        grid-template-columns:
            1fr;
    }}

    .report-sidebar {{
        position:
            static;
    }}

    .toc-card {{
        max-height:
            none;
    }}

    .toc-nav {{
        display:
            grid;

        grid-template-columns:
            repeat(
                auto-fit,
                minmax(210px, 1fr)
            );
    }}
}}

@media (max-width: 700px) {{

    .report-shell {{
        width:
            calc(100% - 18px);

        margin-top:
            10px;
    }}

    .report-header {{
        padding:
            22px;
    }}

    .brand-logo {{
        width:
            58px;

        height:
            58px;
    }}

    .report-meta {{
        flex-direction:
            column;
    }}

    .block-header {{
        grid-template-columns:
            auto minmax(0, 1fr);
    }}

    .block-header .status {{
        grid-column:
            2;

        justify-self:
            start;
    }}

    .parameter-grid {{
        grid-template-columns:
            1fr;
    }}
}}


/* ------------------------------------------------------------------
   PRINT
   ------------------------------------------------------------------ */

@media print {{

    body {{
        background:
            white;
    }}

    .report-shell {{
        width:
            100%;

        margin:
            0;
    }}

    .report-layout {{
        display:
            block;
    }}

    .report-sidebar {{
        display:
            none;
    }}

    .report-header,
    .report-block {{
        box-shadow:
            none;
    }}

    .report-block {{
        break-inside:
            avoid;
    }}
}}

</style>

</head>


<body>

<div class="report-shell">


<header class="report-header">

    {logo_html}

    <div class="brand-copy">

        <p class="eyebrow">
            PySPRESSO
        </p>

        <h1>
            {workflow_name}
        </h1>

        <p class="subtitle">
            {subtitle}
        </p>

    </div>

</header>


<div class="report-meta">

    <span>
        Created:
        {_render_time(manifest.get("created_at"))}
    </span>

    <span>
        Updated:
        {_render_time(manifest.get("updated_at"))}
    </span>

</div>


<div class="report-layout">

    <aside class="report-sidebar">

        <div class="toc-card">

            <div class="toc-title">
                Contents
            </div>

            <div class="toc-actions">

                <button
                    type="button"
                    id="expand-all"
                >
                    Expand all
                </button>

                <button
                    type="button"
                    id="collapse-all"
                >
                    Collapse all
                </button>

            </div>

            <nav class="toc-nav">
                {toc_html}
            </nav>

        </div>

    </aside>


    <main class="report-content">

        {blocks_html}

    </main>

</div>


<footer class="report-footer">
    Generated by PySPRESSO
</footer>


</div>

{script_html}

</body>
</html>
"""


def _block_anchor(
    block: Mapping[str, Any],
    block_index: int,
) -> str:
    """
    Return a safe, stable HTML anchor for a report block.
    """

    raw_id = str(
        block.get(
            "id"
        )
        or f"step-{block_index}"
    )

    safe_id = re.sub(
        r"[^A-Za-z0-9_-]+",
        "-",
        raw_id,
    ).strip(
        "-"
    )

    if not safe_id:
        safe_id = (
            f"step-{block_index}"
        )

    return (
        "block-"
        + safe_id
    )


def _render_toc(
    blocks: Sequence[
        Mapping[str, Any]
    ],
) -> str:
    """
    Render the clickable report table of contents.
    """

    if not blocks:
        return (
            '<div class="toc-empty">'
            "No workflow steps have been recorded yet."
            "</div>"
        )

    entries = []

    for index, block in enumerate(
        blocks,
        start=1,
    ):
        anchor = _block_anchor(
            block,
            index,
        )

        operation_name = escape(
            str(
                block.get(
                    "operation_name"
                )
                or block.get(
                    "operation_id"
                )
                or f"Step {index}"
            )
        )

        run_number = escape(
            str(
                block.get(
                    "run_number",
                    1,
                )
            )
        )

        status = str(
            block.get(
                "status"
            )
            or "running"
        ).lower()

        entries.append(
            f"""
<a
    class="toc-link"
    href="#{anchor}"
    data-step-target="{anchor}"
>

    <span class="toc-number">
        {index}
    </span>

    <span class="toc-label-wrap">

        <span class="toc-label">
            {operation_name}
        </span>

        <span class="toc-meta">
            Run {run_number}
            ·
            {escape(status)}
        </span>

    </span>

</a>
"""
        )

    return "\n".join(
        entries
    )


def _render_block(
    block: Mapping[str, Any],
    block_index: int,
) -> str:

    status = str(
        block.get(
            "status"
        )
        or "running"
    ).lower()

    status_class = (
        re.sub(
            r"[^a-z-]",
            "",
            status,
        )
        or "running"
    )

    operation_name = escape(
        str(
            block.get(
                "operation_name"
            )
            or "Operation"
        )
    )

    operation_id = block.get(
        "operation_id"
    )

    run_number = block.get(
        "run_number",
        1,
    )

    kicker_parts = [
        "Run "
        + escape(
            str(
                run_number
            )
        )
    ]

    if operation_id:
        kicker_parts.append(
            escape(
                str(
                    operation_id
                )
            )
        )

    kicker = " · ".join(
        kicker_parts
    )

    started_at = block.get(
        "started_at"
    )

    finished_at = block.get(
        "finished_at"
    )

    time_text = _render_time(
        started_at
    )

    if finished_at:
        time_text += (
            " → "
            + _render_time(
                finished_at
            )
        )

    parameters_html = (
        _render_parameters(
            block.get(
                "parameters"
            )
            or {}
        )
    )

    items_html = "\n".join(
        _render_item(
            item
        )
        for item
        in block.get(
            "items",
            [],
        )
        if isinstance(
            item,
            Mapping,
        )
    )

    anchor = _block_anchor(
        block,
        block_index,
    )

    return f"""
<details
    class="report-block"
    id="{anchor}"
    open
>

<summary class="block-header">

    <div class="block-title-wrap">

        <div class="block-kicker">
            {kicker}
        </div>

        <h2>
            {operation_name}
        </h2>

        <div class="block-time">
            {time_text}
        </div>

    </div>

    <span
        class="status status-{status_class}"
    >
        {escape(status)}
    </span>

</summary>


<div class="block-body">

    {parameters_html}

    <div class="block-content">
        {items_html}
    </div>

</div>


</details>
"""


def _render_parameters(
    parameters: Mapping[str, Any],
) -> str:

    if not parameters:
        return """
<details class="parameters" open>

<summary>
    Parameters
</summary>

<div class="parameter-grid">

    <div class="parameter-key">
        Parameters
    </div>

    <div class="parameter-value">
        None
    </div>

</div>

</details>
"""

    cells = []

    for key, value in (
        parameters.items()
    ):
        cells.append(
            '<div class="parameter-key">'
            + escape(
                str(key)
            )
            + "</div>"
        )

        cells.append(
            '<div class="parameter-value">'
            + _render_value(
                value
            )
            + "</div>"
        )

    return (
        '<details class="parameters" open>'
        "<summary>"
        "Parameters"
        "</summary>"
        '<div class="parameter-grid">'
        + "".join(
            cells
        )
        + "</div>"
        "</details>"
    )


def _render_value(
    value: Any,
) -> str:

    if isinstance(
        value,
        (
            dict,
            list,
            tuple,
        ),
    ):
        text = json.dumps(
            _json_safe(
                value
            ),
            ensure_ascii=False,
            indent=2,
        )

        return (
            "<pre>"
            + escape(
                text
            )
            + "</pre>"
        )

    return escape(
        ""
        if value is None
        else str(
            value
        )
    )


def _render_item(
    item: Mapping[str, Any],
) -> str:

    item_type = item.get(
        "type"
    )

    if item_type == "text":
        return _render_text_item(
            item
        )

    if item_type == "table":
        return _render_table_item(
            item
        )

    if item_type == "figure":
        return _render_figure_item(
            item
        )

    return ""


def _render_item_title(
    title: Any,
) -> str:

    if (
        title is None
        or str(
            title
        ).strip()
        == ""
    ):
        return ""

    return (
        "<h3>"
        + escape(
            str(
                title
            )
        )
        + "</h3>"
    )


def _render_text_item(
    item: Mapping[str, Any],
) -> str:

    title = _render_item_title(
        item.get(
            "title"
        )
    )

    text = escape(
        str(
            item.get(
                "text"
            )
            or ""
        )
    )

    if item.get(
        "preformatted"
    ):
        body = (
            "<pre>"
            + text
            + "</pre>"
        )

    else:
        body = (
            "<p>"
            + text
            + "</p>"
        )

    return (
        '<div class="report-item text-item">'
        + title
        + body
        + "</div>"
    )


def _render_table_item(
    item: Mapping[str, Any],
) -> str:

    title = _render_item_title(
        item.get(
            "title"
        )
    )

    columns = (
        item.get(
            "columns"
        )
        or []
    )

    rows = (
        item.get(
            "rows"
        )
        or []
    )

    header = "".join(
        "<th>"
        + escape(
            str(
                column
            )
        )
        + "</th>"
        for column
        in columns
    )

    body_rows = []

    for row in rows:
        cells = "".join(
            "<td>"
            + _render_value(
                value
            )
            + "</td>"
            for value
            in row
        )

        body_rows.append(
            "<tr>"
            + cells
            + "</tr>"
        )

    shown_rows = item.get(
        "shown_rows"
    )

    total_rows = item.get(
        "total_rows"
    )

    note = ""

    if (
        isinstance(
            shown_rows,
            int,
        )
        and isinstance(
            total_rows,
            int,
        )
        and shown_rows < total_rows
    ):
        note = (
            '<div class="table-note">'
            f"Showing {shown_rows} of "
            f"{total_rows} rows."
            "</div>"
        )

    return f"""
<div class="report-item table-item">

{title}

<div class="table-wrap">

<table>

<thead>
<tr>
{header}
</tr>
</thead>

<tbody>
{"".join(body_rows)}
</tbody>

</table>

</div>

{note}

</div>
"""


def _render_figure_item(
    item: Mapping[str, Any],
) -> str:

    title = _render_item_title(
        item.get(
            "title"
        )
    )

    src = escape(
        str(
            item.get(
                "src"
            )
            or ""
        ),
        quote=True,
    )

    alt = escape(
        str(
            item.get(
                "alt"
            )
            or "Plot"
        ),
        quote=True,
    )

    caption = item.get(
        "caption"
    )

    caption_html = ""

    if (
        caption is not None
        and str(
            caption
        ).strip()
        != ""
    ):
        caption_html = (
            "<figcaption>"
            + escape(
                str(
                    caption
                )
            )
            + "</figcaption>"
        )

    return f"""
<div class="report-item figure-item">

{title}

<figure class="figure-card">

<img
    src="{src}"
    alt="{alt}"
>

{caption_html}

</figure>

</div>
"""
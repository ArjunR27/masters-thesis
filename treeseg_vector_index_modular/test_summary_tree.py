import argparse
import sys
from numbers import Integral
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from treeseg_vector_index_modular.constants import PROJECT_DIR
from treeseg_vector_index_modular.lecture_catalog import LectureCatalog
from treeseg_vector_index_modular.lecture_segment_builder import LectureSegmentBuilder
from treeseg_vector_index_modular.lpm_config_builder import LpmConfigBuilder


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build and inspect a TreeSeg summary tree for one lecture."
    )
    parser.add_argument(
        "--list-lectures",
        action="store_true",
        help="List available lectures and exit (unless --lecture is also provided).",
    )
    parser.add_argument(
        "--lecture",
        default=None,
        help="Lecture key or list index to process.",
    )
    parser.add_argument(
        "--leaves-only",
        action="store_true",
        help="Only print leaf nodes.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Maximum number of nodes to print.",
    )
    parser.add_argument(
        "--max-summary-chars",
        type=int,
        default=280,
        help="Maximum characters to show per node summary; <=0 disables truncation.",
    )
    return parser.parse_args(argv)

def print_separator(char="-", width=72):
    print(char * width)


def _node_depth(node):
    depth = getattr(node, "depth", None)
    if isinstance(depth, Integral) and depth >= 0:
        return int(depth)
    return 0


def _format_seconds(value):
    if isinstance(value, (int, float)):
        return f"{value:.2f}s"
    return "N/A"


def _format_summary(summary, max_summary_chars=None):
    text = "<blank>" if summary is None else str(summary).strip()
    if not text:
        text = "<blank>"

    if max_summary_chars is not None and max_summary_chars > 0 and len(text) > max_summary_chars:
        return text[: max_summary_chars - 3].rstrip() + "..."
    return text


def print_node(node, max_summary_chars=None):
    kind = "LEAF    " if getattr(node, "is_leaf", False) else "INTERNAL"

    depth = _node_depth(node)
    depth_indent = " " * depth
    is_leaf = bool(getattr(node, "is_leaf", False))
    content_label = "text" if is_leaf else "summary"
    content = getattr(node, "raw_text", None) if is_leaf else getattr(node, "summary", None)
    formatted_content = _format_summary(content, max_summary_chars=max_summary_chars)
    start = _format_seconds(getattr(node, "start", None))
    end = _format_seconds(getattr(node, "end", None))
    segment = getattr(node, "segment", []) or []
    embedding = getattr(node, "embedding", None)

    print(f"{depth_indent}[{kind}] id={getattr(node, 'identifier', '<unknown>')}  depth={depth}")
    print(
        f"{depth_indent}         start={start}  end={end}  "
        f"utterances={len(segment)}"
    )
    print(f"{depth_indent}         embedding={'yes' if embedding is not None else 'MISSING'}")
    print(f"{depth_indent}         {content_label}: {formatted_content}")
    print()


def print_tree_stats(all_nodes):
    """Print aggregate statistics about the built tree."""
    leaves = [n for n in all_nodes if getattr(n, "is_leaf", False)]
    internals = [n for n in all_nodes if not getattr(n, "is_leaf", False)]
    max_depth = max((_node_depth(n) for n in all_nodes), default=0)
    missing_depth = [n for n in all_nodes if getattr(n, "depth", None) is None]
    missing_summary = [
        n
        for n in internals
        if not str(getattr(n, "summary", "")).strip()
        or str(getattr(n, "summary", "")).strip() == "<blank>"
    ]
    missing_embedding_internal = [
        n for n in internals if getattr(n, "embedding", None) is None
    ]

    print_separator("=")
    print("TREE STATISTICS")
    print_separator()
    print(f"  Total nodes      : {len(all_nodes)}")
    print(f"  Leaf nodes       : {len(leaves)}")
    print(f"  Internal nodes   : {len(internals)}")
    print(f"  Max depth        : {max_depth}")
    print(f"  Missing depth    : {len(missing_depth)}")
    print(f"  Missing summary  : {len(missing_summary)}")
    print(f"  Missing embedding: {len(missing_embedding_internal)} (internal nodes)")
    print_separator("=")
    print()



def main():
    args = parse_args()

    data_dir = PROJECT_DIR / "lpm_data"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Discover lectures ──────────────────────────────────────────────────────
    lectures = LectureCatalog.discover_lectures(data_dir=data_dir)
    if not lectures:
        print("No lectures found. Check the data directory.")
        return

    if args.list_lectures:
        print(LectureCatalog.format_lecture_list(lectures))
        if args.lecture is None:
            return

    if args.lecture:
        try:
            lecture_key = LectureCatalog.resolve_lecture_choice(
                lectures, args.lecture, allow_all=False
            )
        except ValueError as exc:
            print(str(exc))
            return
    else:
        print("Available lectures:")
        print(LectureCatalog.format_lecture_list(lectures))
        while True:
            choice = input("lecture> ").strip()
            if not choice:
                continue
            try:
                lecture_key = LectureCatalog.resolve_lecture_choice(
                    lectures, choice, allow_all=False
                )
                break
            except ValueError as exc:
                print(str(exc))

    lecture = next(lec for lec in lectures if lec.key == lecture_key)
    print(f"\nBuilding summary tree for: {lecture_key}\n")

    # ── Build config and load utterances ──────────────────────────────────────
    treeseg_config = LpmConfigBuilder.build_lpm_config(embedding_model=embedding_model)

    utterances = LectureSegmentBuilder.load_lecture_utterances(
        lecture,
    )
    print(f"Loaded {len(utterances)} utterances.\n")

    # ── Build the summary tree ─────────────────────────────────────────────────
    print("Running TreeSeg + DFS summarisation (this may take a while)...\n")
    root, all_nodes = LectureSegmentBuilder.build_summary_tree_for_lecture(
        lecture=lecture,
        utterances=utterances,
        treeseg_config=treeseg_config,
        target_segments=None,
        include_ocr=False,
    )

    if root is None:
        print("ERROR: root is None. Tree could not be built.")
        return

    # ── Stats ──────────────────────────────────────────────────────────────────
    print_tree_stats(all_nodes)

    # ── Print nodes ───────────────────────────────────────────────────────────
    nodes_to_print = all_nodes
    if args.leaves_only:
        nodes_to_print = [n for n in all_nodes if getattr(n, "is_leaf", False)]

    # all_nodes is post-order (leaves first, root last).
    # Sort by depth then identifier for a more readable top-down view.
    nodes_to_print = sorted(
        nodes_to_print,
        key=lambda n: (_node_depth(n), str(getattr(n, "identifier", ""))),
    )

    if args.max_nodes is not None:
        nodes_to_print = nodes_to_print[: args.max_nodes]

    print(f"Showing {len(nodes_to_print)} node(s):\n")
    print_separator("=")
    for node in nodes_to_print:
        print_node(node, max_summary_chars=args.max_summary_chars)

    # ── Root summary ──────────────────────────────────────────────────────────
    print_separator("=")
    if getattr(root, "is_leaf", False):
        print("ROOT NODE TEXT (full):")
        root_content = getattr(root, "raw_text", None)
    else:
        print("ROOT NODE SUMMARY (full):")
        root_content = getattr(root, "summary", None)
    print_separator()
    print(_format_summary(root_content, max_summary_chars=None))
    print_separator("=")


if __name__ == "__main__":
    main()

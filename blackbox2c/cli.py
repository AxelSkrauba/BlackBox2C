"""
Command-line interface for BlackBox2C.

Usage:
    blackbox2c convert  --help
    blackbox2c analyze  --help
    blackbox2c export   --help
"""

import argparse
import sys
import os
import pickle


def _load_model(path: str):
    """Load a pickled scikit-learn model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_data(path: str):
    """Load a NumPy .npy or .npz array from disk."""
    import numpy as np
    if path.endswith(".npz"):
        data = np.load(path)
        keys = list(data.keys())
        if len(keys) == 1:
            return data[keys[0]]
        raise ValueError(
            f"NPZ file contains multiple arrays ({keys}). "
            "Save X_train as a single-array .npy file."
        )
    return np.load(path)


def cmd_convert(args):
    """Handle 'blackbox2c convert' sub-command."""
    import numpy as np
    from blackbox2c import convert, ConversionConfig

    print(f"Loading model from: {args.model}")
    model = _load_model(args.model)

    print(f"Loading training data from: {args.data}")
    X_train = _load_data(args.data)

    X_test = None
    if args.test_data:
        print(f"Loading test data from: {args.test_data}")
        X_test = _load_data(args.test_data)

    feature_names = None
    if args.feature_names:
        feature_names = [n.strip() for n in args.feature_names.split(",")]

    class_names = None
    if args.class_names:
        class_names = [n.strip() for n in args.class_names.split(",")]

    config = ConversionConfig(
        max_depth=args.max_depth,
        optimize_rules=args.optimize,
        use_fixed_point=args.fixed_point,
        precision=args.precision,
        function_name=args.function_name,
        n_samples=args.n_samples,
    )

    code = convert(
        model,
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        X_test=X_test,
        target=args.target,
        config=config,
    )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"\n[OK] Code written to: {args.output}")
    else:
        print("\n--- Generated Code ---")
        print(code)


def cmd_analyze(args):
    """Handle 'blackbox2c analyze' sub-command."""
    from blackbox2c.analysis import FeatureSensitivityAnalyzer

    print(f"Loading model from: {args.model}")
    model = _load_model(args.model)

    print(f"Loading data from: {args.data}")
    X = _load_data(args.data)

    import numpy as np
    y = model.predict(X)

    feature_names = None
    if args.feature_names:
        feature_names = [n.strip() for n in args.feature_names.split(",")]

    analyzer = FeatureSensitivityAnalyzer(
        n_repeats=args.n_repeats,
        random_state=42,
    )
    results = analyzer.analyze(model, X, y, feature_names=feature_names)

    print(results.summary())

    if args.top_n:
        top = results.get_top_features(args.top_n)
        print(f"\nTop {args.top_n} features:")
        for idx, name, imp in top:
            print(f"  [{idx:2d}] {name:<30s}  importance={imp:.4f}")

    if args.output:
        summary = results.summary()
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\n[OK] Analysis report written to: {args.output}")


def cmd_export(args):
    """Handle 'blackbox2c export' sub-command."""
    from blackbox2c.exporters import create_exporter
    import pickle

    print(f"Loading model from: {args.model}")
    model = _load_model(args.model)

    feature_names = None
    if args.feature_names:
        feature_names = [n.strip() for n in args.feature_names.split(",")]

    class_names = None
    if args.class_names:
        class_names = [n.strip() for n in args.class_names.split(",")]

    exporter = create_exporter(
        args.format,
        function_name=args.function_name,
    )
    code = exporter.generate(
        model,
        feature_names=feature_names,
        class_names=class_names,
    )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"[OK] Code written to: {args.output}")
    else:
        print(code)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blackbox2c",
        description=(
            "BlackBox2C — Convert scikit-learn models to optimized embedded code "
            "(C, C++, Arduino, MicroPython)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # ── convert ──────────────────────────────────────────────────────────────
    p_conv = subparsers.add_parser(
        "convert",
        help="Convert a trained model to embedded code.",
        description=(
            "Convert a pickled scikit-learn model to C, C++, Arduino, "
            "or MicroPython code."
        ),
    )
    p_conv.add_argument("model", help="Path to pickled model (.pkl)")
    p_conv.add_argument("data", help="Path to training data (.npy or .npz)")
    p_conv.add_argument(
        "-t", "--target",
        default="c",
        choices=["c", "cpp", "arduino", "micropython"],
        help="Output format (default: c)",
    )
    p_conv.add_argument("-o", "--output", help="Output file path (default: stdout)")
    p_conv.add_argument("--test-data", help="Path to test data for fidelity evaluation")
    p_conv.add_argument(
        "--feature-names",
        help="Comma-separated feature names, e.g. 'temp,humidity,pressure'",
    )
    p_conv.add_argument(
        "--class-names",
        help="Comma-separated class names, e.g. 'LOW,MEDIUM,HIGH'",
    )
    p_conv.add_argument(
        "--max-depth", type=int, default=5, help="Max surrogate tree depth (default: 5)"
    )
    p_conv.add_argument(
        "--optimize",
        default="medium",
        choices=["low", "medium", "high"],
        help="Rule optimization level (default: medium)",
    )
    p_conv.add_argument(
        "--fixed-point", action="store_true", help="Use fixed-point arithmetic"
    )
    p_conv.add_argument(
        "--precision",
        type=int,
        default=8,
        choices=[8, 16, 32],
        help="Bit precision for fixed-point (default: 8)",
    )
    p_conv.add_argument(
        "--function-name", default="predict", help="Name of the generated function"
    )
    p_conv.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Samples for boundary synthesis (default: 10000)",
    )
    p_conv.set_defaults(func=cmd_convert)

    # ── analyze ───────────────────────────────────────────────────────────────
    p_ana = subparsers.add_parser(
        "analyze",
        help="Analyze feature sensitivity of a trained model.",
        description="Run permutation importance analysis to identify key features.",
    )
    p_ana.add_argument("model", help="Path to pickled model (.pkl)")
    p_ana.add_argument("data", help="Path to data (.npy or .npz)")
    p_ana.add_argument(
        "--feature-names",
        help="Comma-separated feature names",
    )
    p_ana.add_argument(
        "--top-n", type=int, default=None, help="Print top N most important features"
    )
    p_ana.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Permutation repeats for importance estimation (default: 10)",
    )
    p_ana.add_argument("-o", "--output", help="Write report to file instead of stdout")
    p_ana.set_defaults(func=cmd_analyze)

    # ── export ────────────────────────────────────────────────────────────────
    p_exp = subparsers.add_parser(
        "export",
        help="Export a decision tree model directly (no surrogate extraction).",
        description=(
            "Export a decision tree model directly to a target format. "
            "For non-tree models, use 'convert' which performs surrogate extraction."
        ),
    )
    p_exp.add_argument("model", help="Path to pickled DecisionTree model (.pkl)")
    p_exp.add_argument(
        "-f", "--format",
        default="cpp",
        choices=["cpp", "arduino", "micropython"],
        help="Output format (default: cpp)",
    )
    p_exp.add_argument("-o", "--output", help="Output file path (default: stdout)")
    p_exp.add_argument(
        "--feature-names", help="Comma-separated feature names"
    )
    p_exp.add_argument(
        "--class-names", help="Comma-separated class names"
    )
    p_exp.add_argument(
        "--function-name", default="predict", help="Name of the generated function"
    )
    p_exp.set_defaults(func=cmd_export)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

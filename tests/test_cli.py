"""
Tests for the blackbox2c CLI.
"""

import os
import pickle
import tempfile

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_regression

from blackbox2c.cli import build_parser, cmd_convert, cmd_analyze, cmd_export


@pytest.fixture
def clf_model_file(tmp_path):
    """Persisted classification model + data."""
    iris = load_iris()
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(iris.data, iris.target)
    model_path = tmp_path / "model.pkl"
    data_path = tmp_path / "X.npy"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    np.save(str(data_path), iris.data)
    return str(model_path), str(data_path)


@pytest.fixture
def reg_model_file(tmp_path):
    """Persisted regression model + data."""
    X, y = make_regression(n_samples=100, n_features=3, random_state=42)
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X, y)
    model_path = tmp_path / "reg_model.pkl"
    data_path = tmp_path / "X_reg.npy"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    np.save(str(data_path), X)
    return str(model_path), str(data_path)


class TestCLIParser:
    def test_parser_created(self):
        parser = build_parser()
        assert parser is not None

    def test_convert_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["convert", "model.pkl", "data.npy"])
        assert args.command == "convert"
        assert args.target == "c"
        assert args.max_depth == 5

    def test_convert_all_targets(self):
        parser = build_parser()
        for tgt in ["c", "cpp", "arduino", "micropython"]:
            args = parser.parse_args(["convert", "m.pkl", "d.npy", "-t", tgt])
            assert args.target == tgt

    def test_analyze_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["analyze", "model.pkl", "data.npy"])
        assert args.command == "analyze"
        assert args.n_repeats == 10

    def test_export_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["export", "model.pkl", "-f", "arduino"])
        assert args.command == "export"
        assert args.format == "arduino"

    def test_version_flag(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_invalid_target_raises(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["convert", "m.pkl", "d.npy", "-t", "invalid"])


class TestCLIConvert:
    def test_convert_to_c(self, clf_model_file, tmp_path):
        model_path, data_path = clf_model_file
        output_path = str(tmp_path / "out.c")
        parser = build_parser()
        args = parser.parse_args([
            "convert", model_path, data_path,
            "-t", "c",
            "-o", output_path,
            "--max-depth", "4",
        ])
        cmd_convert(args)
        assert os.path.exists(output_path)
        content = open(output_path).read()
        assert "uint8_t predict" in content

    def test_convert_to_cpp(self, clf_model_file, tmp_path):
        model_path, data_path = clf_model_file
        output_path = str(tmp_path / "out.hpp")
        parser = build_parser()
        args = parser.parse_args([
            "convert", model_path, data_path,
            "-t", "cpp", "-o", output_path,
        ])
        cmd_convert(args)
        assert os.path.exists(output_path)
        content = open(output_path).read()
        assert "class Predictor" in content

    def test_convert_to_arduino(self, clf_model_file, tmp_path):
        model_path, data_path = clf_model_file
        output_path = str(tmp_path / "out.h")
        parser = build_parser()
        args = parser.parse_args([
            "convert", model_path, data_path,
            "-t", "arduino", "-o", output_path,
        ])
        cmd_convert(args)
        assert os.path.exists(output_path)
        content = open(output_path).read()
        assert "#include <Arduino.h>" in content

    def test_convert_to_micropython(self, clf_model_file, tmp_path):
        model_path, data_path = clf_model_file
        output_path = str(tmp_path / "out.py")
        parser = build_parser()
        args = parser.parse_args([
            "convert", model_path, data_path,
            "-t", "micropython", "-o", output_path,
        ])
        cmd_convert(args)
        assert os.path.exists(output_path)
        content = open(output_path).read()
        assert "class Predictor:" in content

    def test_convert_with_feature_and_class_names(self, clf_model_file, tmp_path):
        model_path, data_path = clf_model_file
        output_path = str(tmp_path / "named.c")
        parser = build_parser()
        args = parser.parse_args([
            "convert", model_path, data_path,
            "-o", output_path,
            "--feature-names", "sepal_l,sepal_w,petal_l,petal_w",
            "--class-names", "setosa,versicolor,virginica",
        ])
        cmd_convert(args)
        content = open(output_path).read()
        assert "setosa" in content or "CLASS_0" in content

    def test_convert_regression(self, reg_model_file, tmp_path):
        model_path, data_path = reg_model_file
        output_path = str(tmp_path / "reg.c")
        parser = build_parser()
        args = parser.parse_args([
            "convert", model_path, data_path,
            "-o", output_path,
        ])
        cmd_convert(args)
        content = open(output_path).read()
        assert "float predict" in content

    def test_convert_missing_model_raises(self, tmp_path):
        parser = build_parser()
        args = parser.parse_args([
            "convert", "nonexistent.pkl", "data.npy",
        ])
        with pytest.raises(FileNotFoundError):
            cmd_convert(args)


class TestCLIAnalyze:
    def test_analyze_basic(self, clf_model_file, tmp_path):
        model_path, data_path = clf_model_file
        output_path = str(tmp_path / "report.txt")
        parser = build_parser()
        args = parser.parse_args([
            "analyze", model_path, data_path,
            "--n-repeats", "3",
            "--top-n", "2",
            "-o", output_path,
        ])
        cmd_analyze(args)
        assert os.path.exists(output_path)

    def test_analyze_with_feature_names(self, clf_model_file):
        model_path, data_path = clf_model_file
        parser = build_parser()
        args = parser.parse_args([
            "analyze", model_path, data_path,
            "--feature-names", "sepal_l,sepal_w,petal_l,petal_w",
            "--n-repeats", "2",
        ])
        cmd_analyze(args)


class TestCLIExport:
    def test_export_cpp(self, clf_model_file, tmp_path):
        model_path, _ = clf_model_file
        output_path = str(tmp_path / "out.hpp")
        parser = build_parser()
        args = parser.parse_args([
            "export", model_path,
            "-f", "cpp",
            "-o", output_path,
            "--feature-names", "sl,sw,pl,pw",
            "--class-names", "setosa,versicolor,virginica",
        ])
        cmd_export(args)
        assert os.path.exists(output_path)
        content = open(output_path).read()
        assert "class Predictor" in content

    def test_export_arduino(self, clf_model_file, tmp_path):
        model_path, _ = clf_model_file
        output_path = str(tmp_path / "out.h")
        parser = build_parser()
        args = parser.parse_args([
            "export", model_path, "-f", "arduino", "-o", output_path,
        ])
        cmd_export(args)
        assert os.path.exists(output_path)

    def test_export_micropython(self, clf_model_file, tmp_path):
        model_path, _ = clf_model_file
        output_path = str(tmp_path / "model.py")
        parser = build_parser()
        args = parser.parse_args([
            "export", model_path, "-f", "micropython", "-o", output_path,
        ])
        cmd_export(args)
        assert os.path.exists(output_path)

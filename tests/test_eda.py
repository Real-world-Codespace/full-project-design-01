from cooling_load.config import load_config
from cooling_load.eda import generate_eda_report
from cooling_load.synthetic import generate_synthetic_data


def test_eda_report_persists_all_expected_artifacts(tmp_path):
    config = load_config("configs/base.yaml")
    frame = generate_synthetic_data(periods=72, buildings=("A", "B"))
    report = generate_eda_report(
        frame,
        config.data,
        output_dir=tmp_path / "figures",
        summary_path=tmp_path / "eda_summary.json",
    )
    assert len(report.figures) == 7
    assert all(path.exists() and path.stat().st_size > 0 for path in report.figures)
    assert report.summary_path.exists()

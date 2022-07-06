import pandas as pd
import pytest

from deepchecks.tabular.checks import train_test_validation as ttv
from deepchecks.tabular import Dataset

from deepchecks.tabular.checks import 

def test_category_mismatch_train_test(
    train_dataset: Dataset,
    test_dataset: Dataset,
    columns: Union[Hashable, List[Hashable], None] = None,
    ignore_columns: Union[Hashable, List[Hashable], None] = None,
    max_features_to_show: int = 5,
    max_new_categories_to_show: int = 5,
):
    check = ttv.CategoryMismatchTrainTest(
        columns, ignore_columns, max_features_to_show, max_features_to_show
    )
    assert check.run(train_dataset, test_dataset).passed_conditions()


def test_datasets_size_comparison(train_dataset: Dataset, test_dataset: Dataset):
    check = ttv.DatasetsSizeComparison()
    check.add_condition_test_train_size_ratio_greater_than(0.2)
    assert check.run(train_dataset, test_dataset).passed_conditions()


def test_date_train_test_leakage_duplicates(
    train_dataset: Dataset, test_dataset: Dataset
):
    check = ttv.DateTrainTestLeakageDuplicates()
    check.add_condition_leakage_ratio_less_or_equal(0.0)
    assert check.run(train_dataset, test_dataset).passed_conditions()


def test_date_train_test_leakage_overlap(train_dataset: Dataset, test_dataset: Dataset):
    check = ttv.DateTrainTestLeakageOverlap()
    assert check.run(train_dataset, test_dataset).passed_conditions()


def test_dominant_frequency_change(train_dataset: Dataset, test_dataset: Dataset):
    check = ttv.DominantFrequencyChange()
    assert check.run(train_dataset, test_dataset).passed_conditions()


def test_feature_label_correlation_change(
    train_dataset: Dataset, test_dataset: Dataset
):
    check = ttv.FeatureLabelCorrelationChange()
    assert check.run(train_dataset, test_dataset).passed_conditions()


def test_identifier_leakage(train_dataset: Dataset):
    check = ttv.IdentifierLeakage()
    assert check.run(train_dataset).passed_conditions()


def test_index_leakage(train_dataset: Dataset, test_dataset: Dataset):
    check = ttv.IndexTrainTestLeakage()
    assert check.run(train_dataset, test_dataset).passed_conditions()


def test_new_label_train_test(train_dataset: Dataset, test_dataset: Dataset):
    check = ttv.NewLabelTrainTest()
    assert check.run(train_dataset, test_dataset).passed_conditions()


def test_string_mismatch_comparison(train_dataset: Dataset, test_dataset: Dataset):
    check = ttv.StringMismatchComparison()
    assert check.run(train_dataset, test_dataset).passed_conditions()


def test_train_test_feature_drift(train_dataset: Dataset, test_dataset: Dataset):
    check = ttv.TrainTestFeatureDrift()
    assert check.run(train_dataset, test_dataset).passed_conditions()


def test_train_test_label_drift(train_dataset: Dataset, test_dataset: Dataset):
    check = ttv.TrainTestLabelDrift()()
    assert check.run(train_dataset, test_dataset).passed_conditions()


def test_train_test_samples_mix(train_dataset: Dataset, test_dataset: Dataset):
    check = ttv.TrainTestSamplesMix()()
    assert check.run(train_dataset, test_dataset).passed_conditions()


def test_whole_dataset_drift(train_dataset: Dataset, test_dataset: Dataset):
    check = ttv.WholeDatasetDrift()
    assert check.run(train_dataset, test_dataset).passed_conditions()

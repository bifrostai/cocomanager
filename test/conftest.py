import pytest


def pytest_collection_modifyitems(items: list[pytest.Function]):
    """Order test files in a specific order"""
    file_order = ["test_parse"]  # NOTE: Add the test files in the order to run

    file_mapping = {item: item.path.stem for item in items}
    # add items that are not in file_order to the end
    unique_files = {file_mapping[item] for item in items}
    file_order.extend(sorted(list(unique_files - set(file_order))))

    sorted_items = items.copy()
    sorted_items.sort(key=lambda item: file_order.index(file_mapping[item]))
    items[:] = sorted_items

# 2YP Test Suite

Automated test suite for the 2YP research-front monitoring pipeline.

## Test Organization

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Shared pytest fixtures
├── test_transform.py           # Tests for DataFrame transformations
├── test_validate.py            # Tests for schema validation
├── test_slicing.py             # Tests for temporal slicing
├── test_logging_config.py      # Tests for logging framework
└── test_settings.py            # Tests for settings management
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_transform.py
```

### Run Specific Test Class

```bash
pytest tests/test_transform.py::TestAddTimeVars
```

### Run Specific Test Function

```bash
pytest tests/test_transform.py::TestAddTimeVars::test_adds_pub_year_column
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Verbose Output

```bash
pytest -v
```

### Show Print Statements

```bash
pytest -s
```

### Stop on First Failure

```bash
pytest -x
```

### Run Last Failed Tests

```bash
pytest --lf
```

## Test Coverage

To generate coverage reports:

```bash
# Terminal report
pytest --cov=src --cov-report=term

# HTML report
pytest --cov=src --cov-report=html

# View HTML report
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Fast unit tests for individual functions
- `@pytest.mark.integration` - Integration tests for module interactions
- `@pytest.mark.slow` - Tests that take significant time to run
- `@pytest.mark.requires_api` - Tests requiring external API access

## Writing New Tests

### Test File Naming

- Test files must start with `test_` or end with `_test.py`
- Place in `tests/` directory

### Test Function Naming

- Test functions must start with `test_`
- Use descriptive names: `test_function_name_does_something`

### Using Fixtures

Common fixtures are defined in `conftest.py`:

- `temp_dir` - Temporary directory for test files
- `sample_works_df` - Sample DataFrame with OpenAlex data
- `sample_schema_yaml` - Sample schema configuration
- `sample_slices_yaml` - Sample slicing configuration
- `sample_settings_json` - Sample settings file

Example:

```python
def test_my_function(sample_works_df):
    result = my_function(sample_works_df)
    assert len(result) > 0
```

### Test Structure

Follow the Arrange-Act-Assert pattern:

```python
def test_feature():
    # Arrange: Set up test data
    df = pd.DataFrame({"col": [1, 2, 3]})

    # Act: Execute the function
    result = process_data(df)

    # Assert: Verify results
    assert len(result) == 3
    assert result["col"].sum() == 6
```

## Continuous Integration

Tests should pass before committing changes. Run the full test suite:

```bash
pytest -v --cov=src --cov-report=term-missing
```

This ensures:

- All tests pass
- Code coverage is maintained
- No regressions introduced

## Test Development Tips

1. **Keep tests independent** - Each test should run in isolation
2. **Use fixtures for setup** - Avoid repetitive setup code
3. **Test edge cases** - Empty inputs, None values, invalid data
4. **Test error conditions** - Use `pytest.raises()` for expected errors
5. **Keep tests fast** - Mark slow tests with `@pytest.mark.slow`
6. **Use descriptive assertions** - Make failures easy to debug

## Example Test

```python
import pytest
import pandas as pd
from src.transform import add_time_vars

@pytest.mark.unit
class TestAddTimeVars:
    def test_adds_pub_year_column(self, sample_works_df):
        """Test that pub_year column is added correctly."""
        result = add_time_vars(sample_works_df)

        assert "pub_year" in result.columns
        assert result["pub_year"].dtype == "int64"
```

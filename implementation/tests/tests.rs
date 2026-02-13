use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::config::ConfigOptions;
use datafusion_common::ScalarValue;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs};

use datafusion_regexp_extract::regexp_extract_udf;

fn field(name: &str, data_type: DataType, nullable: bool) -> FieldRef {
    Arc::new(Field::new(name, data_type, nullable))
}

fn invoke_raw(
    args: Vec<ColumnarValue>,
    arg_fields: Vec<FieldRef>,
    number_rows: usize,
) -> datafusion_common::Result<ColumnarValue> {
    let udf = regexp_extract_udf();
    let return_field = field("x", DataType::Utf8, true);

    udf.invoke_with_args(ScalarFunctionArgs {
        args,
        arg_fields,
        number_rows,
        return_field,
        config_options: Arc::new(ConfigOptions::new()),
    })
}

fn regexp_extract(
    s: ColumnarValue,
    pattern: &str,
    idx: i64,
) -> datafusion_common::Result<ColumnarValue> {
    let number_rows = match &s {
        ColumnarValue::Scalar(_) => 1,
        ColumnarValue::Array(arr) => arr.len(),
    };
    let arg_fields = vec![
        field("s", DataType::Utf8, true),
        field("pattern", DataType::Utf8, true),
        field("idx", DataType::Int64, true),
    ];
    let args = vec![
        s,
        ColumnarValue::Scalar(ScalarValue::Utf8(Some(pattern.to_string()))),
        ColumnarValue::Scalar(ScalarValue::Int64(Some(idx))),
    ];
    invoke_raw(args, arg_fields, number_rows)
}

#[test]
fn regexp_extract_scalar_literals() -> datafusion_common::Result<()> {
    let s: ArrayRef = Arc::new(StringArray::from(vec![Some("abc-123")]));
    let out = regexp_extract(
        ColumnarValue::Array(s),
        "([a-z]+)-(\\d+)",
        2,
    )?;

    match out {
        ColumnarValue::Array(arr) => {
            let arr = arr
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("regexp_extract should return Utf8 array");
            assert_eq!(arr.value(0), "123");
        }
        other => panic!("unexpected output: {other:?}"),
    }
    Ok(())
}

#[test]
fn regexp_extract_idx_zero_returns_whole_match() -> datafusion_common::Result<()> {
    let s: ArrayRef = Arc::new(StringArray::from(vec![Some("xxabcdyy")]));
    let out = regexp_extract(
        ColumnarValue::Array(s),
        "(ab)(cd)",
        0,
    )?;

    match out {
        ColumnarValue::Array(arr) => {
            let arr = arr
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("regexp_extract should return Utf8 array");
            assert_eq!(arr.value(0), "abcd");
        }
        other => panic!("unexpected output: {other:?}"),
    }
    Ok(())
}

#[test]
fn regexp_extract_array_values_with_no_match() -> datafusion_common::Result<()> {
    let s: ArrayRef = Arc::new(StringArray::from(vec![
        Some("xxabcdyy"),
        Some("no match"),
        Some("abc-123"),
    ]));

    let out = regexp_extract(ColumnarValue::Array(s), "(ab)(cd)", 1)?;

    let arr = match out {
        ColumnarValue::Array(arr) => arr,
        other => panic!("unexpected output: {other:?}"),
    };
    let arr = arr
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("regexp_extract should return Utf8 array");

    let actual = (0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i).to_string())
            }
        })
        .collect::<Vec<_>>();

    assert_eq!(
        actual,
        vec![
            Some("ab".to_string()),
            Some("".to_string()),
            Some("".to_string())
        ]
    );
    Ok(())
}

#[test]
fn regexp_extract_errors_on_null_pattern() {
    let err = invoke_raw(
        vec![
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("abc".to_string()))),
            ColumnarValue::Scalar(ScalarValue::Utf8(None)),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(1))),
        ],
        vec![
            field("s", DataType::Utf8, true),
            field("pattern", DataType::Utf8, true),
            field("idx", DataType::Int64, true),
        ],
        1,
    )
    .expect_err("null pattern should fail");
    assert!(err.to_string().contains("pattern must not be NULL"));
}

#[test]
fn regexp_extract_errors_on_null_index() {
    let err = invoke_raw(
        vec![
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("abc".to_string()))),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("(a)".to_string()))),
            ColumnarValue::Scalar(ScalarValue::Int64(None)),
        ],
        vec![
            field("s", DataType::Utf8, true),
            field("pattern", DataType::Utf8, true),
            field("idx", DataType::Int64, true),
        ],
        1,
    )
    .expect_err("null idx should fail");
    assert!(err.to_string().contains("idx must not be NULL"));
}

#[test]
fn regexp_extract_errors_on_negative_index() {
    let err = invoke_raw(
        vec![
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("abc".to_string()))),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("(a)".to_string()))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(-1))),
        ],
        vec![
            field("s", DataType::Utf8, true),
            field("pattern", DataType::Utf8, true),
            field("idx", DataType::Int64, true),
        ],
        1,
    )
    .expect_err("negative index should fail");

    assert!(err.to_string().contains("group index must be non-negative"));
}

#[test]
fn regexp_extract_errors_on_out_of_range_group() {
    let s: ArrayRef = Arc::new(StringArray::from(vec![Some("abc")]));
    let err = regexp_extract(
        ColumnarValue::Array(s),
        "(a)",
        2,
    )
    .expect_err("out-of-range group index should fail");

    assert!(err.to_string().contains("out of range"));
}

#[test]
fn regexp_extract_errors_on_wrong_arity() {
    let err = invoke_raw(
        vec![
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("abc".to_string()))),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("(a)".to_string()))),
        ],
        vec![
            field("s", DataType::Utf8, true),
            field("pattern", DataType::Utf8, true),
        ],
        1,
    )
    .expect_err("wrong number of arguments should fail");

    assert!(err.to_string().contains("expects 3 arguments"));
}

#[test]
fn regexp_extract_errors_when_pattern_is_non_utf8_scalar() {
    let err = invoke_raw(
        vec![
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("abc".to_string()))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(42))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(1))),
        ],
        vec![
            field("s", DataType::Utf8, true),
            field("pattern", DataType::Utf8, true),
            field("idx", DataType::Int64, true),
        ],
        1,
    )
    .expect_err("non-utf8 pattern should fail");

    assert!(err.to_string().contains("pattern must be Utf8 scalar"));
}

#[test]
fn regexp_extract_errors_when_pattern_is_array() {
    let pattern_arr: ArrayRef = Arc::new(StringArray::from(vec![Some("(a)")]));
    let err = invoke_raw(
        vec![
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("abc".to_string()))),
            ColumnarValue::Array(pattern_arr),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(1))),
        ],
        vec![
            field("s", DataType::Utf8, true),
            field("pattern", DataType::Utf8, true),
            field("idx", DataType::Int64, true),
        ],
        1,
    )
    .expect_err("pattern array should fail");

    assert!(err.to_string().contains("pattern must be a scalar"));
}

#[test]
fn regexp_extract_errors_when_idx_is_non_int64_scalar() {
    let err = invoke_raw(
        vec![
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("abc".to_string()))),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("(a)".to_string()))),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("1".to_string()))),
        ],
        vec![
            field("s", DataType::Utf8, true),
            field("pattern", DataType::Utf8, true),
            field("idx", DataType::Int64, true),
        ],
        1,
    )
    .expect_err("non-int64 idx should fail");

    assert!(err.to_string().contains("idx must be Int64 scalar"));
}

#[test]
fn regexp_extract_errors_when_idx_is_array() {
    let idx_arr: ArrayRef = Arc::new(Int32Array::from(vec![1]));
    let err = invoke_raw(
        vec![
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("abc".to_string()))),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("(a)".to_string()))),
            ColumnarValue::Array(idx_arr),
        ],
        vec![
            field("s", DataType::Utf8, true),
            field("pattern", DataType::Utf8, true),
            field("idx", DataType::Int64, true),
        ],
        1,
    )
    .expect_err("idx array should fail");

    assert!(err.to_string().contains("idx must be a scalar"));
}

#[test]
fn regexp_extract_errors_on_invalid_regex_pattern() {
    let err = regexp_extract(
        ColumnarValue::Scalar(ScalarValue::Utf8(Some("abc".to_string()))),
        "(",
        1,
    )
    .expect_err("invalid regex should fail");

    assert!(err.to_string().contains("Invalid regex pattern"));
}

#[test]
fn regexp_extract_errors_when_scalar_str_is_non_utf8() {
    let err = invoke_raw(
        vec![
            ColumnarValue::Scalar(ScalarValue::Int64(Some(7))),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("(a)".to_string()))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(1))),
        ],
        vec![
            field("s", DataType::Utf8, true),
            field("pattern", DataType::Utf8, true),
            field("idx", DataType::Int64, true),
        ],
        1,
    )
    .expect_err("non-utf8 string scalar should fail");

    assert!(err.to_string().contains("str must be an array in this implementation"));
}

#[test]
fn regexp_extract_errors_when_scalar_str_is_null() {
    let err = regexp_extract(
        ColumnarValue::Scalar(ScalarValue::Utf8(None)),
        "(a)",
        1,
    )
    .expect_err("null string scalar should fail");

    assert!(err.to_string().contains("str must be an array in this implementation"));
}

#[test]
fn regexp_extract_errors_when_array_str_is_non_utf8() {
    let s_arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
    let err = regexp_extract(ColumnarValue::Array(s_arr), "(a)", 1)
    .expect_err("non-utf8 string array should fail");

    assert!(err.to_string().contains("str must be a Utf8 array"));
}

#[test]
fn regexp_extract_array_str_nulls_are_accepted() {
    let s_arr: ArrayRef = Arc::new(StringArray::from(vec![Some("abc"), None, Some("def")]));
    let out = regexp_extract(ColumnarValue::Array(s_arr), "(a)", 1)
        .expect("null rows in string array should be accepted");

    let arr = match out {
        ColumnarValue::Array(arr) => arr,
        other => panic!("unexpected output: {other:?}"),
    };
    let arr = arr
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("regexp_extract should return Utf8 array");

    let actual = (0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i).to_string())
            }
        })
        .collect::<Vec<_>>();

    assert_eq!(actual, vec![Some("a".to_string()), None, Some("".to_string())]);
}

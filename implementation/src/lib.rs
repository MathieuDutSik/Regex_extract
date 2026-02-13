use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use arrow::array::Array as _;
use arrow::array::{ArrayRef, StringArray};
use arrow::datatypes::DataType;

use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{
    create_udf, ColumnarValue, ScalarUDF, Volatility,
};

use regex::Regex;

#[derive(Default)]
struct RegexCache {
    inner: Mutex<HashMap<String, Regex>>,
}

impl RegexCache {
    fn get(&self, pattern: &str) -> Result<Regex> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| DataFusionError::Execution("Regex cache mutex poisoned".into()))?;

        if let Some(r) = guard.get(pattern) {
            return Ok(r.clone());
        }

        let compiled = Regex::new(pattern).map_err(|e| {
            DataFusionError::Execution(format!("Invalid regex pattern: {pattern}: {e}"))
        })?;

        guard.insert(pattern.to_string(), compiled.clone());
        Ok(compiled)
    }
}

/// Create and return a DataFusion ScalarUDF implementing Spark-like regexp_extract.
pub fn regexp_extract_udf() -> ScalarUDF {
    let cache = Arc::new(RegexCache::default());

    let fun = {
        let cache = cache.clone();

        Arc::new(move |args: &[ColumnarValue]| -> Result<ColumnarValue> {
            if args.len() != 3 {
                return Err(DataFusionError::Execution(
                    "regexp_extract expects 3 arguments: (str, pattern, idx)".into(),
                ));
            }

            let s = &args[0];
            let pattern = &args[1];
            let idx = &args[2];

            // Helper: get scalar string
            let scalar_str = |v: &ColumnarValue, name: &str| -> Result<String> {
                match v {
                    ColumnarValue::Scalar(sv) => match sv {
                        datafusion_common::ScalarValue::Utf8(Some(x)) => Ok(x.clone()),
                        datafusion_common::ScalarValue::Utf8(None) => Err(
                            DataFusionError::Execution(format!("{name} must not be NULL"))
                        ),
                        _ => Err(DataFusionError::Execution(format!(
                            "{name} must be Utf8 scalar"
                        ))),
                    },
                    _ => Err(DataFusionError::Execution(format!(
                        "{name} must be a scalar in this implementation"
                    ))),
                }
            };

            // Helper: get scalar i64
            let scalar_i64 = |v: &ColumnarValue, name: &str| -> Result<i64> {
                match v {
                    ColumnarValue::Scalar(sv) => match sv {
                        datafusion_common::ScalarValue::Int64(Some(x)) => Ok(*x),
                        datafusion_common::ScalarValue::Int64(None) => Err(
                            DataFusionError::Execution(format!("{name} must not be NULL"))
                        ),
                        _ => Err(DataFusionError::Execution(format!(
                            "{name} must be Int64 scalar"
                        ))),
                    },
                    _ => Err(DataFusionError::Execution(format!(
                        "{name} must be a scalar in this implementation"
                    ))),
                }
            };

            let pattern = scalar_str(pattern, "pattern")?;
            let idx = scalar_i64(idx, "idx")?;

            if idx < 0 {
                return Err(DataFusionError::Execution(format!(
                    "regexp_extract group index must be non-negative, got {idx}"
                )));
            }
            let idx_usize: usize = idx as usize;

            let re = cache.get(&pattern)?;

            let arr = match s {
                ColumnarValue::Array(arr) => arr
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        DataFusionError::Execution(
                            "str must be a Utf8 array in this implementation".into(),
                        )
                    })?,
                _ => {
                    return Err(DataFusionError::Execution(
                        "str must be an array in this implementation".into(),
                    ))
                }
            };

            let mut out: Vec<Option<String>> = Vec::with_capacity(arr.len());
            for i in 0..arr.len() {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    let s = arr.value(i);
                    out.push(Some(extract_one(&re, s, idx_usize)?));
                }
            }

            let out_arr: ArrayRef = Arc::new(StringArray::from(out));
            Ok(ColumnarValue::Array(out_arr))
        })
    };

    create_udf(
        "regexp_extract",
        vec![DataType::Utf8, DataType::Utf8, DataType::Int64],
        DataType::Utf8,
        Volatility::Immutable,
        fun,
    )
}

fn extract_one(re: &Regex, s: &str, idx: usize) -> Result<String> {
    if let Some(caps) = re.captures(s) {
        // caps.len() includes group 0 (whole match)
        if idx >= caps.len() {
            return Err(DataFusionError::Execution(format!(
                "regexp_extract group index {idx} out of range; pattern has {} groups",
                caps.len() - 1
            )));
        }
        Ok(caps
            .get(idx)
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| "".to_string()))
    } else {
        Ok("".to_string())
    }
}

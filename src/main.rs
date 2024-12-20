use parking_lot::Mutex;
use rayon::prelude::*;
use std::fmt::Write;
use std::fs::File;
use std::io::BufWriter;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Date64Array, Float32Array, Float64Array,
    Int16Array, Int32Array, Int64Array, Int8Array, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt16Array,
    UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::{DataType, TimeUnit};
use chrono::{DateTime, NaiveDate};
use clap::Parser;
use csv::Writer;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::NamedTempFile;

use anyhow::{Context, Result};
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the parquet file
    #[arg(value_name = "FILE")]
    file: String,

    /// Number of rows to preview (-1 for all)
    #[arg(short, long, default_value = "-1")]
    rows: i64,

    /// Buffer size in KB for CSV writing
    #[arg(short, long, default_value = "64")]
    buffer_size: usize,

    /// Number of threads for parallel processing
    #[arg(short, long, default_value_t = num_cpus::get())]
    threads: usize,
}

#[allow(dead_code)]
fn generate_temp_filename() -> String {
    let random: String = thread_rng()
        .sample_iter(&Alphanumeric)
        .take(10)
        .map(char::from)
        .collect();
    format!("parquet_view_{}.csv", random)
}

fn create_progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} rows ({eta})")
            .unwrap()
            .progress_chars("=>-")
    );
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

struct StringPool {
    buffers: Vec<String>,
}

impl StringPool {
    fn new(capacity: usize) -> Self {
        Self {
            buffers: (0..capacity).map(|_| String::with_capacity(64)).collect(),
        }
    }

    fn get(&mut self) -> String {
        self.buffers
            .pop()
            .unwrap_or_else(|| String::with_capacity(64))
    }

    fn return_string(&mut self, mut s: String) {
        s.clear();
        self.buffers.push(s);
    }
}

fn get_column_value_with_buffer(
    column: &ArrayRef,
    row_idx: usize,
    buffer: &mut String,
) -> Result<()> {
    if column.is_null(row_idx) {
        buffer.push_str("NULL");
        return Ok(());
    }

    match column.data_type() {
        DataType::Utf8 => {
            let array = column
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Failed to cast UTF8 array")?;
            buffer.push_str(array.value(row_idx));
        }

        // Integers
        DataType::Int8 => {
            let array = column
                .as_any()
                .downcast_ref::<Int8Array>()
                .context("Failed to cast Int8 array")?;
            write!(buffer, "{}", array.value(row_idx)).context("Failed to write Int8 value")?;
        }
        DataType::Int16 => {
            let array = column
                .as_any()
                .downcast_ref::<Int16Array>()
                .context("Failed to cast Int16 array")?;
            write!(buffer, "{}", array.value(row_idx)).context("Failed to write Int16 value")?;
        }
        DataType::Int32 => {
            let array = column
                .as_any()
                .downcast_ref::<Int32Array>()
                .context("Failed to cast Int32 array")?;
            write!(buffer, "{}", array.value(row_idx)).context("Failed to write Int32 value")?;
        }
        DataType::Int64 => {
            let array = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .context("Failed to cast Int64 array")?;
            write!(buffer, "{}", array.value(row_idx)).context("Failed to write Int64 value")?;
        }

        // Unsigned Integers
        DataType::UInt8 => {
            let array = column
                .as_any()
                .downcast_ref::<UInt8Array>()
                .context("Failed to cast UInt8 array")?;
            write!(buffer, "{}", array.value(row_idx)).context("Failed to write UInt8 value")?;
        }
        DataType::UInt16 => {
            let array = column
                .as_any()
                .downcast_ref::<UInt16Array>()
                .context("Failed to cast UInt16 array")?;
            write!(buffer, "{}", array.value(row_idx)).context("Failed to write UInt16 value")?;
        }
        DataType::UInt32 => {
            let array = column
                .as_any()
                .downcast_ref::<UInt32Array>()
                .context("Failed to cast UInt32 array")?;
            write!(buffer, "{}", array.value(row_idx)).context("Failed to write UInt32 value")?;
        }
        DataType::UInt64 => {
            let array = column
                .as_any()
                .downcast_ref::<UInt64Array>()
                .context("Failed to cast UInt64 array")?;
            write!(buffer, "{}", array.value(row_idx)).context("Failed to write UInt64 value")?;
        }

        // Floating Point
        DataType::Float32 => {
            let array = column
                .as_any()
                .downcast_ref::<Float32Array>()
                .context("Failed to cast Float32 array")?;
            write!(buffer, "{}", array.value(row_idx)).context("Failed to write Float32 value")?;
        }
        DataType::Float64 => {
            let array = column
                .as_any()
                .downcast_ref::<Float64Array>()
                .context("Failed to cast Float64 array")?;
            write!(buffer, "{}", array.value(row_idx)).context("Failed to write Float64 value")?;
        }

        // Boolean
        DataType::Boolean => {
            let array = column
                .as_any()
                .downcast_ref::<BooleanArray>()
                .context("Failed to cast Boolean array")?;
            write!(buffer, "{}", array.value(row_idx)).context("Failed to write Boolean value")?;
        }

        // Dates
        DataType::Date32 => {
            let array = column
                .as_any()
                .downcast_ref::<Date32Array>()
                .context("Failed to cast Date32 array")?;
            let days = array.value(row_idx);
            if let Some(d) = NaiveDate::from_num_days_from_ce_opt(days + 719163) {
                write!(buffer, "{}", d.format("%Y-%m-%d"))
                    .context("Failed to write Date32 value")?;
            } else {
                buffer.push_str("invalid date");
            }
        }
        DataType::Date64 => {
            let array = column
                .as_any()
                .downcast_ref::<Date64Array>()
                .context("Failed to cast Date64 array")?;
            let milliseconds = array.value(row_idx);
            if let Some(dt) = DateTime::from_timestamp_millis(milliseconds) {
                write!(buffer, "{}", dt.format("%Y-%m-%d"))
                    .context("Failed to write Date64 value")?;
            } else {
                buffer.push_str("invalid date");
            }
        }

        // Timestamps
        DataType::Timestamp(TimeUnit::Second, tz) => {
            let array = column
                .as_any()
                .downcast_ref::<TimestampSecondArray>()
                .context("Failed to cast TimestampSecond array")?;
            let seconds = array.value(row_idx);
            format_timestamp_with_buffer(seconds, 0, tz, buffer)?;
        }
        DataType::Timestamp(TimeUnit::Millisecond, tz) => {
            let array = column
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .context("Failed to cast TimestampMillisecond array")?;
            let ms = array.value(row_idx);
            format_timestamp_with_buffer(ms / 1000, (ms % 1000) * 1_000_000, tz, buffer)?;
        }
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            let array = column
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .context("Failed to cast TimestampMicrosecond array")?;
            let us = array.value(row_idx);
            format_timestamp_with_buffer(us / 1_000_000, (us % 1_000_000) * 1000, tz, buffer)?;
        }
        DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
            let array = column
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .context("Failed to cast TimestampNanosecond array")?;
            let ns = array.value(row_idx);
            format_timestamp_with_buffer(ns / 1_000_000_000, ns % 1_000_000_000, tz, buffer)?;
        }

        // Fallback for other types
        _ => {
            buffer.push_str(&format!("{:?}", column));
        }
    }
    Ok(())
}

fn format_timestamp_with_buffer(
    seconds: i64,
    nanos: i64,
    timezone: &Option<Arc<str>>,
    buffer: &mut String,
) -> Result<()> {
    let dt = DateTime::from_timestamp(seconds, nanos as u32).context("Invalid timestamp")?;

    match timezone {
        Some(tz) => {
            write!(buffer, "{} {}", dt.format("%Y-%m-%d %H:%M:%S%.9f"), tz)
                .context("Failed to format timestamp")?;
        }
        None => {
            buffer.push_str(&dt.format("%Y-%m-%d %H:%M:%S%.9f").to_string());
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Set number of threads for parallel processing
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .context("Failed to initialize thread pool")?;

    let parquet_file = File::open(&args.file)
        .with_context(|| format!("Failed to open parquet file '{}'", args.file))?;

    // Create a reader builder
    let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_file)
        .context("Failed to create parquet reader")?;
    let schema = builder.schema().clone();

    // Get metadata before building the reader
    let metadata = builder.metadata();
    let total_rows = (0..metadata.num_row_groups())
        .map(|i| metadata.row_group(i).num_rows() as u64)
        .sum();

    // Build the reader
    let batch_reader = builder.build().context("Failed to build batch reader")?;

    let progress = create_progress_bar(total_rows);
    let string_pool = Arc::new(Mutex::new(StringPool::new(1000)));

    let temp_csv = NamedTempFile::new().context("Failed to create temporary file")?;

    {
        let file = temp_csv
            .reopen()
            .context("Failed to reopen temporary file")?;
        let buffered = BufWriter::with_capacity(args.buffer_size * 1024, file);
        let mut wtr = Writer::from_writer(buffered);

        // Write headers
        let headers: Vec<String> = schema
            .fields()
            .iter()
            .map(|field| field.name().to_string())
            .collect();
        wtr.write_record(&headers)
            .context("Failed to write CSV headers")?;

        // Write data
        for batch in batch_reader {
            let batch = batch.context("Failed to read batch")?;
            let string_pool_clone = Arc::clone(&string_pool);

            let rows: Vec<Vec<String>> = (0..batch.num_rows())
                .into_par_iter()
                .map(|row_idx| {
                    let mut row = Vec::with_capacity(batch.num_columns());
                    let mut pool = string_pool_clone.lock();

                    for col_idx in 0..batch.num_columns() {
                        let col = batch.column(col_idx);
                        let mut buffer = pool.get();
                        get_column_value_with_buffer(col, row_idx, &mut buffer).with_context(
                            || format!("Failed to process row {} column {}", row_idx, col_idx),
                        )?;
                        row.push(buffer);
                    }
                    Ok(row)
                })
                .collect::<Result<Vec<Vec<String>>>>()?;

            for row in rows {
                wtr.write_record(&row)
                    .context("Failed to write CSV record")?;
                let mut pool = string_pool.lock();
                for s in row {
                    pool.return_string(s);
                }
            }
            progress.inc(batch.num_rows() as u64);
        }

        wtr.flush().context("Failed to flush CSV writer")?;
        progress.finish();

        if let Err(e) = csvlens::run_csvlens([temp_csv.path().to_str().unwrap()]) {
            eprintln!("Error displaying CSV: {}", e);
        }
    }

    Ok(())
}

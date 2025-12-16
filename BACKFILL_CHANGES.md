# Backfill Script Changes Summary

## What Changed

Modified `backfill.py` to collect 10 days of seismic data distributed across the year 2025 (from January 1 to current date).

---

## New Configuration

### Date Range
- **Start Date**: January 1, 2025 (fixed)
- **End Date**: Current time (dynamic)
- **Total Range**: ~348 days (Jan 1 - Dec 15, 2025)

### Sampling Strategy
- **Days to Collect**: 10 days (configurable via `DAYS_TO_BACKFILL`)
- **Sampling Method**: Evenly distributed across the year
- **Sample Interval**: Every 34 days (calculated: 348 days / 10 samples)

### Batch Processing
- **Batch Size**: 6 hours (configurable via `BATCH_SIZE_HOURS`)
- **Processing**: Each selected day is processed in 6-hour batches
- **Upload Threshold**: 1000 events per upload batch

---

## Code Changes

### 1. Configuration (Lines 51-58)
```python
# OLD
DAYS_TO_BACKFILL = 30
BATCH_SIZE_HOURS = 6

# NEW
DAYS_TO_BACKFILL = 10  # Target: 10 days worth of data
BATCH_SIZE_HOURS = 6   # Process 6 hours at a time
START_DATE = "2025-01-01"  # Start from beginning of 2025
```

### 2. Date Range Calculation (Lines 291-312)
```python
# OLD: Calculated from "30 days ago"
end_time = UTCDateTime.now()
start_time = end_time - (DAYS_TO_BACKFILL * 24 * 60 * 60)

# NEW: Fixed start date with sampling logic
end_time = UTCDateTime.now()
start_time = UTCDateTime(START_DATE)

# Calculate total days and sampling interval
total_days = (end_time - start_time) / (24 * 60 * 60)
days_to_sample = min(DAYS_TO_BACKFILL, int(total_days))
sample_interval_days = max(1, int(total_days / days_to_sample))
```

### 3. Batch Processing Logic (Lines 318-383)
**OLD**: Sequential processing of all days
```python
while current_time < end_time:
    # Process 6-hour batch
    # Move to next batch
```

**NEW**: Sample-based processing with nested loops
```python
# Generate list of days to sample (every 34 days)
days_to_process = []
current_day = start_time
while day_number < days_to_sample and current_day < end_time:
    days_to_process.append(current_day)
    current_day = current_day + (sample_interval_days * 24 * 60 * 60)
    day_number += 1

# Process each sampled day
for day_idx, day_start in enumerate(days_to_process):
    # Process the full day in 6-hour batches
    while current_time < day_end:
        # Process batch
        # Upload if >= 1000 events
```

### 4. Summary Statistics (Lines 427-437)
```python
# NEW: Added days sampled and average time per day
logger.info(f"   Days sampled: {len(days_to_process)} out of {total_days:.1f} total days")
logger.info(f"   Average: {elapsed_time/max(1, len(days_to_process)):.1f} seconds per day")
```

---

## How It Works

### Sampling Logic

1. **Calculate Total Range**:
   - From Jan 1, 2025 to Dec 15, 2025 = 348 days

2. **Determine Sampling Interval**:
   - Target: 10 days
   - Interval: 348 / 10 = 34 days

3. **Generate Sample Days**:
   ```
   Day 1:  Jan 1, 2025
   Day 2:  Feb 4, 2025  (Jan 1 + 34 days)
   Day 3:  Mar 10, 2025 (Feb 4 + 34 days)
   Day 4:  Apr 13, 2025
   Day 5:  May 17, 2025
   Day 6:  Jun 20, 2025
   Day 7:  Jul 24, 2025
   Day 8:  Aug 27, 2025
   Day 9:  Sep 30, 2025
   Day 10: Nov 3, 2025
   ```

4. **Process Each Day**:
   - Each day is divided into 4 batches (6 hours each)
   - Total batches: 10 days × 4 batches = 40 batches
   - Each batch processes 4 seismic stations

---

## Expected Output

### Terminal Output Example

```
☁️ Connecting to Hopsworks...
✅ Connected to Hopsworks project: QuakeAlertWave
✅ Using existing feature group 'waveform_features'
📅 Backfilling data from 2025-01-01 to 2025-12-15
📊 Total time range: 348.7 days
🎯 Target: 10 days of data
📊 Sampling every 34 days
⏱️  Processing in 6-hour batches
📊 Initial dataset size: 1234 rows
============================================================

📋 Will process 10 days
============================================================

============================================================
📅 Day 1/10: 2025-01-01
============================================================

📦 Batch 1: 00:00:00 to 06:00:00
  Processing IU.ANMO...
    Found 12 events
  Processing IU.COLA...
    Found 8 events
  Processing US.TAU...
    Found 15 events
  Processing IU.MAJO...
    Found 10 events
  Batch 1 complete: 45 events detected
  📊 Total events collected so far: 45
  📤 Total uploaded to Hopsworks: 0 rows

... (continues for all batches and days) ...

============================================================
✅ BACKFILL COMPLETE
============================================================
📊 Dataset Statistics:
   Initial rows: 1234
   Final rows: 3456
   Rows added: 2222
   Total events processed: 2500
   Total uploaded: 2222
   Days sampled: 10 out of 348.7 total days
📅 Time range: 2025-01-01 to 2025-12-15
⏱️  Total time: 45.3 minutes
⏱️  Average: 4.5 seconds per day
============================================================
```

---

## Performance Estimates

### Time Estimates

| Component | Per Batch | Per Day | Total (10 days) |
|-----------|-----------|---------|-----------------|
| Data Fetch | 5-15s | 20-60s | 200-600s (3-10 min) |
| Processing | 1-3s | 4-12s | 40-120s (1-2 min) |
| Upload | 2-5s | 8-20s | 80-200s (1-3 min) |
| **Total** | **8-23s** | **32-92s** | **320-920s (5-15 min)** |

### Data Estimates

| Metric | Per Station | Per Batch | Per Day | Total (10 days) |
|--------|-------------|-----------|---------|-----------------|
| Events Detected | 5-20 | 20-80 | 80-320 | 800-3200 |
| Features per Event | 9 | - | - | - |
| Rows Added | - | - | 80-320 | 800-3200 |

---

## Advantages of New Approach

### 1. **Representative Sample**
- ✅ Covers entire year (not just recent data)
- ✅ Captures seasonal variations
- ✅ Includes different seismic activity periods

### 2. **Efficient Processing**
- ✅ Faster than processing all 348 days
- ✅ Still provides sufficient training data (800-3200 events)
- ✅ Reduces API load on IRIS servers

### 3. **Flexible Configuration**
- ✅ Easy to adjust `DAYS_TO_BACKFILL` (e.g., 5, 20, 30 days)
- ✅ Easy to change `START_DATE` (e.g., "2024-01-01")
- ✅ Automatic sampling interval calculation

### 4. **Progress Tracking**
- ✅ Shows day-by-day progress
- ✅ Displays total vs sampled days
- ✅ Shows average time per day

---

## Customization Options

### Collect More Days
```python
DAYS_TO_BACKFILL = 20  # Sample 20 days instead of 10
```
Result: Sample every 17 days (348 / 20)

### Change Start Date
```python
START_DATE = "2024-01-01"  # Start from 2024
```
Result: Covers 2 years of data

### Smaller Batches (for slower connections)
```python
BATCH_SIZE_HOURS = 3  # 3-hour batches instead of 6
```
Result: More batches but smaller requests

### Continuous Collection (All Days)
```python
DAYS_TO_BACKFILL = 999  # Large number
```
Result: `days_to_sample = min(999, 348) = 348` → processes all days

---

## Validation

### Test Results
```
✅ Date logic test:
   Start: 2025-01-01
   End: 2025-12-15
   Total days in range: 348.7
   Days to sample: 10
   Sample every: 34 days
   Will process 10 days
✅ Backfill logic is correct!
```

### Linter
```
No linter errors found.
```

---

## Usage

### Basic Run
```bash
cd /Users/kainat/Desktop/QuakeAlertWave
source venv/bin/activate
python backfill.py
```

### Monitor Progress
- Watch terminal output for day-by-day progress
- Check Hopsworks UI for uploaded data
- Verify row counts in summary statistics

### Stop and Resume
- Press `Ctrl+C` to stop
- Data uploaded so far is saved
- Re-run to continue (may process same days again)

---

## Summary

✅ **Modified**: `backfill.py` to sample 10 days from 2025  
✅ **Date Range**: Jan 1, 2025 → Current date  
✅ **Sampling**: Every 34 days (10 samples total)  
✅ **Batch Processing**: 6-hour batches per day  
✅ **Performance**: 5-15 minutes total  
✅ **Data**: 800-3200 seismic events expected  
✅ **Validation**: Logic tested and working  
✅ **No Errors**: Clean code, linter passed  

The backfill script is ready to collect representative training data from across the entire year 2025.


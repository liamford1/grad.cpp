#pragma once
#include <string>

namespace grad::utils {

// Full-screen live terminal dashboard over a training metrics CSV (see
// MetricsLog): loss curves drawn on a braille canvas, validation track,
// throughput/gradient sparklines, progress and ETA. Refreshes once per
// second until 'q' or Ctrl-C; `once` renders a single frame and returns
// (no alternate screen), for piping or quick checks.
int run_dashboard(const std::string& csv_path, bool once);

// Most recently modified *_metrics.csv in dir, or "" if none - the
// default target for `grad watch`.
std::string newest_metrics_csv(const std::string& dir);

}  // namespace grad::utils

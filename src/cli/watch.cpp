// grad watch: live terminal dashboard over a run's metrics CSV.

#include "cli/args.h"
#include "commands.h"

#include "utils/dashboard.h"

#include <iostream>
#include <string>

namespace cli {

int run_watch(const Invocation& invocation) {
    std::string run;
    bool once = false;

    Command cmd(invocation.usage_name(), std::string(invocation.summary));
    cmd.describe("Loss curves (raw and EMA), the validation track with its running best, "
                 "gradient-norm and step-time sparklines, progress and ETA. Refreshes once a "
                 "second; q quits. Open it in a second terminal while training.");
    cmd.optional("run", run,
                 "a run prefix (tinystories_modern) or a metrics CSV path; defaults to the "
                 "most recently modified *_metrics.csv in this directory");
    cmd.flag("--once", once, "render one frame and exit, for piping or a quick check");
    if (cmd.parse(invocation.args) == ParseResult::HelpShown) return 0;

    std::string csv;
    if (!run.empty()) {
        csv = run.ends_with(".csv") ? run : run + "_metrics.csv";
    } else {
        csv = utils::newest_metrics_csv(".");
        if (csv.empty()) {
            std::cerr << "No *_metrics.csv found; start a training run first "
                      << "(runs on this build log metrics automatically)." << std::endl;
            return 1;
        }
    }
    return utils::run_dashboard(csv, once);
}

}  // namespace cli

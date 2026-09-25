// grad: command-line front end. Each subcommand lives in src/cli/; this
// file maps names to them, renders the top-level help, and is the single
// place errors become an exit status.

#include "cli/args.h"
#include "cli/commands.h"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct CommandEntry {
    std::string_view name;
    int (*run)(const cli::Invocation&);
    std::string_view summary;
};

constexpr std::array kCommands{
    CommandEntry{"prepare", cli::run_prepare, "Pre-tokenize a corpus into memory-mapped .bin token files"},
    CommandEntry{"train", cli::run_train, "Train a preset on a corpus; Ctrl-C saves a resumable state"},
    CommandEntry{"train-fast", cli::run_train_fast, "Train the tiny 'fast' preset: a one-minute smoke test"},
    CommandEntry{"generate", cli::run_generate, "Sample greedy and sampled continuations from a checkpoint"},
    CommandEntry{"chat", cli::run_chat, "Interactive REPL: type a prompt, watch the model continue it"},
    CommandEntry{"eval", cli::run_eval, "Loss and perplexity on held-out and training windows"},
    CommandEntry{"bench", cli::run_bench, "Training and generation throughput, median of repeated trials"},
    CommandEntry{"watch", cli::run_watch, "Live terminal dashboard for a training run"},
};

void print_usage(std::ostream& out, std::string_view program) {
    size_t width = 0;
    for (const CommandEntry& entry : kCommands) width = std::max(width, entry.name.size());

    out << "Usage: " << program << " <command> [arguments] [options]\n\nCommands:\n";
    for (const CommandEntry& entry : kCommands) {
        out << "  " << entry.name << std::string(width + 3 - entry.name.size(), ' ')
            << entry.summary << '\n';
    }
    out << "\nRun '" << program << " <command> --help' for a command's arguments and options.\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::vector<std::string_view> args(argv, argv + argc);
    const std::string_view program = args.empty() ? "grad" : args[0];

    if (args.size() < 2) {
        print_usage(std::cerr, program);
        return 1;
    }
    const std::string_view name = args[1];
    if (name == "-h" || name == "--help" || name == "help") {
        print_usage(std::cout, program);
        return 0;
    }

    const CommandEntry* entry = nullptr;
    for (const CommandEntry& candidate : kCommands) {
        if (candidate.name == name) entry = &candidate;
    }
    if (!entry) {
        std::cerr << "Error: unknown command '" << name << "'\n\n";
        print_usage(std::cerr, program);
        return 1;
    }

    const cli::Invocation invocation{
        .program = program,
        .command = entry->name,
        .summary = entry->summary,
        .args = std::span(args).subspan(2),
    };
    try {
        return entry->run(invocation);
    } catch (const cli::UsageError& e) {
        std::cerr << "Error: " << e.what() << "\nRun '" << invocation.usage_name()
                  << " --help' for usage." << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}

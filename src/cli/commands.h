#pragma once

// Entry points of the grad subcommands, one per file in src/cli/. main.cpp
// maps command names to these; each parses its own arguments with
// cli::Command and returns the process exit code. Errors propagate as
// exceptions to main's single handler (cli::UsageError for a malformed
// command line).

#include <span>
#include <string>
#include <string_view>

namespace cli {

struct Invocation {
    std::string_view program;  // argv[0]
    std::string_view command;  // the subcommand name as typed
    std::string_view summary;  // its one-line description from the dispatch table
    std::span<const std::string_view> args;  // everything after the command name

    // "<program> <command>", as usage lines and hints spell it.
    [[nodiscard]] std::string usage_name() const {
        return std::string(program) + " " + std::string(command);
    }
};

int run_prepare(const Invocation& invocation);
int run_train(const Invocation& invocation);
int run_train_fast(const Invocation& invocation);
int run_generate(const Invocation& invocation);
int run_chat(const Invocation& invocation);
int run_eval(const Invocation& invocation);
int run_bench(const Invocation& invocation);
int run_watch(const Invocation& invocation);
int run_presets(const Invocation& invocation);

}  // namespace cli

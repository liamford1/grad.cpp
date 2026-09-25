// Unit tests for the grad CLI argument parser (include/cli/args.h).

#include "cli/args.h"
#include "../test_util.h"

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using cli::Command;
using cli::ParseResult;

ParseResult parse(Command& cmd, std::initializer_list<std::string_view> tokens) {
    const std::vector<std::string_view> args(tokens);
    return cmd.parse(args);
}

// The UsageError message parse throws, or "" if it accepts the tokens.
std::string usage_error(Command& cmd, std::initializer_list<std::string_view> tokens) {
    try {
        (void)parse(cmd, tokens);
    } catch (const cli::UsageError& e) {
        return e.what();
    }
    return "";
}

// The shape of `grad eval`: two required positionals, then optionals.
struct EvalArgs {
    std::string checkpoint;
    std::string corpus;
    std::optional<int> vocab;
    int seq = 256;
    int max_batches = 0;
    Command cmd{"grad eval", "Score a checkpoint."};

    EvalArgs() {
        cmd.positional("ckpt", checkpoint, "checkpoint");
        cmd.positional("corpus", corpus, "corpus");
        cmd.optional("vocab", vocab, "vocab size").at_least(1);
        cmd.optional("seq", seq, "window length").at_least(1);
        cmd.optional("max_batches", max_batches, "batch cap").at_least(0).named("--batches");
    }
};

void test_positionals() {
    {
        EvalArgs a;
        CHECK(parse(a.cmd, {"model.bin", "data.txt"}) == ParseResult::Ok);
        CHECK(a.checkpoint == "model.bin");
        CHECK(a.corpus == "data.txt");
        CHECK(!a.vocab.has_value());
        CHECK(a.seq == 256);
        CHECK(a.max_batches == 0);
    }
    {
        EvalArgs a;
        CHECK(parse(a.cmd, {"model.bin", "data.txt", "16000", "128", "32"}) == ParseResult::Ok);
        CHECK(a.vocab == 16000);
        CHECK(a.seq == 128);
        CHECK(a.max_batches == 32);
    }
    {
        // Optional positionals by name, in any order and around positionals.
        EvalArgs a;
        CHECK(parse(a.cmd, {"--batches", "32", "model.bin", "--seq=64", "data.txt"})
              == ParseResult::Ok);
        CHECK(a.checkpoint == "model.bin");
        CHECK(a.corpus == "data.txt");
        CHECK(!a.vocab.has_value());
        CHECK(a.seq == 64);
        CHECK(a.max_batches == 32);
    }
    {
        // An empty string is a value, not an absent argument.
        std::string prompt = "unset";
        Command cmd("grad generate", "Sample.");
        cmd.optional("prompt", prompt, "prompt");
        CHECK(parse(cmd, {""}) == ParseResult::Ok);
        CHECK(prompt.empty());
    }
}

void test_usage_errors() {
    {
        EvalArgs a;
        CHECK(usage_error(a.cmd, {"model.bin"}) == "missing required argument <corpus>");
    }
    {
        EvalArgs a;
        CHECK(usage_error(a.cmd, {"m", "c", "1", "2", "3", "4"}) == "unexpected argument '4'");
    }
    {
        EvalArgs a;
        CHECK(usage_error(a.cmd, {"m", "c", "--bogus", "1"}) == "unknown option '--bogus'");
    }
    {
        EvalArgs a;
        CHECK(usage_error(a.cmd, {"m", "c", "--seq"}) == "--seq: missing value");
    }
    {
        // A known option where the value should be means the value is missing.
        EvalArgs a;
        CHECK(usage_error(a.cmd, {"m", "c", "--seq", "--batches", "3"}) == "--seq: missing value");
    }
    {
        EvalArgs a;
        CHECK(usage_error(a.cmd, {"m", "c", "--seq", "abc"})
              == "--seq: expected an integer, got 'abc'");
    }
    {
        // Positionally the argument is named by its positional name.
        EvalArgs a;
        CHECK(usage_error(a.cmd, {"m", "c", "16k"}) == "vocab: expected an integer, got '16k'");
    }
    {
        EvalArgs a;
        CHECK(usage_error(a.cmd, {"m", "c", "16000x"}) == "vocab: expected an integer, got '16000x'");
    }
    {
        EvalArgs a;
        CHECK(usage_error(a.cmd, {"m", "c", "0"}) == "vocab: must be >= 1, got 0");
    }
    {
        EvalArgs a;
        CHECK(usage_error(a.cmd, {"m", "c", "--batches", "-1"}) == "--batches: must be >= 0, got -1");
    }
    {
        EvalArgs a;
        CHECK(usage_error(a.cmd, {"m", "c", "--seq", "1", "--seq", "2"})
              == "--seq: given more than once");
    }
    {
        EvalArgs a;
        CHECK(usage_error(a.cmd, {"m", "c", "1", "--vocab", "2"})
              == "vocab: given both as an argument and as --vocab");
    }
    {
        EvalArgs a;
        CHECK(usage_error(a.cmd, {"m", "c", "--vocab", "2", "3"})
              == "vocab: given both as an argument and as --vocab");
    }
    {
        EvalArgs a;
        CHECK(usage_error(a.cmd, {"m", "c", "--seq", "99999999999"})
              == "--seq: '99999999999' is out of range");
    }
    {
        // The renamed alias replaces the default one.
        EvalArgs a;
        CHECK(usage_error(a.cmd, {"m", "c", "--max-batches", "3"})
              == "unknown option '--max-batches'");
    }
}

void test_options_and_switches() {
    int steps = 20;
    float temperature = 0.8f;
    std::uint32_t seed = 42;
    std::string json;
    bool greedy = false;
    const auto declare = [&](Command& cmd) {
        cmd.option("--steps", steps, "steps").at_least(1);
        cmd.option("--temperature", temperature, "temperature").positive();
        cmd.option("--top-p", temperature, "top-p").at_least(0).at_most(1);
        cmd.option("--seed", seed, "seed");
        cmd.option("--json", json, "output path").metavar("PATH");
        cmd.flag("--greedy", greedy, "greedy");
    };
    {
        Command cmd("grad x", "X.");
        declare(cmd);
        CHECK(parse(cmd, {"--steps", "5", "--temperature=0.25", "--json", "out.json", "--greedy",
                          "--seed", "7"})
              == ParseResult::Ok);
        CHECK(steps == 5);
        CHECK_NEAR(temperature, 0.25, 1e-7);
        CHECK(json == "out.json");
        CHECK(greedy);
        CHECK(seed == 7u);
    }
    {
        Command cmd("grad x", "X.");
        declare(cmd);
        CHECK(usage_error(cmd, {"--greedy=1"}) == "--greedy: takes no value");
    }
    {
        Command cmd("grad x", "X.");
        declare(cmd);
        CHECK(usage_error(cmd, {"--temperature", "0"}) == "--temperature: must be > 0, got 0");
    }
    {
        Command cmd("grad x", "X.");
        declare(cmd);
        CHECK(usage_error(cmd, {"--top-p", "1.5"}) == "--top-p: must be <= 1, got 1.5");
    }
    {
        Command cmd("grad x", "X.");
        declare(cmd);
        CHECK(usage_error(cmd, {"--temperature", "hot"})
              == "--temperature: expected a number, got 'hot'");
    }
    {
        Command cmd("grad x", "X.");
        declare(cmd);
        CHECK(usage_error(cmd, {"--temperature", "inf"})
              == "--temperature: expected a finite number, got 'inf'");
    }
    {
        Command cmd("grad x", "X.");
        declare(cmd);
        CHECK(usage_error(cmd, {"--seed", "-1"})
              == "--seed: expected a non-negative integer, got '-1'");
    }
    {
        Command cmd("grad x", "X.");
        declare(cmd);
        CHECK(usage_error(cmd, {"--json"}) == "--json: missing value");
    }
}

void test_option_terminator_and_dashes() {
    std::string prompt;
    std::string corpus = "default.txt";
    Command cmd("grad generate", "Sample.");
    cmd.optional("prompt", prompt, "prompt");
    cmd.optional("corpus", corpus, "corpus");
    // A single dash is never an option; after "--" nothing is.
    CHECK(parse(cmd, {"-5 degrees", "--", "--help"}) == ParseResult::Ok);
    CHECK(prompt == "-5 degrees");
    CHECK(corpus == "--help");
}

void test_help() {
    {
        EvalArgs a;
        CHECK(parse(a.cmd, {"m", "--bogus", "-h"}) == ParseResult::HelpShown);
        CHECK(a.checkpoint.empty());  // help wins before anything binds
    }
    {
        EvalArgs a;
        const std::string help = a.cmd.help();
        CHECK(help.rfind("Usage: grad eval <ckpt> <corpus> [vocab] [seq] [max_batches] [options]\n", 0)
              == 0);
        CHECK(help.find("Score a checkpoint.") != std::string::npos);
        CHECK(help.find("[seq], --seq N") != std::string::npos);
        CHECK(help.find("(default: 256)") != std::string::npos);
        CHECK(help.find("[max_batches], --batches N") != std::string::npos);
        CHECK(help.find("-h, --help") != std::string::npos);
    }
}

void test_parse_value() {
    CHECK(cli::parse_value<int>("-12", "n") == -12);
    CHECK(cli::parse_value<std::string>("x y", "s") == "x y");
    CHECK_NEAR(cli::parse_value<double>("3e-4", "lr"), 3e-4, 1e-12);
    bool threw = false;
    try {
        (void)cli::parse_value<int>("", "n");
    } catch (const cli::UsageError&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        (void)cli::parse_value<float>("1e300", "x");
    } catch (const cli::UsageError& e) {
        threw = std::string(e.what()) == "x: '1e300' is out of range";
    }
    CHECK(threw);
}

}  // namespace

int main() {
    test_positionals();
    test_usage_errors();
    test_options_and_switches();
    test_option_terminator_and_dashes();
    test_help();
    test_parse_value();
    return test_util::exit_code();
}

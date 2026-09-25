#pragma once

// A small declarative command-line parser for the grad subcommands.
//
// Each command declares what it accepts and binds every argument to a
// variable whose current value is the default:
//
//     int seq = 256;
//     std::optional<int> vocab;
//     cli::Command cmd("grad eval", "Score a checkpoint");
//     cmd.positional("ckpt", checkpoint, "checkpoint to score");
//     cmd.optional("vocab", vocab, "vocab size").at_least(1);
//     cmd.optional("seq", seq, "window length").at_least(1);
//     if (cmd.parse(args) == cli::ParseResult::HelpShown) return 0;
//
// Accepted forms: positionals in declaration order; `--name value` and
// `--name=value` for options; bare `--name` for switches; `--` ends option
// parsing. An optional positional can also be given by name (`--vocab N`),
// so a late one can be set without spelling out the ones before it.
// `-h`/`--help` anywhere prints generated help. Every malformed command
// line throws UsageError with a message naming the argument, e.g.
// "--steps: expected an integer, got 'abc'".

#include <charconv>
#include <concepts>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace cli {

// A command line that does not match the command's declaration.
class UsageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Value types an argument can bind to: strings, integers, floating point,
// or std::optional of one of those (for "unset unless given").
template <class T>
concept Scalar = std::same_as<T, std::string> || std::floating_point<T>
              || (std::integral<T> && !std::same_as<T, bool>);

template <class T> struct optional_value { using type = T; };
template <class T> struct optional_value<std::optional<T>> { using type = T; };
template <class T> using optional_value_t = typename optional_value<T>::type;

template <class T>
concept Bindable = Scalar<optional_value_t<T>>;

namespace detail {

[[noreturn]] void throw_bad_value(std::string_view label, std::string_view expected,
                                  std::string_view text);
[[noreturn]] void throw_out_of_range(std::string_view label, std::string_view text);
// Whole-string, finite, locale-independent parse of a decimal number.
[[nodiscard]] double parse_double(std::string_view text, std::string_view label);
// Compact text for a default or bound in help and errors: 0.8, 0.0003, 1.
[[nodiscard]] std::string format_number(double value);

}  // namespace detail

// Converts one command-line token. The whole token must be the value:
// "16k", "", and "16000x" are errors rather than atoi's 16 / 0 / 16000.
// `label` names the argument in error messages ("--steps", "vocab").
template <Scalar T>
[[nodiscard]] T parse_value(std::string_view text, std::string_view label) {
    if constexpr (std::same_as<T, std::string>) {
        return std::string(text);
    } else if constexpr (std::floating_point<T>) {
        const double value = detail::parse_double(text, label);
        if (value > static_cast<double>(std::numeric_limits<T>::max())
            || value < static_cast<double>(std::numeric_limits<T>::lowest())) {
            detail::throw_out_of_range(label, text);
        }
        return static_cast<T>(value);
    } else {
        T value{};
        const char* const first = text.data();
        const char* const last = first + text.size();
        const auto [ptr, ec] = std::from_chars(first, last, value);
        if (ec == std::errc::result_out_of_range) detail::throw_out_of_range(label, text);
        if (ec != std::errc() || ptr != last || text.empty()) {
            detail::throw_bad_value(
                label, std::is_signed_v<T> ? "an integer" : "a non-negative integer", text);
        }
        return value;
    }
}

enum class ParseResult {
    Ok,         // every argument bound; run the command
    HelpShown,  // -h/--help printed the help; the command should return 0
};

// One declared argument. Returned by the Command declaration methods so
// constraints can be chained onto it: cmd.option(...).at_least(1).
class Arg {
public:
    // Inclusive lower / upper bound, and a strict > 0 bound, checked on
    // the parsed value of a numeric argument.
    Arg& at_least(double bound);
    Arg& at_most(double bound);
    Arg& positive();
    // For an optional positional, the flag that also sets it (default
    // "--<name>").
    Arg& named(std::string flag);
    // Placeholder for the value in help ("N", "PATH"); defaults by type.
    Arg& metavar(std::string text);
    // Replaces the default shown in help, e.g. "the checkpoint's" for a
    // std::optional whose absence means "derive it".
    Arg& default_text(std::string text);

private:
    friend class Command;

    enum class Kind { Required, Optional, Option, Switch };
    enum class Given { No, Positionally, ByName };

    Arg(Kind kind, std::string name, std::string help)
        : kind_(kind), name_(std::move(name)), help_(std::move(help)) {}

    void check_bounds(double value, std::string_view text, std::string_view label) const;
    [[nodiscard]] bool is_positional() const noexcept {
        return kind_ == Kind::Required || kind_ == Kind::Optional;
    }

    Kind kind_;
    std::string name_;      // positional name, or the option's flag
    std::string flag_;      // flag that sets it by name ("" for a required positional)
    std::string metavar_;
    std::string help_;
    std::string default_;
    std::optional<double> lower_, upper_;
    bool lower_strict_ = false;
    Given given_ = Given::No;
    // Parses text into the bound variable; label names the argument as the
    // user gave it ("--vocab" or "vocab") in any UsageError.
    std::function<void(std::string_view text, std::string_view label)> store_;
};

class Command {
public:
    // usage_name is what the user typed to reach this command
    // ("./build/grad eval"); summary is its one-line description.
    Command(std::string usage_name, std::string summary);
    Command(const Command&) = delete;
    Command& operator=(const Command&) = delete;

    // Adds a paragraph to --help, between the summary and the argument list.
    Command& describe(std::string paragraph);

    // A required positional argument.
    template <Bindable T>
    Arg& positional(std::string name, T& out, std::string help) {
        return bind(Arg::Kind::Required, std::move(name), out, std::move(help));
    }
    // An optional positional argument, also settable as --<name>. Must
    // follow every required one; its default is out's current value.
    template <Bindable T>
    Arg& optional(std::string name, T& out, std::string help) {
        return bind(Arg::Kind::Optional, std::move(name), out, std::move(help));
    }
    // A named option taking a value: --flag value or --flag=value.
    template <Bindable T>
    Arg& option(std::string flag, T& out, std::string help) {
        return bind(Arg::Kind::Option, std::move(flag), out, std::move(help));
    }
    // A named switch: --flag sets out to true.
    Arg& flag(std::string flag, bool& out, std::string help);

    // Binds args (the tokens after the command name). Throws UsageError on
    // a malformed command line; prints help and returns HelpShown when
    // -h/--help appears before any "--".
    [[nodiscard]] ParseResult parse(std::span<const std::string_view> args);

    [[nodiscard]] std::string help() const;

private:
    template <Bindable T>
    Arg& bind(Arg::Kind kind, std::string name, T& out, std::string help) {
        using Value = optional_value_t<T>;
        Arg& arg = add(kind, std::move(name), std::move(help));
        if constexpr (std::integral<Value>) {
            arg.metavar_ = "N";
        } else if constexpr (std::floating_point<Value>) {
            arg.metavar_ = "X";
        } else {
            arg.metavar_ = kind == Arg::Kind::Option ? "VALUE" : uppercase(arg.name_);
        }
        if constexpr (!std::same_as<T, Value>) {
            // std::optional: no default unless default_text() gives one.
        } else if constexpr (std::same_as<Value, std::string>) {
            arg.default_ = out;
        } else if constexpr (std::integral<Value>) {
            arg.default_ = std::to_string(out);
        } else {
            arg.default_ = detail::format_number(static_cast<double>(out));
        }
        // arg lives in a deque that only grows, and Command is neither
        // copyable nor movable, so the captured reference stays valid.
        arg.store_ = [&out, &arg](std::string_view text, std::string_view label) {
            Value value = parse_value<Value>(text, label);
            if constexpr (std::is_arithmetic_v<Value>) {
                arg.check_bounds(static_cast<double>(value), text, label);
            } else {
                (void)arg;  // strings have no bounds
            }
            out = std::move(value);
        };
        return arg;
    }

    Arg& add(Arg::Kind kind, std::string name, std::string help);
    [[nodiscard]] Arg* find_named(std::string_view flag);
    [[nodiscard]] static std::string uppercase(std::string_view text);

    std::string usage_name_;
    std::string summary_;
    std::vector<std::string> description_;
    std::deque<Arg> args_;
};

}  // namespace cli

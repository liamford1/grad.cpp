#include "cli/args.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>

// libc++ gained floating-point std::from_chars in LLVM 17; older Apple
// toolchains (Xcode 15) only have the integer overloads.
#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION < 170000
#define GRAD_CLI_STRTOD_FALLBACK 1
#include <cerrno>
#endif

namespace cli {

namespace detail {

void throw_bad_value(std::string_view label, std::string_view expected, std::string_view text) {
    throw UsageError(std::string(label) + ": expected " + std::string(expected) + ", got '"
                     + std::string(text) + "'");
}

void throw_out_of_range(std::string_view label, std::string_view text) {
    throw UsageError(std::string(label) + ": '" + std::string(text) + "' is out of range");
}

double parse_double(std::string_view text, std::string_view label) {
    double value = 0.0;
#ifdef GRAD_CLI_STRTOD_FALLBACK
    // strtod skips leading whitespace and accepts hex; reject both so the
    // accepted syntax matches the from_chars path.
    const std::string copy(text);
    const bool plain = !copy.empty() && !std::isspace(static_cast<unsigned char>(copy[0]))
                    && copy.find_first_of("xX") == std::string::npos;
    char* end = nullptr;
    errno = 0;
    value = std::strtod(copy.c_str(), &end);
    if (!plain || end != copy.c_str() + copy.size()) throw_bad_value(label, "a number", text);
    if (errno == ERANGE) throw_out_of_range(label, text);
#else
    const char* const first = text.data();
    const char* const last = first + text.size();
    const auto [ptr, ec] = std::from_chars(first, last, value);
    if (ec == std::errc::result_out_of_range) throw_out_of_range(label, text);
    if (ec != std::errc() || ptr != last || text.empty()) throw_bad_value(label, "a number", text);
#endif
    if (!std::isfinite(value)) throw_bad_value(label, "a finite number", text);
    return value;
}

std::string format_number(double value) {
    std::ostringstream out;
    out << value;
    return out.str();
}

}  // namespace detail

// ---- Arg -------------------------------------------------------------------

Arg& Arg::at_least(double bound) {
    lower_ = bound;
    lower_strict_ = false;
    return *this;
}

Arg& Arg::at_most(double bound) {
    upper_ = bound;
    return *this;
}

Arg& Arg::positive() {
    lower_ = 0.0;
    lower_strict_ = true;
    return *this;
}

Arg& Arg::named(std::string flag) {
    if (kind_ != Kind::Optional) {
        throw std::logic_error("cli: only an optional positional takes an alias flag");
    }
    flag_ = std::move(flag);
    return *this;
}

Arg& Arg::metavar(std::string text) {
    metavar_ = std::move(text);
    return *this;
}

Arg& Arg::default_text(std::string text) {
    default_ = std::move(text);
    return *this;
}

void Arg::check_bounds(double value, std::string_view text, std::string_view label) const {
    const auto fail = [&](std::string_view relation, double bound) {
        throw UsageError(std::string(label) + ": must be " + std::string(relation) + " "
                         + detail::format_number(bound) + ", got " + std::string(text));
    };
    if (lower_) {
        if (lower_strict_ ? !(value > *lower_) : !(value >= *lower_)) {
            fail(lower_strict_ ? ">" : ">=", *lower_);
        }
    }
    if (upper_ && !(value <= *upper_)) fail("<=", *upper_);
}

// ---- Command ---------------------------------------------------------------

Command::Command(std::string usage_name, std::string summary)
    : usage_name_(std::move(usage_name)), summary_(std::move(summary)) {}

Command& Command::describe(std::string paragraph) {
    description_.push_back(std::move(paragraph));
    return *this;
}

Arg& Command::flag(std::string flag, bool& out, std::string help) {
    Arg& arg = add(Arg::Kind::Switch, std::move(flag), std::move(help));
    arg.store_ = [&out](std::string_view, std::string_view) { out = true; };
    return arg;
}

Arg& Command::add(Arg::Kind kind, std::string name, std::string help) {
    const bool positional = kind == Arg::Kind::Required || kind == Arg::Kind::Optional;
    if (positional) {
        if (name.empty() || name.front() == '-') {
            throw std::logic_error("cli: positional '" + name + "' must not look like a flag");
        }
        if (kind == Arg::Kind::Required
            && std::any_of(args_.begin(), args_.end(),
                           [](const Arg& a) { return a.kind_ == Arg::Kind::Optional; })) {
            throw std::logic_error("cli: required positional '" + name
                                   + "' declared after an optional one");
        }
    } else if (name.size() < 3 || name.compare(0, 2, "--") != 0) {
        throw std::logic_error("cli: option '" + name + "' must be spelled --name");
    }

    Arg arg(kind, std::move(name), std::move(help));
    if (kind == Arg::Kind::Optional) {
        arg.flag_ = "--" + arg.name_;
        std::replace(arg.flag_.begin(), arg.flag_.end(), '_', '-');
    } else if (!positional) {
        arg.flag_ = arg.name_;
    }
    args_.push_back(std::move(arg));
    return args_.back();
}

Arg* Command::find_named(std::string_view flag) {
    for (Arg& arg : args_) {
        if (!arg.flag_.empty() && arg.flag_ == flag) return &arg;
    }
    return nullptr;
}

std::string Command::uppercase(std::string_view text) {
    std::string out(text);
    for (char& c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return out;
}

namespace {

bool is_help(std::string_view token) { return token == "-h" || token == "--help"; }

// Only "--name" tokens are options. A single dash stays positional, so a
// prompt such as "- a list" or a negative number is never mistaken for one.
bool looks_like_option(std::string_view token) {
    return token.size() > 2 && token.compare(0, 2, "--") == 0;
}

}  // namespace

ParseResult Command::parse(std::span<const std::string_view> tokens) {
    for (std::string_view token : tokens) {
        if (token == "--") break;
        if (is_help(token)) {
            std::cout << help() << std::flush;
            return ParseResult::HelpShown;
        }
    }

    std::vector<Arg*> positionals;
    for (Arg& arg : args_) {
        if (arg.is_positional()) positionals.push_back(&arg);
    }

    // An option's value is the next token unless that token is itself a
    // known option, which means the value was left out.
    const auto is_known_option = [&](std::string_view token) {
        if (!looks_like_option(token)) return false;
        return find_named(token.substr(0, token.find('='))) != nullptr;
    };

    size_t next_positional = 0;
    bool options_ended = false;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string_view token = tokens[i];
        if (!options_ended && token == "--") {
            options_ended = true;
            continue;
        }

        if (!options_ended && looks_like_option(token)) {
            const size_t eq = token.find('=');
            const std::string_view name = token.substr(0, eq);
            Arg* arg = find_named(name);
            if (!arg) throw UsageError("unknown option '" + std::string(name) + "'");
            const std::string label(name);

            if (arg->kind_ == Arg::Kind::Switch) {
                if (eq != std::string_view::npos) throw UsageError(label + ": takes no value");
                arg->store_({}, label);
                arg->given_ = Arg::Given::ByName;
                continue;
            }
            std::string_view value;
            if (eq != std::string_view::npos) {
                value = token.substr(eq + 1);
            } else if (i + 1 < tokens.size() && !is_known_option(tokens[i + 1])) {
                value = tokens[++i];
            } else {
                throw UsageError(label + ": missing value");
            }
            if (arg->given_ == Arg::Given::Positionally) {
                throw UsageError(arg->name_ + ": given both as an argument and as " + label);
            }
            if (arg->given_ == Arg::Given::ByName) throw UsageError(label + ": given more than once");
            arg->store_(value, label);
            arg->given_ = Arg::Given::ByName;
            continue;
        }

        if (next_positional == positionals.size()) {
            throw UsageError("unexpected argument '" + std::string(token) + "'");
        }
        Arg* arg = positionals[next_positional++];
        if (arg->given_ == Arg::Given::ByName) {
            throw UsageError(arg->name_ + ": given both as an argument and as " + arg->flag_);
        }
        arg->store_(token, arg->name_);
        arg->given_ = Arg::Given::Positionally;
    }

    for (const Arg* arg : positionals) {
        if (arg->kind_ == Arg::Kind::Required && arg->given_ == Arg::Given::No) {
            throw UsageError("missing required argument <" + arg->name_ + ">");
        }
    }
    return ParseResult::Ok;
}

namespace {

constexpr size_t kHelpWidth = 80;
constexpr size_t kMaxLabelColumn = 30;

// Appends text word-wrapped to kHelpWidth, every line indented by indent;
// the first line starts at column `column` (already written).
void append_wrapped(std::string& out, std::string_view text, size_t indent, size_t column) {
    std::istringstream words{std::string(text)};
    std::string word;
    bool line_empty = true;
    while (words >> word) {
        if (!line_empty && column + 1 + word.size() > kHelpWidth) {
            out += '\n';
            out.append(indent, ' ');
            column = indent;
            line_empty = true;
        }
        if (!line_empty) {
            out += ' ';
            ++column;
        }
        out += word;
        column += word.size();
        line_empty = false;
    }
    out += '\n';
}

}  // namespace

std::string Command::help() const {
    // Every command has at least -h, so "[options]" is always true.
    std::string usage = "Usage: " + usage_name_;
    for (const Arg& arg : args_) {
        if (arg.kind_ == Arg::Kind::Required) usage += " <" + arg.name_ + ">";
        else if (arg.kind_ == Arg::Kind::Optional) usage += " [" + arg.name_ + "]";
    }
    usage += " [options]";

    std::string out;
    append_wrapped(out, usage, 4, 0);
    out += '\n';
    append_wrapped(out, summary_, 0, 0);
    for (const std::string& paragraph : description_) {
        out += '\n';
        append_wrapped(out, paragraph, 0, 0);
    }

    struct Row {
        std::string label;
        std::string text;
    };
    std::vector<Row> arguments;
    std::vector<Row> options;
    for (const Arg& arg : args_) {
        std::string text = arg.help_;
        if (!arg.default_.empty()) text += " (default: " + arg.default_ + ")";
        switch (arg.kind_) {
            case Arg::Kind::Required:
                arguments.push_back({"<" + arg.name_ + ">", text});
                break;
            case Arg::Kind::Optional:
                arguments.push_back({"[" + arg.name_ + "], " + arg.flag_ + " " + arg.metavar_, text});
                break;
            case Arg::Kind::Option:
                options.push_back({arg.flag_ + " " + arg.metavar_, text});
                break;
            case Arg::Kind::Switch:
                options.push_back({arg.flag_, text});
                break;
        }
    }
    options.push_back({"-h, --help", "show this help"});

    size_t column = 0;
    for (const auto* rows : {&arguments, &options}) {
        for (const Row& row : *rows) column = std::max(column, row.label.size());
    }
    column = std::min(column + 4, kMaxLabelColumn);

    const auto section = [&](std::string_view title, const std::vector<Row>& rows) {
        if (rows.empty()) return;
        out += '\n';
        out += title;
        out += ":\n";
        for (const Row& row : rows) {
            out += "  " + row.label;
            if (row.label.size() + 2 < column) {
                out.append(column - row.label.size() - 2, ' ');
            } else {
                out += '\n';
                out.append(column, ' ');
            }
            append_wrapped(out, row.text, column, column);
        }
    };
    section("Arguments", arguments);
    section("Options", options);
    return out;
}

}  // namespace cli

// grad presets: print the preset table, for people (a table) or for tools
// (--json; benchmarks/check_presets.py compares it with the PyTorch
// baseline's copy).

#include "cli/args.h"
#include "commands.h"
#include "presets.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace cli {

namespace {

std::string format_float(float value) {
    // Shortest fixed-point text that reads back as the same float: 0.0003,
    // not 0.000300000014 or 3e-04.
    std::array<char, 64> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                      std::chars_format::fixed);
    return std::string(buffer.data(), result.ptr);
}

// The numeric fields, in display order, as JSON-ready text.
struct Field {
    std::string_view name;
    std::string (*format)(const Preset&);
};

constexpr std::array kFields{
    Field{"vocab_size", [](const Preset& p) { return std::to_string(p.vocab_size); }},
    Field{"d_model", [](const Preset& p) { return std::to_string(p.d_model); }},
    Field{"num_layers", [](const Preset& p) { return std::to_string(p.num_layers); }},
    Field{"num_heads", [](const Preset& p) { return std::to_string(p.num_heads); }},
    Field{"max_len", [](const Preset& p) { return std::to_string(p.max_len); }},
    Field{"seq_length", [](const Preset& p) { return std::to_string(p.seq_length); }},
    Field{"batch_size", [](const Preset& p) { return std::to_string(p.batch_size); }},
    Field{"grad_accum", [](const Preset& p) { return std::to_string(p.grad_accum); }},
    Field{"learning_rate", [](const Preset& p) { return format_float(p.learning_rate); }},
    Field{"dropout", [](const Preset& p) { return format_float(p.dropout); }},
    Field{"warmup_steps", [](const Preset& p) { return std::to_string(p.warmup_steps); }},
    Field{"num_steps", [](const Preset& p) { return std::to_string(p.num_steps); }},
    Field{"checkpoint_interval", [](const Preset& p) { return std::to_string(p.checkpoint_interval); }},
    Field{"eval_interval", [](const Preset& p) { return std::to_string(p.eval_interval); }},
    Field{"max_eval_batches", [](const Preset& p) { return std::to_string(p.max_eval_batches); }},
};

std::string json_string(std::string_view text) {
    std::string out = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

void print_json() {
    std::cout << "[\n";
    const auto all = presets();
    for (size_t i = 0; i < all.size(); ++i) {
        const Preset& preset = all[i];
        std::cout << "  {\"name\": " << json_string(preset.name)
                  << ", \"purpose\": " << json_string(preset.purpose)
                  << ", \"modern\": " << (preset.modern ? "true" : "false");
        for (const Field& field : kFields) {
            std::cout << ", \"" << field.name << "\": " << field.format(preset);
        }
        std::cout << "}" << (i + 1 < all.size() ? "," : "") << "\n";
    }
    std::cout << "]" << std::endl;
}

// One column per preset, one row per field: five presets fit in 80
// columns where sixteen fields side by side would not.
void print_table() {
    const auto all = presets();
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> header{""};
    std::vector<std::string> arch{"arch"};
    for (const Preset& preset : all) {
        header.emplace_back(preset.name);
        arch.emplace_back(preset.modern ? "modern" : "gpt2");
    }
    rows.push_back(header);
    rows.push_back(arch);
    for (const Field& field : kFields) {
        std::vector<std::string> row{std::string(field.name)};
        for (const Preset& preset : all) row.push_back(field.format(preset));
        rows.push_back(row);
    }

    std::vector<size_t> widths(header.size(), 0);
    for (const auto& row : rows) {
        for (size_t c = 0; c < row.size(); ++c) widths[c] = std::max(widths[c], row[c].size());
    }

    size_t name_width = 0;
    for (const Preset& preset : all) name_width = std::max(name_width, preset.name.size());
    for (const Preset& preset : all) {
        std::cout << "  " << preset.name << std::string(name_width + 2 - preset.name.size(), ' ')
                  << preset.purpose << "\n";
    }
    std::cout << "\n";
    for (const auto& row : rows) {
        std::string line = row[0] + std::string(widths[0] - row[0].size(), ' ');
        for (size_t c = 1; c < row.size(); ++c) {
            line += std::string(widths[c] - row[c].size() + 2, ' ') + row[c];
        }
        std::cout << line << "\n";
    }
    std::cout << "\nEffective batch: batch_size x grad_accum sequences of seq_length tokens.\n"
                 "max_eval_batches 0 evaluates the whole val split.\n"
              << std::flush;
}

}  // namespace

int run_presets(const Invocation& invocation) {
    bool json = false;

    Command cmd(invocation.usage_name(), std::string(invocation.summary));
    cmd.describe("Presets are passed to train by name (train-fast runs 'fast').");
    cmd.flag("--json", json, "print the table as a JSON array, one object per preset");
    if (cmd.parse(invocation.args) == ParseResult::HelpShown) return 0;

    if (json) print_json();
    else print_table();
    return 0;
}

}  // namespace cli

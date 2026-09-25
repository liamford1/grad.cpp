// grad chat: interactive REPL. Type a prompt, watch the model continue it
// token by token. This is a base language model, not an instruction-tuned
// assistant: it continues text in the style of its training corpus rather
// than answering questions.

#include "cli/args.h"
#include "commands.h"
#include "common.h"

#include "transformer/text_gen.h"

#include <iostream>
#include <optional>
#include <string>

namespace cli {

int run_chat(const Invocation& invocation) {
    std::string checkpoint = "shakespeare_final.bin";
    std::string corpus = kDefaultCorpus;
    std::optional<int> vocab;

    Command cmd(invocation.usage_name(), std::string(invocation.summary));
    cmd.describe("Each line you type is continued by the model, streamed as it is sampled. "
                 "An empty line, 'exit', or 'quit' ends the session.");
    cmd.optional("ckpt", checkpoint, "checkpoint to chat with");
    cmd.optional("corpus", corpus, "the corpus the checkpoint was trained on, for its tokenizer");
    cmd.optional("vocab", vocab, "vocab size; must match the checkpoint's")
        .at_least(1)
        .default_text("the checkpoint's");
    if (cmd.parse(invocation.args) == ParseResult::HelpShown) return 0;

    std::cout << "\ngrad.cpp Chat\n" << std::endl;
    const auto [model, tokenizer] = load_for_inference(checkpoint, corpus, vocab);
    TextGen generator(model, &tokenizer);

    std::cout << "\nThis is a base language model: it continues text in the"
              << " style of its training corpus.\nEmpty line or 'exit' quits.\n" << std::endl;

    std::string line;
    while (true) {
        std::cout << "> " << std::flush;
        if (!std::getline(std::cin, line)) break;
        if (line.empty() || line == "exit" || line == "quit") break;

        const auto prompt = tokenizer.encode(line + "\n");
        std::cout << line << std::flush;
        generator.generate_stream(
            prompt, [](const std::string& piece) { std::cout << piece << std::flush; }, 0.8f, 200);
        std::cout << "\n" << std::endl;
    }
    return 0;
}

}  // namespace cli

#include "utils/dashboard.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>

namespace utils {
namespace {

// ---------------------------------------------------------------- palette
constexpr const char* RST = "\033[0m";
constexpr const char* DIM = "\033[38;5;240m";   // borders, axes
constexpr const char* FAINT = "\033[38;5;238m"; // raw loss dots
constexpr const char* CYAN = "\033[38;5;44m";   // loss EMA
constexpr const char* MAG = "\033[38;5;177m";   // validation
constexpr const char* YEL = "\033[38;5;221m";   // recent zoom
constexpr const char* GRN = "\033[38;5;77m";    // good numbers
constexpr const char* WHT = "\033[1;38;5;255m"; // key values
constexpr const char* LBL = "\033[38;5;246m";   // labels

// ------------------------------------------------------------------ data
struct TrainRow { int step; float loss, lr, grad_norm, step_ms, mem_mb; bool has_gn; };
struct EvalRow { int step; float val_loss; };

struct RunData {
    std::vector<TrainRow> train;
    std::vector<EvalRow> evals;
    int total_steps = 0;
    long tokens_per_step = 0;
};

RunData parse_csv(const std::string& path) {
    RunData d;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.size() < 3) continue;
        std::stringstream ss(line.substr(2));
        std::string f[6];
        for (int i = 0; i < 6 && std::getline(ss, f[i], ','); i++) {}
        try {
            if (line[0] == 'm') {
                d.total_steps = std::stoi(f[0]);
                d.tokens_per_step = std::stol(f[1]);
            } else if (line[0] == 't') {
                TrainRow r;
                r.step = std::stoi(f[0]);
                r.loss = std::stof(f[1]);
                r.lr = std::stof(f[2]);
                r.has_gn = !f[3].empty();
                r.grad_norm = r.has_gn ? std::stof(f[3]) : 0.0f;
                r.step_ms = std::stof(f[4]);
                r.mem_mb = std::stof(f[5]);
                d.train.push_back(r);
            } else if (line[0] == 'e') {
                d.evals.push_back({std::stoi(f[0]), std::stof(f[1])});
            }
        } catch (...) {
            // Partial last line while the trainer is mid-write; skip.
        }
    }
    return d;
}

// -------------------------------------------------------- braille canvas
// Each terminal cell holds a 2x4 grid of braille dots, so a WxH cell
// canvas plots at 2W x 4H pixel resolution - the "technical" look.
struct Canvas {
    int W, H;
    std::vector<uint8_t> cells;
    Canvas(int w, int h) : W(w), H(h), cells(static_cast<size_t>(w) * h, 0) {}

    void set(int px, int py) {
        if (px < 0 || py < 0 || px >= W * 2 || py >= H * 4) return;
        static const uint8_t bit[4][2] = {{0x01, 0x08}, {0x02, 0x10}, {0x04, 0x20}, {0x40, 0x80}};
        cells[static_cast<size_t>(py / 4) * W + px / 2] |= bit[py % 4][px % 2];
    }

    // Vertical segment so steep polylines stay connected.
    void vline(int px, int py0, int py1) {
        if (py0 > py1) std::swap(py0, py1);
        for (int y = py0; y <= py1; y++) set(px, y);
    }

    std::string row(int r) const {
        std::string s;
        for (int c = 0; c < W; c++) {
            uint8_t v = cells[static_cast<size_t>(r) * W + c];
            // UTF-8 encode U+2800+v
            unsigned cp = 0x2800 + v;
            s += static_cast<char>(0xE0 | (cp >> 12));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        }
        return s;
    }
};

// Map a (step, value) series onto canvas pixels and draw a connected line.
void plot_line(Canvas& cv, const std::vector<std::pair<float, float>>& pts,
               float x0, float x1, float y0, float y1) {
    if (pts.empty() || x1 <= x0 || y1 <= y0) return;
    const int PW = cv.W * 2, PH = cv.H * 4;
    int prev_px = -1, prev_py = -1;
    for (const auto& p : pts) {
        int px = static_cast<int>((p.first - x0) / (x1 - x0) * (PW - 1) + 0.5f);
        int py = static_cast<int>((1.0f - (p.second - y0) / (y1 - y0)) * (PH - 1) + 0.5f);
        px = std::clamp(px, 0, PW - 1);
        py = std::clamp(py, 0, PH - 1);
        if (prev_px >= 0 && px == prev_px) {
            cv.vline(px, std::min(prev_py, py), std::max(prev_py, py));
        } else if (prev_px >= 0) {
            for (int x = prev_px; x <= px; x++) {
                float t = (px == prev_px) ? 0.0f : float(x - prev_px) / (px - prev_px);
                int y = static_cast<int>(prev_py + t * (py - prev_py) + 0.5f);
                cv.set(x, y);
            }
        } else {
            cv.set(px, py);
        }
        prev_px = px;
        prev_py = py;
    }
}

std::string sparkline(const std::vector<float>& v, int width) {
    static const char* blocks[8] = {"▁", "▂", "▃", "▄",
                                    "▅", "▆", "▇", "█"};
    if (v.empty()) return std::string(width, ' ');
    // Bucket the series into `width` columns.
    float lo = *std::min_element(v.begin(), v.end());
    float hi = *std::max_element(v.begin(), v.end());
    std::string s;
    for (int i = 0; i < width; i++) {
        size_t a = static_cast<size_t>(i) * v.size() / width;
        size_t b = std::max(a + 1, static_cast<size_t>(i + 1) * v.size() / width);
        float m = 0;
        for (size_t j = a; j < b && j < v.size(); j++) m += v[j];
        m /= (b - a);
        int lvl = (hi > lo) ? static_cast<int>((m - lo) / (hi - lo) * 7.99f) : 3;
        s += blocks[std::clamp(lvl, 0, 7)];
    }
    return s;
}

std::string fmt(float v, int prec = 3) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return buf;
}

std::string fmt_eta(double seconds) {
    if (seconds < 0) return "--";
    long h = static_cast<long>(seconds) / 3600;
    long m = (static_cast<long>(seconds) % 3600) / 60;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%ldh%02ldm", h, m);
    return buf;
}

// Visible width of a string that mixes ANSI codes and UTF-8.
size_t visible_width(const std::string& s) {
    size_t w = 0;
    for (size_t i = 0; i < s.size();) {
        if (s[i] == '\033') {
            while (i < s.size() && s[i] != 'm') i++;
            i++;
        } else {
            unsigned char c = s[i];
            i += (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 1;
            w++;
        }
    }
    return w;
}

// A panel: title bar, boxed border, colored body lines.
void emit_panel(std::string& out, const std::string& title, int width,
                const std::vector<std::string>& body) {
    out += DIM;
    out += "┌─ ";
    out += RST;
    out += LBL;
    out += title;
    out += RST;
    out += DIM;
    out += " ";
    int fill = width - 5 - static_cast<int>(visible_width(title));
    for (int i = 0; i < fill; i++) out += "─";
    out += "┐";
    out += RST;
    out += "\r\n";
    for (const auto& line : body) {
        out += DIM;
        out += "│";
        out += RST;
        out += line;
        int pad = width - 2 - static_cast<int>(visible_width(line));
        out.append(std::max(0, pad), ' ');
        out += DIM;
        out += "│";
        out += RST;
        out += "\r\n";
    }
    out += DIM;
    out += "└";
    for (int i = 0; i < width - 2; i++) out += "─";
    out += "┘";
    out += RST;
    out += "\r\n";
}

// ------------------------------------------------------------- the frame
std::string render(const RunData& d, const std::string& name, int tw, int th) {
    std::string out;
    if (d.train.empty()) {
        out += LBL;
        out += "waiting for metrics";
        out += RST;
        out += "\r\n";
        return out;
    }

    const TrainRow& last = d.train.back();
    const int total = d.total_steps > 0 ? d.total_steps : last.step + 1;

    // EMA of loss; alpha tuned so the line is smooth at 40K steps but
    // still tracks turns within a few hundred.
    std::vector<std::pair<float, float>> ema_pts, raw_pts;
    float ema = d.train.front().loss;
    for (const auto& r : d.train) {
        ema = 0.98f * ema + 0.02f * r.loss;
        ema_pts.push_back({static_cast<float>(r.step), ema});
        raw_pts.push_back({static_cast<float>(r.step), r.loss});
    }

    // Recent throughput: median step_ms over the last 50 steps.
    std::vector<float> recent_ms;
    for (size_t i = d.train.size() > 50 ? d.train.size() - 50 : 0; i < d.train.size(); i++)
        recent_ms.push_back(d.train[i].step_ms);
    std::vector<float> ms_sorted = recent_ms;
    std::sort(ms_sorted.begin(), ms_sorted.end());
    float med_ms = ms_sorted.empty() ? 0 : ms_sorted[ms_sorted.size() / 2];
    float tok_s = (med_ms > 0) ? d.tokens_per_step / (med_ms / 1000.0f) : 0;
    double eta_s = (med_ms > 0) ? (total - last.step - 1) * (med_ms / 1000.0) : -1;

    // ------------------------------------------------------ status block
    double done = 100.0 * (last.step + 1) / total;
    int barw = std::max(10, tw - 64);
    int fillw = static_cast<int>(barw * done / 100.0 + 0.5);
    std::string bar;
    for (int i = 0; i < barw; i++) bar += (i < fillw) ? "█" : "░";

    out += WHT;
    out += " TRANSFORMER ";
    out += RST;
    out += LBL;
    out += name;
    out += RST;
    out += "   ";
    out += DIM;
    out += "step ";
    out += RST;
    out += WHT;
    out += std::to_string(last.step + 1);
    out += RST;
    out += DIM;
    out += "/" + std::to_string(total);
    out += RST;
    out += "  ";
    out += GRN;
    out += bar;
    out += RST;
    out += " ";
    out += WHT;
    out += fmt(done, 1) + "%";
    out += RST;
    out += "  ";
    out += DIM;
    out += "eta ";
    out += RST;
    out += WHT;
    out += fmt_eta(eta_s);
    out += RST;
    out += "\r\n";

    char stat[256];
    std::snprintf(stat, sizeof(stat),
                  " %sloss%s %s%.4f%s  %slr%s %.2e  %stok/s%s %.0f  %sstep%s %.1fs  %smem%s %.0fMB",
                  DIM, RST, WHT, ema, RST, DIM, RST, last.lr,
                  DIM, RST, tok_s, DIM, RST, med_ms / 1000.0f, DIM, RST, last.mem_mb);
    out += stat;
    out += "\r\n";

    // -------------------------------------------------------- loss panel
    const int lossw = tw;
    const int lossh = std::max(6, (th - 13) * 3 / 5);
    {
        Canvas cv(lossw - 2, lossh);
        float ymin = 1e30f, ymax = -1e30f;
        for (auto& p : ema_pts) { ymin = std::min(ymin, p.second); ymax = std::max(ymax, p.second); }
        for (auto& p : raw_pts) ymax = std::max(ymax, p.second);
        float pad = 0.05f * (ymax - ymin + 1e-6f);
        ymin -= pad; ymax += pad;
        float x0 = raw_pts.front().first, x1 = std::max(raw_pts.back().first, x0 + 1);

        // Raw loss as sparse dots (subsampled), EMA as the solid line.
        Canvas raw_cv(lossw - 2, lossh);
        size_t stride = std::max<size_t>(1, raw_pts.size() / (cv.W * 2));
        std::vector<std::pair<float, float>> raw_sub;
        for (size_t i = 0; i < raw_pts.size(); i += stride) raw_sub.push_back(raw_pts[i]);
        for (auto& p : raw_sub) {
            int px = static_cast<int>((p.first - x0) / (x1 - x0) * (raw_cv.W * 2 - 1));
            int py = static_cast<int>((1.0f - (p.second - ymin) / (ymax - ymin)) * (raw_cv.H * 4 - 1));
            raw_cv.set(px, py);
        }
        plot_line(cv, ema_pts, x0, x1, ymin, ymax);

        // Composite the two layers cell by cell: the EMA line (cyan)
        // wins over the raw dots (faint) where both are present.
        std::vector<std::string> body(lossh);
        for (int r = 0; r < lossh; r++) {
            std::string line;
            for (int c = 0; c < cv.W; c++) {
                uint8_t e = cv.cells[static_cast<size_t>(r) * cv.W + c];
                uint8_t w = raw_cv.cells[static_cast<size_t>(r) * raw_cv.W + c];
                unsigned cp = 0x2800 + (e ? e : w);
                std::string ch;
                ch += static_cast<char>(0xE0 | (cp >> 12));
                ch += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                ch += static_cast<char>(0x80 | (cp & 0x3F));
                if (e) { line += CYAN; line += ch; line += RST; }
                else if (w) { line += FAINT; line += ch; line += RST; }
                else line += " ";
            }
            body[r] = line;
        }
        emit_panel(out, "train loss  [" + fmt(ymin + pad, 3) + " .. " + fmt(ymax - pad, 3) +
                            "]  raw " + FAINT + "⣿" + RST + LBL + "  ema " + CYAN + "⣿" + RST,
                   lossw, body);
    }

    // ------------------------------------- validation + recent loss row
    const int half = tw / 2;
    const int vh = std::max(4, th - 13 - lossh - 1);
    std::vector<std::string> lbody, rbody;
    {
        // Left: validation loss curve.
        Canvas cv(half - 2, vh);
        if (d.evals.size() >= 2) {
            std::vector<std::pair<float, float>> pts;
            float ymin = 1e30f, ymax = -1e30f;
            for (const auto& e : d.evals) {
                pts.push_back({static_cast<float>(e.step), e.val_loss});
                ymin = std::min(ymin, e.val_loss);
                ymax = std::max(ymax, e.val_loss);
            }
            float pad = 0.08f * (ymax - ymin + 1e-6f);
            plot_line(cv, pts, pts.front().first, std::max(pts.back().first, pts.front().first + 1),
                      ymin - pad, ymax + pad);
        }
        for (int r = 0; r < vh; r++) {
            lbody.push_back(std::string(MAG) + cv.row(r) + RST);
        }
        std::string title = "val loss";
        if (!d.evals.empty()) {
            auto best = *std::min_element(d.evals.begin(), d.evals.end(),
                                          [](auto& a, auto& b) { return a.val_loss < b.val_loss; });
            title += "  last " + fmt(d.evals.back().val_loss) + "  best " + fmt(best.val_loss) +
                     " (ppl " + fmt(std::exp(best.val_loss), 1) + ") @" + std::to_string(best.step);
        } else {
            title += "  (first eval pending)";
        }
        std::string panel;
        emit_panel(panel, title, half, lbody);
        // Right: recent loss zoom (last quarter or 1000 steps).
        Canvas rc(tw - half - 2, vh);
        size_t nrecent = std::min<size_t>(std::max<size_t>(200, d.train.size() / 4), 1000);
        size_t start = d.train.size() > nrecent ? d.train.size() - nrecent : 0;
        std::vector<std::pair<float, float>> rpts;
        float rmin = 1e30f, rmax = -1e30f;
        for (size_t i = start; i < d.train.size(); i++) {
            rpts.push_back({static_cast<float>(d.train[i].step), d.train[i].loss});
            rmin = std::min(rmin, d.train[i].loss);
            rmax = std::max(rmax, d.train[i].loss);
        }
        if (rpts.size() >= 2) {
            float pad = 0.08f * (rmax - rmin + 1e-6f);
            plot_line(rc, rpts, rpts.front().first, rpts.back().first, rmin - pad, rmax + pad);
        }
        for (int r = 0; r < vh; r++) rbody.push_back(std::string(YEL) + rc.row(r) + RST);
        std::string rpanel;
        emit_panel(rpanel, "recent loss  last " + std::to_string(rpts.size()) + " steps  [" +
                               fmt(rmin) + " .. " + fmt(rmax) + "]",
                   tw - half, rbody);

        // Stitch the two panels side by side.
        std::istringstream ls(panel), rs(rpanel);
        std::string a, b;
        while (std::getline(ls, a) && std::getline(rs, b)) {
            if (!a.empty() && a.back() == '\r') a.pop_back();
            if (!b.empty() && b.back() == '\r') b.pop_back();
            out += a + b + "\r\n";
        }
    }

    // --------------------------------------------------------- sparkline
    std::vector<float> gn, ms_series;
    for (const auto& r : d.train) {
        if (r.has_gn) gn.push_back(r.grad_norm);
        ms_series.push_back(r.step_ms);
    }
    int sw = (tw - 30) / 2;
    out += " ";
    out += DIM;
    out += "grad‖g‖ ";
    out += RST;
    out += LBL;
    out += sparkline(gn, sw);
    out += RST;
    out += "  ";
    out += DIM;
    out += "step-time ";
    out += RST;
    out += LBL;
    out += sparkline(ms_series, sw);
    out += RST;
    out += "\r\n";
    out += DIM;
    out += " q quit · refreshes 1s · " + std::to_string(d.train.size()) + " steps logged";
    out += RST;
    out += "\r\n";
    return out;
}

// ----------------------------------------------------------- tty control
volatile std::sig_atomic_t g_dash_stop = 0;
void dash_sigint(int) { g_dash_stop = 1; }

struct RawTerm {
    termios saved{};
    bool active = false;
    RawTerm() {
        if (!isatty(STDIN_FILENO)) return;
        tcgetattr(STDIN_FILENO, &saved);
        termios raw = saved;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
        active = true;
    }
    ~RawTerm() {
        if (active) tcsetattr(STDIN_FILENO, TCSANOW, &saved);
    }
};

void term_size(int& w, int& h) {
    winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 20) {
        w = ws.ws_col;
        h = ws.ws_row;
    } else {
        w = 110;
        h = 32;
    }
}

}  // namespace

std::string newest_metrics_csv(const std::string& dir) {
    DIR* d = opendir(dir.c_str());
    if (!d) return "";
    std::string best;
    time_t best_mtime = 0;
    const std::string suffix = "_metrics.csv";
    while (dirent* e = readdir(d)) {
        std::string n = e->d_name;
        if (n.size() <= suffix.size() ||
            n.compare(n.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
        struct stat st{};
        std::string path = dir + "/" + n;
        if (stat(path.c_str(), &st) == 0 && st.st_mtime >= best_mtime) {
            best_mtime = st.st_mtime;
            best = path;
        }
    }
    closedir(d);
    return best;
}

int run_dashboard(const std::string& csv_path, bool once) {
    // Derive a display name from the file: tinystories_modern_metrics.csv
    // -> tinystories_modern.
    std::string name = csv_path;
    size_t slash = name.find_last_of('/');
    if (slash != std::string::npos) name = name.substr(slash + 1);
    size_t suffix = name.rfind("_metrics.csv");
    if (suffix != std::string::npos) name = name.substr(0, suffix);

    if (once) {
        int w, h;
        term_size(w, h);
        RunData d = parse_csv(csv_path);
        std::string frame = render(d, name, w, h);
        fwrite(frame.data(), 1, frame.size(), stdout);
        return 0;
    }

    g_dash_stop = 0;
    auto prev = std::signal(SIGINT, dash_sigint);
    RawTerm raw;
    fputs("\033[?1049h\033[?25l", stdout);  // alt screen, hide cursor

    while (!g_dash_stop) {
        int w, h;
        term_size(w, h);
        RunData d = parse_csv(csv_path);
        std::string frame = "\033[H\033[2J" + render(d, name, w, h);
        fwrite(frame.data(), 1, frame.size(), stdout);
        fflush(stdout);

        for (int i = 0; i < 10 && !g_dash_stop; i++) {
            char c;
            if (read(STDIN_FILENO, &c, 1) == 1 && (c == 'q' || c == 'Q')) g_dash_stop = 1;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    fputs("\033[?1049l\033[?25h", stdout);  // restore screen and cursor
    fflush(stdout);
    std::signal(SIGINT, prev);
    return 0;
}

}  // namespace utils

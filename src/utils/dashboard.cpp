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
constexpr const char* YEL = "\033[38;5;221m";   // recent zoom / grad norm
constexpr const char* GRN = "\033[38;5;77m";    // throughput, progress
constexpr const char* ORN = "\033[38;5;209m";   // learning rate
constexpr const char* RED = "\033[38;5;203m";   // clip threshold
constexpr const char* WHT = "\033[1;38;5;255m"; // key values
constexpr const char* LBL = "\033[38;5;246m";   // labels

constexpr float kClipNorm = 5.0f;  // optimizer's clip_grad_norm threshold

// ------------------------------------------------------------------ data
struct TrainRow { int step; float loss, lr, grad_norm, step_ms, mem_mb, wall_s; };
struct EvalRow { int step; float val_loss; };

struct RunData {
    std::vector<TrainRow> train;
    std::vector<EvalRow> evals;
    int total_steps = 0;
    long tokens_per_step = 0;
    long params = 0;
    std::string desc;
    double elapsed_s = 0;  // summed across resume segments
};

RunData parse_csv(const std::string& path) {
    RunData d;
    std::ifstream in(path);
    std::string line;
    double seg_base = 0, prev_wall = 0;
    while (std::getline(in, line)) {
        if (line.size() < 3) continue;
        std::stringstream ss(line.substr(2));
        std::string f[8];
        int n = 0;
        while (n < 8 && std::getline(ss, f[n], ',')) n++;
        try {
            if (line[0] == 'm' && n >= 2) {
                d.total_steps = std::stoi(f[0]);
                d.tokens_per_step = std::stol(f[1]);
                if (n >= 3) d.params = std::stol(f[2]);
                if (n >= 4) d.desc = f[3];
            } else if (line[0] == 't' && n >= 6) {
                TrainRow r{};
                r.step = std::stoi(f[0]);
                r.loss = std::stof(f[1]);
                r.lr = std::stof(f[2]);
                r.grad_norm = f[3].empty() ? 0.0f : std::stof(f[3]);
                r.step_ms = std::stof(f[4]);
                r.mem_mb = std::stof(f[5]);
                r.wall_s = (n >= 7) ? std::stof(f[6]) : 0.0f;
                // wall_s restarts on resume; fold segments into one clock.
                if (r.wall_s < prev_wall) seg_base += prev_wall;
                prev_wall = r.wall_s;
                d.train.push_back(r);
            } else if (line[0] == 'e' && n >= 2) {
                d.evals.push_back({std::stoi(f[0]), std::stof(f[1])});
            }
        } catch (...) {
            // Partial last line while the trainer is mid-write; skip.
        }
    }
    d.elapsed_s = seg_base + prev_wall;
    return d;
}

// -------------------------------------------------------- braille canvas
// Each terminal cell holds a 2x4 grid of braille dots, so a WxH cell
// canvas plots at 2W x 4H pixel resolution.
struct Canvas {
    int W, H;
    std::vector<uint8_t> cells;
    Canvas(int w, int h) : W(std::max(1, w)), H(std::max(1, h)),
                           cells(static_cast<size_t>(W) * H, 0) {}

    void set(int px, int py) {
        if (px < 0 || py < 0 || px >= W * 2 || py >= H * 4) return;
        static const uint8_t bit[4][2] = {{0x01, 0x08}, {0x02, 0x10}, {0x04, 0x20}, {0x40, 0x80}};
        cells[static_cast<size_t>(py / 4) * W + px / 2] |= bit[py % 4][px % 2];
    }

    void vline(int px, int py0, int py1) {
        if (py0 > py1) std::swap(py0, py1);
        for (int y = py0; y <= py1; y++) set(px, y);
    }
};

std::string braille_utf8(uint8_t v) {
    unsigned cp = 0x2800 + v;
    std::string s;
    s += static_cast<char>(0xE0 | (cp >> 12));
    s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (cp & 0x3F));
    return s;
}

struct Series {
    std::vector<std::pair<float, float>> pts;  // (x, y)
    const char* color;
    bool scatter = false;  // dots only, no connecting line
};

struct Range { float x0, x1, y0, y1; };

Range fit_range(const std::vector<Series>& layers, float ypad_frac) {
    Range r{1e30f, -1e30f, 1e30f, -1e30f};
    for (const auto& s : layers) {
        for (const auto& p : s.pts) {
            r.x0 = std::min(r.x0, p.first);
            r.x1 = std::max(r.x1, p.first);
            r.y0 = std::min(r.y0, p.second);
            r.y1 = std::max(r.y1, p.second);
        }
    }
    if (r.x1 <= r.x0) r.x1 = r.x0 + 1;
    float pad = ypad_frac * (r.y1 - r.y0 + 1e-6f);
    r.y0 -= pad;
    r.y1 += pad;
    return r;
}

void draw_series(Canvas& cv, const Series& s, const Range& r) {
    const int PW = cv.W * 2, PH = cv.H * 4;
    int ppx = -1, ppy = -1;
    for (const auto& p : s.pts) {
        int px = std::clamp(static_cast<int>((p.first - r.x0) / (r.x1 - r.x0) * (PW - 1) + 0.5f), 0, PW - 1);
        int py = std::clamp(static_cast<int>((1.0f - (p.second - r.y0) / (r.y1 - r.y0)) * (PH - 1) + 0.5f), 0, PH - 1);
        if (s.scatter || ppx < 0) {
            cv.set(px, py);
        } else if (px == ppx) {
            cv.vline(px, std::min(ppy, py), std::max(ppy, py));
        } else {
            for (int x = ppx; x <= px; x++) {
                float t = float(x - ppx) / (px - ppx);
                cv.set(x, static_cast<int>(ppy + t * (py - ppy) + 0.5f));
            }
        }
        ppx = px;
        ppy = py;
    }
}

// Renders layered series into colored text rows (earlier layers win).
std::vector<std::string> chart_rows(int w, int h, const std::vector<Series>& layers,
                                    const Range& r, float hline = -1,
                                    const char* hline_color = RED) {
    std::vector<Canvas> cvs;
    for (const auto& s : layers) {
        cvs.emplace_back(w, h);
        draw_series(cvs.back(), s, r);
    }
    Canvas hcv(w, h);
    if (hline > r.y0 && hline < r.y1) {
        int py = static_cast<int>((1.0f - (hline - r.y0) / (r.y1 - r.y0)) * (h * 4 - 1) + 0.5f);
        for (int px = 0; px < w * 2; px += 3) hcv.set(px, py);  // dashed
    }

    std::vector<std::string> rows(h);
    for (int rr = 0; rr < h; rr++) {
        std::string line;
        for (int c = 0; c < w; c++) {
            uint8_t v = 0;
            const char* color = nullptr;
            for (size_t l = 0; l < layers.size(); l++) {
                uint8_t cell = cvs[l].cells[static_cast<size_t>(rr) * w + c];
                if (cell) { v = cell; color = layers[l].color; break; }
            }
            if (!v) {
                uint8_t hc = hcv.cells[static_cast<size_t>(rr) * w + c];
                if (hc) { v = hc; color = hline_color; }
            }
            if (v) {
                line += color;
                line += braille_utf8(v);
                line += RST;
            } else {
                line += " ";
            }
        }
        rows[rr] = line;
    }
    return rows;
}

std::string fmt(float v, int prec = 3) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return buf;
}

std::string fmt_sci(float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2e", v);
    return buf;
}

std::string fmt_count(double v) {
    char buf[32];
    if (v >= 1e9) std::snprintf(buf, sizeof(buf), "%.2fB", v / 1e9);
    else if (v >= 1e6) std::snprintf(buf, sizeof(buf), "%.1fM", v / 1e6);
    else if (v >= 1e3) std::snprintf(buf, sizeof(buf), "%.1fk", v / 1e3);
    else std::snprintf(buf, sizeof(buf), "%.0f", v);
    return buf;
}

std::string fmt_dur(double seconds) {
    if (seconds < 0) return "--";
    long h = static_cast<long>(seconds) / 3600;
    long m = (static_cast<long>(seconds) % 3600) / 60;
    char buf[32];
    if (h > 0) std::snprintf(buf, sizeof(buf), "%ldh%02ldm", h, m);
    else std::snprintf(buf, sizeof(buf), "%ldm%02lds", m, static_cast<long>(seconds) % 60);
    return buf;
}

// Visible width of a string mixing ANSI codes and UTF-8.
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

std::string panel(const std::string& title, int width, const std::vector<std::string>& body) {
    std::string out;
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
    return out;
}

// Join multiple panels side by side, line by line.
std::string beside(const std::vector<std::string>& panels) {
    std::vector<std::vector<std::string>> lines(panels.size());
    size_t rows = 0;
    for (size_t i = 0; i < panels.size(); i++) {
        std::istringstream ss(panels[i]);
        std::string l;
        while (std::getline(ss, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            lines[i].push_back(l);
        }
        rows = std::max(rows, lines[i].size());
    }
    std::string out;
    for (size_t r = 0; r < rows; r++) {
        for (size_t i = 0; i < panels.size(); i++) {
            if (r < lines[i].size()) out += lines[i][r];
        }
        out += "\r\n";
    }
    return out;
}

std::string kv(const char* k, const std::string& v, const char* vc = WHT) {
    return std::string(DIM) + k + " " + RST + vc + v + RST;
}

// ------------------------------------------------------------- the frame
std::string render(const RunData& d, const std::string& name, int tw, int th) {
    if (d.train.empty()) {
        return std::string(LBL) + "waiting for metrics..." + RST + "\r\n";
    }

    const TrainRow& last = d.train.back();
    const int total = d.total_steps > 0 ? d.total_steps : last.step + 1;
    const size_t N = d.train.size();

    // Smoothed series.
    std::vector<std::pair<float, float>> ema_pts, raw_pts, lr_pts, gn_pts, tok_pts;
    float ema = d.train.front().loss;
    float tok_ema = 0;
    int clipped = 0;
    for (const auto& r : d.train) {
        ema = 0.98f * ema + 0.02f * r.loss;
        float x = static_cast<float>(r.step);
        ema_pts.push_back({x, ema});
        raw_pts.push_back({x, r.loss});
        lr_pts.push_back({x, r.lr});
        if (r.grad_norm > 0) gn_pts.push_back({x, r.grad_norm});
        if (r.grad_norm > kClipNorm) clipped++;
        if (r.step_ms > 0) {
            float ts = d.tokens_per_step / (r.step_ms / 1000.0f);
            tok_ema = tok_ema == 0 ? ts : 0.95f * tok_ema + 0.05f * ts;
            tok_pts.push_back({x, tok_ema});
        }
    }
    float clip_pct = gn_pts.empty() ? 0 : 100.0f * clipped / gn_pts.size();

    // Throughput and ETA from the median of recent step times.
    std::vector<float> recent_ms;
    for (size_t i = N > 50 ? N - 50 : 0; i < N; i++) recent_ms.push_back(d.train[i].step_ms);
    std::sort(recent_ms.begin(), recent_ms.end());
    float med_ms = recent_ms.empty() ? 0 : recent_ms[recent_ms.size() / 2];
    float tok_s = (med_ms > 0) ? d.tokens_per_step / (med_ms / 1000.0f) : 0;
    double eta_s = (med_ms > 0) ? (total - last.step - 1) * (med_ms / 1000.0) : -1;
    double tflops = 6.0 * d.params * tok_s / 1e12;  // fwd+bwd ~ 6*N per token

    const EvalRow* best = nullptr;
    for (const auto& e : d.evals) {
        if (!best || e.val_loss < best->val_loss) best = &e;
    }

    // ------------------------------------------------------------ header
    std::string out;
    double done = 100.0 * (last.step + 1) / total;
    {
        std::string left = std::string(WHT) + " TRANSFORMER" + RST + DIM + " ▮ " + RST +
                           LBL + name + RST;
        if (!d.desc.empty()) left += std::string(DIM) + " · " + d.desc + RST;
        if (d.params > 0) left += std::string(DIM) + " · " + fmt_count(double(d.params)) +
                                  " params" + RST;
        out += left + "\r\n";

        std::string right = " " + kv("step", std::to_string(last.step + 1) +
                                     std::string(DIM) + "/" + std::to_string(total) + RST) +
                            "  " + kv("eta", fmt_dur(eta_s)) +
                            "  " + kv("elapsed", fmt_dur(d.elapsed_s), LBL) + "  ";
        int barw = std::max(10, tw - static_cast<int>(visible_width(right)) - 10);
        int fillw = static_cast<int>(barw * done / 100.0 + 0.5);
        std::string bar;
        for (int i = 0; i < barw; i++) bar += (i < fillw) ? "█" : "░";
        out += right + GRN + bar + RST + " " + WHT + fmt(done, 1) + "%" + RST + "\r\n";
    }

    // ------------------------------------------------------- stats strip
    {
        std::string l1 = " " + kv("loss", fmt(ema, 4), CYAN) +
                         "  " + kv("ppl", fmt(std::exp(ema), 2), CYAN);
        if (!d.evals.empty()) {
            l1 += "  " + kv("val", fmt(d.evals.back().val_loss, 4), MAG) +
                  "  " + kv("val-ppl", fmt(std::exp(d.evals.back().val_loss), 2), MAG);
            if (best) {
                l1 += std::string(DIM) + "  best " + RST + GRN + fmt(best->val_loss, 4) + RST +
                      DIM + " @" + std::to_string(best->step) + RST;
            }
        }
        l1 += "  " + kv("lr", fmt_sci(last.lr), ORN);
        out += l1 + "\r\n";

        double tokens_seen = double(last.step + 1) * d.tokens_per_step;
        double tokens_total = double(total) * d.tokens_per_step;
        std::string l2 = " " + kv("tok/s", fmt_count(tok_s), GRN) +
                         "  " + kv("TFLOP/s", fmt(tflops, 2), GRN) +
                         "  " + kv("tokens", fmt_count(tokens_seen) + std::string(DIM) + "/" +
                                             fmt_count(tokens_total) + RST) +
                         "  " + kv("step", fmt(med_ms / 1000.0f, 1) + "s") +
                         "  " + kv("‖g‖", gn_pts.empty() ? "-" : fmt(gn_pts.back().second, 2), YEL) +
                         "  " + kv("clip", fmt(clip_pct, 1) + "%", clip_pct > 20 ? RED : LBL) +
                         "  " + kv("mem", fmt(last.mem_mb / 1024.0f, 1) + "GB");
        out += l2 + "\r\n";
    }

    // ------------------------------------------------------------ layout
    const int th_avail = std::max(16, th - 6);
    const int main_h = std::max(9, th_avail * 3 / 5);   // incl. borders
    const int bot_h = std::max(5, th_avail - main_h);   // incl. borders
    const int left_w = tw * 58 / 100;
    const int right_w = tw - left_w;

    // ------------------------------------------------- main loss chart
    std::string left_panel;
    {
        // Raw loss subsampled to the pixel budget as scatter; EMA and the
        // validation track drawn as lines on the same scale.
        std::vector<std::pair<float, float>> raw_sub;
        size_t stride = std::max<size_t>(1, raw_pts.size() / ((left_w - 2) * 2));
        for (size_t i = 0; i < raw_pts.size(); i += stride) raw_sub.push_back(raw_pts[i]);
        std::vector<std::pair<float, float>> val_pts;
        for (const auto& e : d.evals) val_pts.push_back({static_cast<float>(e.step), e.val_loss});

        std::vector<Series> layers;
        layers.push_back({val_pts, MAG, false});
        layers.push_back({ema_pts, CYAN, false});
        layers.push_back({raw_sub, FAINT, true});
        Range r = fit_range(layers, 0.05f);
        auto rows = chart_rows(left_w - 2, main_h - 2, layers, r);
        std::string title = "loss  " + std::string(CYAN) + "⣿" + RST + LBL + " train-ema  " +
                            FAINT + "⣿" + RST + LBL + " raw  " + MAG + "⣿" + RST + LBL +
                            " val   [" + fmt(r.y0, 3) + " .. " + fmt(r.y1, 3) + "]";
        left_panel = panel(title, left_w, rows);
    }

    // ------------------------------------------- right column: 3 charts
    std::string right_panel;
    {
        int h1 = main_h / 3, h2 = main_h / 3;
        int h3 = main_h - h1 - h2;
        // Validation perplexity.
        std::vector<std::pair<float, float>> ppl_pts;
        for (const auto& e : d.evals)
            ppl_pts.push_back({static_cast<float>(e.step), std::exp(e.val_loss)});
        std::string t1 = "val perplexity";
        if (!ppl_pts.empty()) {
            t1 += "  last " + fmt(ppl_pts.back().second, 1);
            if (best) t1 += "  best " + fmt(std::exp(best->val_loss), 1);
        } else {
            t1 += "  (first eval pending)";
        }
        std::vector<Series> l1{{ppl_pts, MAG, ppl_pts.size() < 2}};
        Range r1 = fit_range(l1, 0.08f);
        std::string p1 = panel(t1, right_w, chart_rows(right_w - 2, std::max(1, h1 - 2), l1, r1));

        // Learning-rate schedule (realized).
        std::vector<Series> l2{{lr_pts, ORN, false}};
        Range r2 = fit_range(l2, 0.08f);
        std::string p2 = panel("learning rate  " + fmt_sci(last.lr), right_w,
                               chart_rows(right_w - 2, std::max(1, h2 - 2), l2, r2));

        // Gradient norm with the clip threshold as a dashed line.
        std::vector<Series> l3{{gn_pts, YEL, false}};
        Range r3 = fit_range(l3, 0.08f);
        std::string t3 = "grad norm  clip@" + fmt(kClipNorm, 0) + " " +
                         std::string(RED) + "┄" + RST + LBL + " " + fmt(clip_pct, 1) + "% clipped";
        std::string p3 = panel(t3, right_w,
                               chart_rows(right_w - 2, std::max(1, h3 - 2), l3, r3, kClipNorm));

        right_panel = p1 + p2 + p3;
    }

    out += beside({left_panel, right_panel});

    // ------------------------------------------------------- bottom row
    {
        int half = tw / 2;
        // Recent loss zoom.
        size_t nrecent = std::min<size_t>(std::max<size_t>(200, N / 4), 1000);
        size_t start = N > nrecent ? N - nrecent : 0;
        std::vector<std::pair<float, float>> rpts;
        for (size_t i = start; i < N; i++)
            rpts.push_back({static_cast<float>(d.train[i].step), d.train[i].loss});
        std::vector<Series> lz{{rpts, YEL, rpts.size() < 2}};
        Range rz = fit_range(lz, 0.08f);
        std::string pz = panel("recent loss  last " + std::to_string(rpts.size()) + " steps  [" +
                                   fmt(rz.y0) + " .. " + fmt(rz.y1) + "]",
                               half, chart_rows(half - 2, std::max(1, bot_h - 2), lz, rz));

        // Throughput.
        std::vector<Series> lt{{tok_pts, GRN, tok_pts.size() < 2}};
        Range rt = fit_range(lt, 0.08f);
        std::string pt = panel("throughput tok/s  now " + fmt_count(tok_s),
                               tw - half, chart_rows(tw - half - 2, std::max(1, bot_h - 2), lt, rt));
        out += beside({pz, pt});
    }

    // ------------------------------------------------------ events feed
    {
        std::string ev = std::string(DIM) + " evals " + RST;
        float running_best = 1e30f;
        std::vector<std::string> items;
        for (const auto& e : d.evals) {
            bool star = e.val_loss < running_best;
            running_best = std::min(running_best, e.val_loss);
            items.push_back(std::string(LBL) + std::to_string(e.step) + RST + DIM + "→" + RST +
                            (star ? GRN : LBL) + fmt(e.val_loss, 4) + (star ? "★" : "") + RST);
        }
        size_t show = std::min<size_t>(items.size(), 5);
        for (size_t i = items.size() - show; i < items.size(); i++) ev += items[i] + "  ";
        if (items.empty()) ev += std::string(DIM) + "(none yet)" + RST;
        out += ev + "\r\n";
        out += std::string(DIM) + " q quit · refresh 1s · " + std::to_string(N) +
               " steps logged" + RST + "\r\n";
    }
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
        h = 34;
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
    // tinystories_modern_metrics.csv -> tinystories_modern
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

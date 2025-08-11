#include <fftw3.h>
#include <ncnn/net.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

/*-----------------------------------------------------------------
 *   显式类型别名
 *----------------------------------------------------------------*/
typedef std::vector<double> VecD;
typedef std::vector<float> VecF;
typedef std::vector<int> VecI;
typedef std::vector<std::vector<double> > MatD;

/*-----------------------------------------------------------------
 *                工具函数：trimmed mean
 *----------------------------------------------------------------*/
static double trimmed_mean(const VecD &v) {
    if (v.empty()) return 0.0;
    VecD tmp = v;
    std::sort(tmp.begin(), tmp.end());
    std::size_t k = static_cast<std::size_t>(tmp.size() * 0.1);
    double sum = 0.0;
    for (std::size_t i = k; i < tmp.size() - k; ++i) sum += tmp[i];
    return sum / static_cast<double>(tmp.size() - 2 * k);
}

/*-----------------------------------------------------------------
 *                FFT：返回 abs 频谱前半段
 *----------------------------------------------------------------*/
static VecD fft_half_abs(const VecD &signal) {
    int N = static_cast<int>(signal.size());
    int halfN = N / 2;
    fftw_complex *in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * N);

    for (int i = 0; i < N; ++i) {
        in[i][0] = signal[i];
        in[i][1] = 0.0;
    }
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    VecD result(halfN, 0.0);
    for (int i = 0; i < halfN; ++i) {
        double re = out[i][0], im = out[i][1];
        result[i] = std::sqrt(re * re + im * im);
    }
    if (!result.empty()) result[0] /= 306.7;

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    return result;
}

/*-----------------------------------------------------------------
 *            文件加载：读取四通道 16 进制数
 *----------------------------------------------------------------*/
static MatD load_data(const std::string &filepath) {
    std::ifstream fin(filepath.c_str());
    std::string line;
    MatD data;

    while (std::getline(fin, line)) {
        std::vector<std::string> token;
        std::string cur;
        for (char ch: line) {
            if (ch == ',') {
                token.push_back(cur);
                cur.clear();
            } else
                cur.push_back(ch);
        }
        token.push_back(cur);

        int idx = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0;
        try {
            if (token.size() >= 6 && token[0] == "@01") {
                idx = std::stoi(token[1], 0, 16);
                d1 = std::stoi(token[2], 0, 16);
                d2 = std::stoi(token[3], 0, 16);
                d3 = std::stoi(token[4], 0, 16);
                d4 = std::stoi(token[5], 0, 16);
            } else if (token.size() >= 8 && token[2] == "@01") {
                idx = std::stoi(token[3], 0, 16);
                d1 = std::stoi(token[4], 0, 16);
                d2 = std::stoi(token[5], 0, 16);
                d3 = std::stoi(token[6], 0, 16);
                d4 = std::stoi(token[7], 0, 16);
            } else
                continue;
        }
        catch (...) {
            continue;
        }

        std::vector<double> row(4);
        row[0] = static_cast<double>(d1);
        row[1] = static_cast<double>(d2);
        row[2] = static_cast<double>(d3);
        row[3] = static_cast<double>(d4);
        data.push_back(row);
    }
    return data;
}

/*-----------------------------------------------------------------
 *                       SignalProcessor (ncnn)
 *----------------------------------------------------------------*/
class SignalProcessor {
public:
    SignalProcessor(const std::string &param_path,
                    const std::string &bin_path,
                    int window_size = 128,
                    int window_size_ori = 256,
                    bool enhanced_mode = true,
                    int shift_val = 0)
            : ws(window_size), ws_ori(window_size_ori), enhanced(enhanced_mode), shift_value(shift_val) {
        /* -------- 加载 ncnn 网络 -------- */
        net.opt.use_vulkan_compute = false;     // 如需 GPU 可改 true
        if (net.load_param(param_path.c_str()) ||
            net.load_model(bin_path.c_str())) {
            throw std::runtime_error("load ncnn model failed");
        }

        coef = 128.0 / static_cast<double>(ws);
        start_freq = 2;
        mid_freq = 25;
        queue_len = (ws == 128) ? 30 : 15;
        queue_len_ori = 15;
        alarm_threshold = 3; /* sensitivity=1 */
    }

    /*---------------------------------------------------------
     *   核心：读取 log → 推理 → 生成四个数组
     *--------------------------------------------------------*/
    void process_signal(const std::string &filepath,
                        int saver_pos,
                        VecD &plot_datas,
                        VecD &plot_datas_ori,
                        VecD &saver,
                        double max_sums[2]) {
        /* ---------- 读数据 ---------- */
        MatD raw = load_data(filepath);
        if (shift_value > 0 && raw.size() > static_cast<std::size_t>(shift_value))
            raw.erase(raw.begin(), raw.begin() + shift_value);

        /* ---------- 基础均值 ---------- */
        double basic_avg[3] = {0.0, 0.0, 0.0};
        for (int ch = 0; ch < 3; ++ch) {
            VecD column;
            column.reserve(raw.size());
            for (auto &r: raw) column.push_back(r[ch]);
            basic_avg[ch] = trimmed_mean(column);
        }

        /* 队列 */
        VecD score_q, score_q_ori;

        /* 状态 / 统计量 */
        int status = -1;
        int alert_start_idx = -1;
        double max_sum = 0.0;
        double overall_max = 0.0;
        double overall_max_ori = 0.0;

        /* ---------------- 滑窗 ---------------- */
        for (std::size_t anchor = 1; anchor <= raw.size(); ++anchor) {
            /* ===== ws 分支 ===== */
            if (anchor >= static_cast<std::size_t>(ws) && anchor % ws == 0) {
                /* 取窗口 */
                MatD proc(ws, std::vector<double>(3, 0.0));
                for (int i = 0; i < ws; ++i)
                    for (int j = 0; j < 3; ++j)
                        proc[i][j] = raw[anchor - ws + i][j];

                /* 近期均值并更新基础均值 */
                double recent_avg[3] = {0.0};
                for (int ch = 0; ch < 3; ++ch) {
                    VecD col(ws, 0.0);
                    for (int i = 0; i < ws; ++i) col[i] = proc[i][ch];
                    recent_avg[ch] = trimmed_mean(col);
                    basic_avg[ch] = basic_avg[ch] * 0.9 + recent_avg[ch] * 0.1;
                }

                /* contr1 / contr2 */
                double contr1 = 0.0, contr2 = 0.0;
                if (enhanced) {
                    VecD sig1(ws, 0.0), sig2(ws, 0.0), sig3(ws, 0.0);
                    for (int i = 0; i < ws; ++i) {
                        sig1[i] = proc[i][0];
                        sig2[i] = proc[i][1];
                        sig3[i] = proc[i][2];
                    }
                    VecD f1 = fft_half_abs(sig1);
                    VecD f2 = fft_half_abs(sig2);
                    VecD f3 = fft_half_abs(sig3);

                    double en1 = 0.0, en2 = 0.0, en3 = 0.0;
                    for (int k = start_freq; k < mid_freq; ++k) {
                        en1 += f1[k] * f1[k];
                        en2 += f2[k] * f2[k];
                        en3 += f3[k] * f3[k];
                    }
                    en1 = std::sqrt(en1);
                    en2 = std::sqrt(en2);
                    en3 = std::sqrt(en3);

                    if (en2 != 0.0 && en3 != 0.0) {
                        contr1 = en1 * 10.0 / en2;
                        contr2 = en1 * 10.0 / en3;
                    }
                }

                /* ------------- ncnn 推理 ------------- */
                /* 输入 0 ：(1,3,ws) -> Mat(w=ws, h=1, c=3) */
                ncnn::Mat in_sig(ws, 1, 3);
                for (int c = 0; c < 3; ++c) {
                    float *ptr = in_sig.channel(c);
                    for (int s = 0; s < ws; ++s) {
                        double val = proc[s][c] - basic_avg[c];
                        ptr[s] = static_cast<float>(val);
                    }
                }

                /* 输入 1 ：(1,2) -> Mat(w=2, h=1, c=1) */
                ncnn::Mat in_enh(2);
                in_enh[0] = static_cast<float>(contr1);
                in_enh[1] = static_cast<float>(contr2);

                ncnn::Extractor ex = net.create_extractor();
                ex.set_light_mode(true);
                ex.input("signal", in_sig);   // ★ 名字保持一致
                ex.input("enhance", in_enh);  // ★

                ncnn::Mat out;
                ex.extract("output", out);    // ★
                float prob = out[0];
                prob = std::max(0.0f, prob * 1.8f - 0.8f);

                /* 队列累计 */
                score_q.push_back(prob);
                if (score_q.size() > static_cast<std::size_t>(queue_len))
                    score_q.erase(score_q.begin());

                double current_sum =
                        std::accumulate(score_q.begin(), score_q.end(), 0.0);
                if (current_sum > overall_max) overall_max = current_sum;
                bool alert = current_sum >= alarm_threshold;

                /* 写 plot_datas (ws 行，每行 7 列) */
                for (int i = 0; i < ws; ++i) {
                    plot_datas.push_back(proc[i][0] - basic_avg[0]);
                    plot_datas.push_back(proc[i][1] - basic_avg[1]);
                    plot_datas.push_back(proc[i][2] - basic_avg[2]);
                    plot_datas.push_back(contr1);
                    plot_datas.push_back(contr2);
                    plot_datas.push_back(static_cast<double>(prob));
                    plot_datas.push_back((prob > 0.5) ? 1.0 : 0.0);
                }

                /* 状态机 */
                if (alert) {
                    if (status != 1) alert_start_idx = static_cast<int>(anchor);
                    status = 1;
                    if (current_sum > max_sum) max_sum = current_sum;
                } else {
                    if (status == 1)
                        std::cout << "+alert @ " << alert_start_idx << " ~ "
                                  << anchor << ", max_sum=" << max_sum
                                  << std::endl;
                    status = 0;
                    max_sum = 0.0;
                }
            } /* --- ws 分支结束 --- */

            /* ===== ws_ori 分支 ===== （保持原样） */
            if (anchor >= static_cast<std::size_t>(ws_ori) && anchor % ws_ori == 0) {
                MatD proc(ws_ori, std::vector<double>(3, 0.0));
                for (int i = 0; i < ws_ori; ++i)
                    for (int j = 0; j < 3; ++j)
                        proc[i][j] = raw[anchor - ws_ori + i][j];

                VecD sig1(ws_ori, 0.0), sig2(ws_ori, 0.0), sig3(ws_ori, 0.0);
                for (int i = 0; i < ws_ori; ++i) {
                    sig1[i] = proc[i][0];
                    sig2[i] = proc[i][1];
                    sig3[i] = proc[i][2];
                }
                VecD f1 = fft_half_abs(sig1);
                VecD f2 = fft_half_abs(sig2);
                VecD f3 = fft_half_abs(sig3);

                double en1 = 0.0, en2 = 0.0, en3 = 0.0;
                for (int k = start_freq; k < mid_freq; ++k) {
                    en1 += f1[k] * f1[k];
                    en2 += f2[k] * f2[k];
                    en3 += f3[k] * f3[k];
                }
                en1 = std::sqrt(en1);
                en2 = std::sqrt(en2);
                en3 = std::sqrt(en3);

                double contr1 = 0.0, contr2 = 0.0;
                if (en2 != 0.0 && en3 != 0.0) {
                    contr1 = en1 * 10.0 / en2;
                    contr2 = en1 * 10.0 / en3;
                }
                int ori_score = (contr1 >= 30.0 && contr2 >= 15.0) ? 1 : 0;

                /* 队列累计 */
                score_q_ori.push_back(static_cast<double>(ori_score));
                if (score_q_ori.size() > static_cast<std::size_t>(queue_len_ori))
                    score_q_ori.erase(score_q_ori.begin());

                double cur_sum_ori =
                        std::accumulate(score_q_ori.begin(), score_q_ori.end(), 0.0);
                if (cur_sum_ori > overall_max_ori) overall_max_ori = cur_sum_ori;

                /* 填 plot_datas_ori (ws_ori 行 × 4 列) */
                for (int i = 0; i < ws_ori; ++i) {
                    plot_datas_ori.push_back(contr1);
                    plot_datas_ori.push_back(contr2);
                    plot_datas_ori.push_back(static_cast<double>(ori_score));
                    plot_datas_ori.push_back((cur_sum_ori >= 2.0) ? 1.0 : 0.0);
                }

                /* saver */
                if (saver_pos != 1 || (saver_pos == 1 && ori_score == 1)) {
                    for (int i = 0; i < ws_ori; ++i)
                        for (int j = 0; j < 3; ++j)
                            saver.push_back(proc[i][j] - basic_avg[j]);
                    saver.push_back(contr1);
                    saver.push_back(contr2);
                    saver.push_back((saver_pos == -1)
                                    ? 0.0
                                    : static_cast<double>(ori_score));
                }
            } /* --- ws_ori 分支结束 --- */
        }     /* --- for anchor --- */

        if (status == 1)
            std::cout << "+alert @ " << alert_start_idx
                      << " ~ end, max_sum=" << max_sum << std::endl;

        max_sums[0] = overall_max;
        max_sums[1] = overall_max_ori;
    }

private:
    /* ncnn Net */
    ncnn::Net net;

    /* ----------------- 算法参数 ----------------- */
    int ws, ws_ori;
    bool enhanced;
    int shift_value;
    double coef;
    int start_freq, mid_freq;
    int queue_len, queue_len_ori;
    double alarm_threshold;
};

/*-----------------------------------------------------------------
 *          保存一维数组到文本文件（17 位精度）
 *----------------------------------------------------------------*/
static void save_flat(const VecD &v, const std::string &fname) {
    std::ofstream fout(fname.c_str());
    if (!fout.is_open()) {
        std::cerr << "cannot open " << fname << std::endl;
        return;
    }
    fout.setf(std::ios::fixed);
    fout << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (double d: v) fout << d << '\n';
}

/*-----------------------------------------------------------------
 *                              main
 *----------------------------------------------------------------*/
int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0]
                  << "  <model.param>  <model.bin>  <signal.log>\n";
        return 0;
    }
    std::string param_path(argv[1]);
    std::string bin_path(argv[2]);
    std::string log_path(argv[3]);

    try {
        SignalProcessor proc(param_path, bin_path, 256, 256, false, 0);

        VecD plot_datas, plot_datas_ori, saver;
        double max_sums[2] = {0.0, 0.0};

        proc.process_signal(log_path, 0, plot_datas, plot_datas_ori, saver,
                            max_sums);

        /* ----------- 保存结果 ---------- */
        save_flat(plot_datas, "/home/manu/tmp/plot_datas_ncnn.txt");
        save_flat(plot_datas_ori, "/home/manu/tmp/plot_datas_ori_ncnn.txt");
        save_flat(saver, "/home/manu/tmp/saver_ncnn.txt");
        VecD mv = {max_sums[0], max_sums[1]};
        save_flat(mv, "/home/manu/tmp/max_sums_ncnn.txt");

        std::cout << "Done. Files written to /home/manu/tmp/ ...\n";
    }
    catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
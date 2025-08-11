#include <onnxruntime_cxx_api.h>
#include <fftw3.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

/*-----------------------------------------------------------------
 *   一些显式类型别名（全部不用 auto）
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
    for (std::size_t i = k; i < tmp.size() - k; ++i)
        sum += tmp[i];
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
        double re = out[i][0];
        double im = out[i][1];
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
        /* 手工分割 ',' */
        std::vector<std::string> token;
        std::string cur;
        for (std::size_t i = 0; i < line.size(); ++i) {
            char ch = line[i];
            if (ch == ',') {
                token.push_back(cur);
                cur.clear();
            } else cur.push_back(ch);
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
            } else continue;
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
 *                       SignalProcessor
 *----------------------------------------------------------------*/
class SignalProcessor {
public:
    SignalProcessor(const std::string &onnx_path,
                    int window_size = 128,
                    int window_size_ori = 256,
                    bool enhanced_mode = true,
                    int shift_val = 0)
            : ws(window_size),
              ws_ori(window_size_ori),
              enhanced(enhanced_mode),
              shift_value(shift_val) {
        /* -------- 创建 ORT Session (CPU) -------- */
        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "demo"};
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        session = Ort::Session(env, onnx_path.c_str(), opts);

        Ort::AllocatorWithDefaultOptions allocator;
        input0_name = session.GetInputNameAllocated(0, allocator).get();
        input1_name = session.GetInputNameAllocated(1, allocator).get();
        output_name = session.GetOutputNameAllocated(0, allocator).get();

        coef = 128.0 / static_cast<double>(ws);
        start_freq = 2;
        mid_freq = 25;
        queue_len = (ws == 128) ? 30 : 15;
        queue_len_ori = 15;
        alarm_threshold = 3;   /* sensitivity=1 */
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
            for (std::size_t i = 0; i < raw.size(); ++i)
                column.push_back(raw[i][ch]);
            basic_avg[ch] = trimmed_mean(column);
        }

        /* 队列 */
        VecD score_q;
        VecD score_q_ori;

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
                for (int i = 0; i < ws; ++i) {
                    for (int j = 0; j < 3; ++j)
                        proc[i][j] = raw[anchor - ws + i][j];
                }

                /* 近期均值 */
                double recent_avg[3] = {0.0, 0.0, 0.0};
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

                    /* energy */
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

                /* -------- onnx 输入 -------- */
                VecF input_signal;
                input_signal.resize(3 * ws, 0.0f);
                for (int c = 0; c < 3; ++c)
                    for (int s = 0; s < ws; ++s) {
                        double val = proc[s][c] - basic_avg[c];
                        input_signal[c * ws + s] = static_cast<float>(val);
                    }
                float enhance_arr[2];
                enhance_arr[0] = static_cast<float>(contr1);
                enhance_arr[1] = static_cast<float>(contr2);

                Ort::MemoryInfo meminfo("Cpu", OrtDeviceAllocator, 0,
                                        OrtMemTypeDefault);
                int64_t dim_sig[3] = {1, 3, ws};
                int64_t dim_enh[2] = {1, 2};

                Ort::Value tensor_sig = Ort::Value::CreateTensor<float>(
                        meminfo,
                        input_signal.data(),            // data ptr
                        input_signal.size(),            // data len
                        dim_sig, 3);                    // shape

                Ort::Value tensor_enh = Ort::Value::CreateTensor<float>(
                        meminfo,
                        enhance_arr, 2,                 // data ptr / len
                        dim_enh, 2);                    // shape

                const char *input_names[2] = {input0_name.c_str(),
                                              input1_name.c_str()};
                const char *output_names[1] = {output_name.c_str()};

                Ort::Value input_tensors[2] = {
                        std::move(tensor_sig),
                        std::move(tensor_enh)};

                std::vector<Ort::Value> output_tensors =
                        session.Run(Ort::RunOptions{nullptr},
                                    input_names,
                                    input_tensors, 2,
                                    output_names, 1);

                float prob = *output_tensors[0].GetTensorMutableData<float>();
                prob = std::max(0.0f, prob * 1.8f - 0.8f);

                /* 队列累计 */
                score_q.push_back(prob);
                if (score_q.size() > static_cast<std::size_t>(queue_len))
                    score_q.erase(score_q.begin());

                double current_sum = std::accumulate(score_q.begin(), score_q.end(), 0.0);
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
//                    plot_datas.push_back(static_cast<double>(0));
//                    plot_datas.push_back(0.0);
                }

                /* 状态机 */
                if (alert) {
                    if (status != 1) alert_start_idx = static_cast<int>(anchor);
                    status = 1;
                    if (current_sum > max_sum) max_sum = current_sum;
                } else {
                    if (status == 1)
                        std::cout << "+alert @ "
                                  << alert_start_idx << " ~ "
                                  << anchor
                                  << ", max_sum=" << max_sum << std::endl;
                    status = 0;
                    max_sum = 0.0;
                }
            }  /* --- ws 分支结束 --- */

            /* ===== ws_ori 分支 ===== */
            if (anchor >= static_cast<std::size_t>(ws_ori) && anchor % ws_ori == 0) {
                /* 取窗口 */
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

                double cur_sum_ori = std::accumulate(score_q_ori.begin(),
                                                     score_q_ori.end(), 0.0);
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
                    saver.push_back((saver_pos == -1) ? 0.0 : static_cast<double>(ori_score));
                }
            } /* --- ws_ori 分支结束 --- */
        }     /* --- for anchor --- */

        if (status == 1)
            std::cout << "+alert @ "
                      << alert_start_idx << " ~ end, max_sum="
                      << max_sum << std::endl;

        max_sums[0] = overall_max;
        max_sums[1] = overall_max_ori;
    }

private:
    /* ----------------- onnxruntime 相关 ----------------- */
    Ort::Session session{nullptr};
    std::string input0_name;
    std::string input1_name;
    std::string output_name;

    /* ----------------- 算法参数 ----------------- */
    int ws;
    int ws_ori;
    bool enhanced;
    int shift_value;
    double coef;
    int start_freq;
    int mid_freq;
    int queue_len;
    int queue_len_ori;
    double alarm_threshold;
};

#include <iomanip>          // for std::setprecision
#include <limits>           // for std::numeric_limits

/*-----------------------------------------------------------------
 *          把一维数组保存到文本文件（每行一个值，17 位精度）
 *----------------------------------------------------------------*/
static void save_flat(const VecD &v, const std::string &fname) {
    std::ofstream fout(fname.c_str());
    if (!fout.is_open()) {
        std::cerr << "cannot open " << fname << std::endl;
        return;
    }

    /* 固定小数格式 + 最大有效位数（≈17） */
    fout.setf(std::ios::fixed);
    fout << std::setprecision(std::numeric_limits<double>::max_digits10);

    const double *p = v.data();
    for (std::size_t i = 0; i < v.size(); ++i, ++p)
        fout << *p << '\n';
}

/*-----------------------------------------------------------------
 *                              main
 *----------------------------------------------------------------*/
int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0]
                  << "  <model.onnx>  <signal.log>\n";
        return 0;
    }
    std::string model_path("/home/manu/tmp/ecg_net_model_sim.onnx");
    std::string log_path("/home/manu/tmp/log_ecg/100m-zgw-2.log");

    SignalProcessor proc(model_path, 256, 256, false, 0);

    VecD plot_datas;
    VecD plot_datas_ori;
    VecD saver;
    double max_sums[2] = {0.0, 0.0};

    proc.process_signal(log_path, 0,
                        plot_datas, plot_datas_ori, saver, max_sums);

    /* ----------- 必须保留的四个 txt ---------- */
    save_flat(plot_datas, "/home/manu/tmp/plot_datas_cpp.txt");
    save_flat(plot_datas_ori, "/home/manu/tmp/plot_datas_ori_cpp.txt");
    save_flat(saver, "/home/manu/tmp/saver_cpp.txt");

    VecD max_vec;
    max_vec.push_back(max_sums[0]);
    max_vec.push_back(max_sums[1]);
    save_flat(max_vec, "/home/manu/tmp/max_sums_cpp.txt");

    std::cout << "Done. Files written to /home/manu/tmp/ ...\n";
    return 0;
}
import os
import sys
import time

import matplotlib
import numpy as np
import onnx
import torch
from mplfonts import use_font
from onnxsim import simplify

from se_conv_ver_mini_logi import ECGNet as ECGNetMini

matplotlib.use('TkAgg')
use_font("Noto Sans CJK SC")


class SignalProcessor:
    def __init__(self, shift_value=0,
                 # model_path='models/best_modelm_0514_25_9991.pth',
                 # model_path='models/best_modelm_0514_36_9997.pth',
                 model_path='/media/manu/ST2000DM005-2U91/workspace/hj/best_modelm_060610_7_9981.pth',  # 128,zero_fr
                 # model_path='models/best_modelm_0527_82_9998.pth',#128,zero_fr
                 # model_path='models/best_modelm_0319_76_9999.pth',
                 window_size=128,  # 256,
                 # window_size=256,
                 window_size_ori=256,
                 enhanced_mode=False  # False
                 ):
        self.model = None
        self.first_dump_done = False
        self.window_size = window_size
        self.window_size_ori = window_size_ori
        self.model_name = os.path.basename(model_path)
        self.init_model(model_path)
        self.coef = 128 / self.window_size
        self.start_freq = 2  # int(2 / self.coef)
        self.mid_freq = 25  # int(15 / self.coef) + 1
        # self.end_freq = int(30 / self.coef) + 1
        self.sensitivity = 1
        self.queue_len_ori = 15
        # self.alarm_thresholds = [4, 8, 12, 21]
        # self.alarm_thresholds = [3, 5, 9, 11]
        if self.window_size == 128:
            self.alarm_thresholds = [4.5, 8, 12]
            self.queue_len = 30  # 15
        else:
            self.alarm_thresholds = [3, 5, 9, 11]
            self.queue_len = 15  # 15
        self.enhanced_mode = enhanced_mode
        self.status = -1
        self.alert_start_idx = -1
        self.score_queue = []
        self.score_queue_ori = []
        self.shift_value = shift_value

    def init_members(self):
        self.status = -1
        self.alert_start_idx = -1
        self.score_queue = []
        self.score_queue_ori = []

    def init_model(self, model_path):
        self.model = ECGNetMini()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    @staticmethod
    def calculate_average(data):
        sorted_data = np.sort(data.flatten())
        num_elements_to_exclude = int(len(sorted_data) * 0.1)
        filtered_data = sorted_data[num_elements_to_exclude:-num_elements_to_exclude]
        return np.mean(filtered_data)

    @staticmethod
    def fft_analysis(signal):
        half_n = len(signal) // 2
        fft_result = np.fft.fft(signal)
        fft_result_half = fft_result[:half_n]
        fft_result_half = np.abs(fft_result_half)
        fft_result_half[0] /= 306.7
        return fft_result_half

    @staticmethod
    def load_data(filepath):
        raw_channels = np.zeros((0, 4))
        with open(filepath, 'r', encoding='utf-8') as f:
            datas = f.readlines()
        for data_one_line in datas:
            data_str = data_one_line.split(',')
            try:
                if data_str[0] == '@01':
                    index = int(data_str[1], 16)
                    d_ir1 = int(data_str[2], 16)
                    d_ir2 = int(data_str[3], 16)
                    d_ir3 = int(data_str[4], 16)
                    d_ir4 = int(data_str[5], 16)
                elif len(data_str) >= 3 and data_str[2] == '@01':
                    index = int(data_str[3], 16)
                    d_ir1 = int(data_str[4], 16)
                    d_ir2 = int(data_str[5], 16)
                    d_ir3 = int(data_str[6], 16)
                    d_ir4 = int(data_str[7], 16)
                else:
                    continue
            except Exception:
                continue
            raw_channels = np.concatenate((raw_channels, np.array([[d_ir1, d_ir2, d_ir3, d_ir4]])))
        # return raw_channels[70:,:]
        return raw_channels

    def process_signal(self, filepath, saver_pos=0, length=256):
        plot_datas = np.zeros((0, 7))
        plot_datas_ori = np.zeros((0, 4))
        self.filepath = filepath
        max_sum, overall_max_sum, overall_max_sum_ori = 0, 0, 0
        time_0 = time.time()
        raw_channels = self.load_data(filepath)[self.shift_value:, :]
        time_1 = time.time()
        print(f' load time={time_1 - time_0:.3f} seconds')
        basic_average = np.array(
            [self.calculate_average(raw_channels[:, 0]), self.calculate_average(raw_channels[:, 1]),
             self.calculate_average(raw_channels[:, 2])])
        # basic_average = self.calculate_average(raw_channels[:, :3])
        anchor_idx = 0
        saver = []
        ori_score = 0
        time_0 = time.time()
        while anchor_idx <= len(raw_channels):
            anchor_idx += 1

            if anchor_idx >= self.window_size and anchor_idx % self.window_size == 0:
                process_data = raw_channels[anchor_idx - self.window_size:anchor_idx, :3].copy()
                # print(anchor_idx, self.window_size)
                if len(process_data) != self.window_size:
                    continue
                # recent_average = self.calculate_average(process_data)
                recent_average = np.array(
                    [self.calculate_average(process_data[:, 0]), self.calculate_average(process_data[:, 1]),
                     self.calculate_average(process_data[:, 2])])
                basic_average = basic_average * 0.9 + recent_average * 0.1
                contr1, contr2 = 0, 0
                if self.enhanced_mode:
                    f1, f2, f3 = self.fft_analysis(process_data[:, 0]), self.fft_analysis(
                        process_data[:, 1]), self.fft_analysis(process_data[:, 2])
                    en1 = np.sqrt(np.sum(f1[self.start_freq:self.mid_freq] ** 2))
                    en2 = np.sqrt(np.sum(f2[self.start_freq:self.mid_freq] ** 2))
                    en3 = np.sqrt(np.sum(f3[self.start_freq:self.mid_freq] ** 2))
                    contr1 = en1 * 10 / en2
                    contr2 = en1 * 10 / en3
                    # print(f'fft:{contr1}, {contr2}')
                    fr = torch.Tensor([[contr1, contr2]])
                    fr = torch.zeros((1, 2))
                else:
                    fr = torch.zeros((1, 2))
                record_len = process_data.shape[0]
                process_data = (process_data - basic_average)
                process_input_data = torch.Tensor(process_data.transpose()).unsqueeze(0)
                outputs = self.model(process_input_data, fr)
                # print(outputs)
                # print("@@@@@@@@@@@@@")
                # ======================================================================================================
                # # ★★★ 3. 导出 & 简化 ONNX
                # onnx_path = "/home/manu/tmp/ecg_net_model.onnx"  # 原始导出文件
                # onnx_path_simplified = "/home/manu/tmp/ecg_net_model_sim.onnx"  # 简化后文件
                # torch.onnx.export(
                #     self.model,  # 要导出的模型
                #     (process_input_data, fr),  # *多输入* 用元组/列表
                #     onnx_path,  # 输出文件名
                #     input_names=['signal', 'enhance'],
                #     output_names=['output'],
                #     # dynamic_axes={                      # 如需动态 batch 可取消注释
                #     #     'signal':  {0: 'batch'},
                #     #     'enhance': {0: 'batch'},
                #     #     'output':  {0: 'batch'}
                #     # },
                #     opset_version=12,  # >=11 即可，推荐 13+
                #     do_constant_folding=True,
                #     verbose=False
                # )
                # print(f"ONNX 已导出到: {onnx_path}")
                # print("开始 simplify ...")
                # onnx_model = onnx.load(onnx_path)
                # model_simp, check = simplify(
                #     onnx_model,
                #     dynamic_input_shape=False,  # 改成 True 如果用了 dynamic_axes
                #     # 也可以指定固定或范围输入，如：
                #     # input_shapes={'signal': [1, 1, 1000], 'enhance': [1, 1]},
                # )
                # assert check, "Simplified ONNX model 在数值校验时不一致！"
                # onnx.save(model_simp, onnx_path_simplified)
                # print(f"Simplified ONNX 已保存到: {onnx_path_simplified}")
                # sys.exit(0)
                # ======================================================================================================
                # _, preds = torch.max(outputs.data, 1)
                probs = outputs.data.detach().cpu().numpy()[0][0]
                preds = int(probs > 0.5)
                # ---------------- ② 只在第一次 probs>0.5 时保存 ---------------- #
                if (not self.first_dump_done) and (probs > 0.5):
                    print(f"probs --> {probs}")
                    self.first_dump_done = True
                    # 组合成一维向量：signal(1,3,L)->展平，fr(1,2)->展平
                    dump_vec = np.concatenate(
                        [process_input_data.numpy().flatten(),
                         fr.numpy().flatten()]
                    )
                    # 也可以附带一个 label（例如 1），方便后面的 ncnn 验证
                    dump_vec = np.append(dump_vec, 1)
                    # 保存到文件
                    np.savetxt("/home/manu/tmp/test_0.txt", dump_vec, fmt="%.6f")
                    print("saved first positive sample to file")
                # ---------------------------------------------------------------- #
                # preds = preds.detach().cpu().numpy()[0]
                # probs = torch.softmax(outputs, 1).detach().cpu().numpy()[0][1]
                # self.score_queue.append(preds)
                probs = max(0.0, probs * 1.8 - 0.8)
                self.score_queue.append(probs)
                if len(self.score_queue) > self.queue_len:
                    self.score_queue.pop(0)
                current_score = np.sum(self.score_queue)
                overall_max_sum = max(overall_max_sum, current_score)
                if overall_max_sum > 2.07:
                    print(f"overall_max_sum --> {overall_max_sum}")

                alert = current_score >= self.alarm_thresholds[self.sensitivity - 1]
                record = np.array([(contr1, contr2, probs, preds) for _ in range(record_len)])
                # record = np.array([(contr1, contr2, 0, 0) for _ in range(record_len)])
                record = np.concatenate([process_data, record], axis=1)
                plot_datas = np.concatenate((plot_datas, record), axis=0)
                if alert:
                    if self.status != 1:
                        # print(f" new alert at {anchor_idx}")
                        self.alert_start_idx = anchor_idx
                    self.status = 1
                    max_sum = max(current_score, max_sum)
                else:
                    if self.status == 1:
                        print(f"+alert at {self.alert_start_idx} ~ {anchor_idx}, max_sum={max_sum}")
                    self.status = 0
                    max_sum = 0

            if anchor_idx >= self.window_size_ori and anchor_idx % self.window_size_ori == 0:
                process_data = raw_channels[anchor_idx - self.window_size_ori:anchor_idx, :3].copy()
                if len(process_data) == self.window_size_ori:
                    # if anchor_idx == 5632:
                    #     print(anchor_idx)
                    f1, f2, f3 = self.fft_analysis(process_data[:, 0]), self.fft_analysis(
                        process_data[:, 1]), self.fft_analysis(process_data[:, 2])
                    en1 = np.sqrt(np.sum(f1[self.start_freq:self.mid_freq] ** 2))
                    en2 = np.sqrt(np.sum(f2[self.start_freq:self.mid_freq] ** 2))
                    en3 = np.sqrt(np.sum(f3[self.start_freq:self.mid_freq] ** 2))
                    contr1 = en1 * 10 / en2
                    contr2 = en1 * 10 / en3

                    if contr1 >= 30 and contr2 >= 15:
                        ori_score = 1
                    else:
                        ori_score = 0
                    self.score_queue_ori.append(ori_score)
                    record_len = process_data.shape[0]
                    if len(self.score_queue_ori) > self.queue_len_ori:
                        self.score_queue_ori.pop(0)

                    current_score_ori = np.sum(self.score_queue_ori)
                    overall_max_sum_ori = max(overall_max_sum_ori, current_score_ori)
                    record = np.array(
                        [(contr1, contr2, ori_score, int(current_score_ori >= 2)) for _ in range(record_len)])
                    # record = np.concatenate([process_data, record], axis=1)
                    plot_datas_ori = np.concatenate((plot_datas_ori, record), axis=0)
                    if saver_pos != 1 or (saver_pos == 1 and ori_score == 1):
                        temp_saver = (process_data - basic_average).flatten()
                        temp_saver = np.append(temp_saver, contr1)
                        temp_saver = np.append(temp_saver, contr2)
                        if saver_pos == -1:
                            temp_saver = np.append(temp_saver, 0)
                        else:
                            temp_saver = np.append(temp_saver, ori_score)
                        saver.append(temp_saver)

        if self.status == 1:
            print(f"+alert at {self.alert_start_idx} ~ {anchor_idx}, max_sum={max_sum}")
        if anchor_idx > self.window_size:
            time_1 = time.time()
            print(
                f" [{overall_max_sum:.1f}/{overall_max_sum_ori}] time: {(time_1 - time_0) / (anchor_idx // self.window_size):.3f}s/cycle, {anchor_idx // self.window_size} cycles, {anchor_idx} samples, {time_1 - time_0:.3f} seconds\n")
        max_sums = (overall_max_sum, overall_max_sum_ori)
        saver = np.array(saver)
        return plot_datas, plot_datas_ori, saver, max_sums


if __name__ == "__main__":
    # main_path = r'G:\Windows\0321\0321'
    # main_path = r'G:\Windows\0318\0318'
    # main_path = r'G:\Windows\0430pir\数据采集'
    # SUB_DIR = '0430pir'  # '0513'
    # # main_path = r'G:\Windows\点型红外20250325\4384\反例'
    # # main_path = r'G:\Windows\点型红外20250325\RD\反例'
    # # main_path = r'G:\Windows\点型红外20250325\RD\正例'
    # # main_path = r'G:\Windows\点型红外样机20250513'
    # # SUB_DIR = '0513d'#'0513'
    # main_path = r'G:\Windows\0528pir\1\SF-EX数据采集正例'
    # SUB_DIR = '0528pir_0606_pos'  # '0513'
    # main_path = r'G:\Windows\0528pir\1\SF-EX误报采集'
    # SUB_DIR = '0528pir_0606_neg'  # '0513'
    # main_path = r'G:\Windows\点型红外样机20250513'
    # SUB_DIR = '0513_0612'  # '0513'
    # main_path = r'G:\Windows\点型红数据采集20250617'
    # SUB_DIR = '0618_0617'  # '0513'
    # main_path = r'G:\Windows\点型红外6x6煤油0626'
    # SUB_DIR = '0626_0626'  # '0513'
    # # main_path = r'data/20250617/dev1/data'
    main_path = "/home/manu/tmp/log_ecg/"
    SUB_DIR = '0702_0703'
    shift_value = 0
    SUB_DIR = SUB_DIR + f'-{shift_value}'

    processor = SignalProcessor(
        shift_value=shift_value,
        # model_path=r'models/best_modelm_0527_61_9995.pth',
        # model_path=r'models/best_modelm_0529_4_9931.pth',
        # model_path=r'models/best_modelm_0529_1_9927.pth',
        # model_path=r'models/best_modelm_0529_0_9703.pth',
        # model_path=r'models/best_modelm_0529_10_9713.pth',# v
        # model_path=r'models/best_modelm_053016_4_9956.pth',# v
        # model_path=r'models/best_modelm_053016_11_9960.pth',# v
        # model_path=r'models/best_modelm_053016_56_9960.pth',# v
        # model_path=r'models/best_modelm_060414_17_9948.pth',# v
        # model_path=r'models/best_modelm_060414_46_9940.pth',# v
        # model_path=r'models/best_modelm_0529_10_9687.pth',
        model_path='/media/manu/ST2000DM005-2U91/workspace/hj/best_modelm_060610_7_9981.pth',  # v-256
        window_size=256,
        enhanced_mode=False
    )
    print(processor.model_name, SUB_DIR)
    if not os.path.exists(f'./figures/{SUB_DIR}'):
        os.makedirs(f'./figures/{SUB_DIR}')
    if not os.path.exists(f'./npdata/{SUB_DIR}'):
        os.makedirs(f'./npdata/{SUB_DIR}')
    # processor.init_members()
    # processor.process_signal(r'G:\Windows\0318\0318\4384\正例\SignalLog【2025-3-18 14_34_47】-25-1-jj.log')
    # print("done")
    res = []
    for root, dirs, files in os.walk(main_path):
        for file_n, file in enumerate(files):
            if file.find('通道') != -1:
                continue
            if os.path.splitext(file)[-1] == ".log":
                file_name = f'{root[15:]}\\{file[:-4]}'
                saver_pos = 0
                if file_name.find('正例') != -1 or file_name.find('烷') != -1 or file_name.find(
                        '酒') != -1 or file_name.find('火') != -1:
                    saver_pos = 1
                elif file_name.find('负例') != -1 or file_name.find('反例') != -1 or file_name.find(
                        '误报') != -1 or file_name.find(
                    '太阳') != -1 or file_name.find('阳光') != -1 or file_name.find('灯') != -1 or file_name.find(
                    '树叶') != -1 or file_name.find('光影') != -1 or file_name.find('玻璃') != -1 or file_name.find(
                    '晃动') != -1 or file_name.find('镜子') != -1 or file_name.find('墙面') != -1:
                    saver_pos = -1

                filepath = os.path.join(root, file)
                print(f" processing file {file_n}: {filepath}")
                processor.init_members()
                plot_datas, plot_datas_ori, saver, max_sums = processor.process_signal(filepath, saver_pos)
                arrays = [plot_datas, plot_datas_ori, saver, max_sums]
                files = ['/home/manu/tmp/plot_datas_py.txt',
                         '/home/manu/tmp/plot_datas_ori_py.txt',
                         '/home/manu/tmp/saver_py.txt',
                         '/home/manu/tmp/max_sums_py.txt']
                for arr, fname in zip(arrays, files):
                    np.savetxt(fname, np.ravel(arr), fmt='%s')
                file_save_name = f"{'P' if saver_pos == 1 else 'N' if saver_pos == -1 else 'U'}_" + file_name.replace(
                    "\\", "-").replace("//", "-").replace("/", "-")
                if len(saver) > 0:
                    np.save(f'./npdata/{SUB_DIR}/{file_save_name}.npy', saver)
                # plt.figure()
                # plt.plot(np.clip(plot_datas[:, 0], -200, 200), label='ch1')
                # plt.plot(np.clip(plot_datas[:, 1], -200, 200), label='ch2')
                # plt.plot(np.clip(plot_datas[:, 2], -200, 200), label='ch3')
                # plt.plot(np.clip(plot_datas[:, 3] * 10 * 0 + 200, 200, 800), label='t1')
                # plt.plot([200 + 150 for _ in range(len(plot_datas[:, 3]))], label='thresh1')
                # plt.plot(np.clip(plot_datas[:, 4] * 10 * 0 + 200, 200, 800), label='t2')
                # plt.plot([200 + 300 for _ in range(len(plot_datas[:, 3]))], label='thresh2')
                # plt.plot(plot_datas[:, 5] * 100 - 300, label='logit')
                # plt.plot([-250 for _ in range(len(plot_datas[:, 5]))], label='logit_thresh')
                # plt.plot(np.clip(plot_datas_ori[:, 0] * 10 + 200, 200, 800), label='tt1')
                # plt.plot(np.clip(plot_datas_ori[:, 1] * 10 + 200, 200, 800), label='tt2')
                # plt.plot(plot_datas_ori[:, 2] * 100 - 400, label='single_ori')
                # plt.plot(plot_datas_ori[:, 3] * 100 - 500, label='alarm_ori')
                # # plt.plot(plot_datas[:, 6] * 100 - 200, label='pred')
                # plt.legend()
                # bn = filepath[15:].replace("\\", "-").replace("//", "-").replace("/", "-")
                # plt.title(f'{file_n}: {bn}')
                # plt.savefig(f'./figures/{SUB_DIR}/{bn}_{processor.model_name}.png', dpi=300)
                # plt.show()
                # # plt.close()
                res.append((file_name, max_sums))
                # print()
    # # main(r'G:\Windows\0423\1\SignalLog【2025-4-23 14_51_59】.log')60~178,
    # with open(f'./results/{SUB_DIR}_{processor.model_name}.txt', 'a') as f:
    #     f.write(processor.model_name + '\n')
    #     for name, ms in res:
    #         # f.write(f'{name}, ({ms[0]:.2f}, {ms[1]})\n')
    #         f.write(f'{name.replace(",", "")},{ms[0]:.3f},{ms[1]}\n')
    print("done")

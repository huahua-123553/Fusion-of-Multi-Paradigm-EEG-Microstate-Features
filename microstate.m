
clear;
clc;
eeglab;  % 启动EEGLAB

%% 设定你存储文件的工作目录
EEGdir = 'E:\ZWH\01yuchuli\测试数据\';  % 设置文件目录，这里是原始EEG数据的存放路径
% 提取所有后缀为set的EEG文件
EEGFiles = dir([EEGdir '*.set']);  % 获取目录下所有的 .set 文件
ALLEEG = [];  % 清空 EEG 数据存储
%% 数据导入
seed = 99;  % 设置随机数种子点，确保结果可重复
num = 5;  % 待分析的被试数目
mi = 0;  % 微状态分析中的平滑参数
output_path='E:\Matlab R2019b\route\bin\'
for i = 1:length(EEGFiles)  % 遍历每个EEG文件
    % 导入EEG数据
    EEG = pop_loadset('filename', EEGFiles(i).name, 'filepath', EEGdir);
    EEG = eeg_checkset(EEG);  % 检查EEG数据集的一致性
    EEG = pop_reref( EEG, []);%重参考
    % 存储数据到EEG结构体
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0);  

    eeglab redraw;  % 更新EEGLAB界面
end

rng(seed);  % 设置随机数种子，使结果可重复

%% 从每个被试中随机选择1000个GFP峰值的数据
% 注意，这里需要确保 `pop_micro_selectdata` 函数已被正确配置
[EEG, ALLEEG] = pop_micro_selectdata(EEG, ALLEEG, 'datatype', 'spontaneous', 'avgref', 0, 'normalise', 0, ...
'MinPeakDist', 10, 'Npeaks', 1000, 'GFPthresh', 0, 'dataset_idx', 1:num);

% 存储数据到EEG结构体
[ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
eeglab redraw;  % 更新EEGLAB界面

%% 选择由GFP峰值构成的数据并让它处于当前活动数据集
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, num, 'retrieve', num + 1, 'study', 0);
eeglab redraw;  % 更新EEGLAB界面

rng(seed);  % 重新设置随机数种子，确保结果可重复

%% 对由GFP峰值构成的模板进行微状态聚类
EEG = pop_micro_segment(EEG, 'algorithm', 'modkmeans', 'sorting', 'Global explained variance', ...
    'Nmicrostates', 5, 'verbose', 1, 'normalise', 0, 'Nrepetitions', 100, 'max_iterations', 1000, ...
    'threshold', 1e-06, 'fitmeas', 'CV', 'optimised', 1);

% 存储数据到EEG结构体
[ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
figure;MicroPlotTopo(EEG,'plot_range',[]);
template_filename = fullfile(output_path, 'Microstate_Template.set');
pop_saveset(EEG, 'filename', 'Microstate_Template.set', 'filepath', output_path);
fprintf('? 微状态模板已保存至: %s\n', template_filename);
%% 选择微状态的数目为4个
%EEG = pop_micro_selectNmicro(EEG, 'Nmicro', 5);
%EEG=pop_micro_selectNmicro(EEG);%查看GEV

% 存储数据到EEG结构体
[ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

LZC_values = [];
%% 依据GFP峰值模板聚类的微状态，判断每一个被试数据的每一个时间点属于哪一个微状态
for i = 1:length(EEGFiles)  % 遍历每一个被试
    fprintf('Importing prototypes and backfitting for dataset %i\n', i);

    % 获取当前数据集
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve', i, 'study', 0);

    % 导入微状态模板
    EEG = pop_micro_import_proto(EEG, ALLEEG, num + 1);
    
    % 微状态拟合
    EEG = pop_micro_fit(EEG, 'polarity', 0);

    % 平滑数据
    EEG = pop_micro_smooth(EEG, 'label_type', 'backfit', 'smooth_type', 'reject segments', 'minTime', mi, 'polarity', 0);

    
    
    
    % 获取微状态统计数据
    EEG = pop_micro_stats(EEG, 'label_type', 'backfit', 'polarity', 0);
    input_str = num2str(EEG.microstate.fit.labels);  % 将 double 数组转换为字符串
    output_str = remove_consecutive_duplicates(input_str);
    LZC_value = calculate_LZC(output_str);  % 计算LZC
    % 存储统计数据
    GEV(i) = EEG.microstate.stats.GEVtotal;  % 全局解释方差
    Duration(i, :) = EEG.microstate.stats.Duration;  % 平均持续时间
    Coverage(i, :) = EEG.microstate.stats.Coverage;  % 覆盖率
    Occurence(i, :) = EEG.microstate.stats.Occurence;  % 出现频次
    eee=EEG.microstate.stats.TP';
    TP(i, :) = eee(: ); % 转换概率
    MspatCorr(i, :) = EEG.microstate.stats.MspatCorr;  % 空间相关系数
    LZC_values = [LZC_values; LZC_value];
    % 存储数据
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    
end
Common_Index = [GEV', Duration, Coverage, Occurence, TP];  % 将所有结果拼接在一起
% 结果存储后，可以根据需要导出这些统计数据并进行进一步分析


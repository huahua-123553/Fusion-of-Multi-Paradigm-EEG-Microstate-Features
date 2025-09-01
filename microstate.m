
clear;
clc;
eeglab;  % ����EEGLAB

%% �趨��洢�ļ��Ĺ���Ŀ¼
EEGdir = 'E:\ZWH\01yuchuli\��������\';  % �����ļ�Ŀ¼��������ԭʼEEG���ݵĴ��·��
% ��ȡ���к�׺Ϊset��EEG�ļ�
EEGFiles = dir([EEGdir '*.set']);  % ��ȡĿ¼�����е� .set �ļ�
ALLEEG = [];  % ��� EEG ���ݴ洢
%% ���ݵ���
seed = 99;  % ������������ӵ㣬ȷ��������ظ�
num = 5;  % �������ı�����Ŀ
mi = 0;  % ΢״̬�����е�ƽ������
output_path='E:\Matlab R2019b\route\bin\'
for i = 1:length(EEGFiles)  % ����ÿ��EEG�ļ�
    % ����EEG����
    EEG = pop_loadset('filename', EEGFiles(i).name, 'filepath', EEGdir);
    EEG = eeg_checkset(EEG);  % ���EEG���ݼ���һ����
    EEG = pop_reref( EEG, []);%�زο�
    % �洢���ݵ�EEG�ṹ��
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0);  

    eeglab redraw;  % ����EEGLAB����
end

rng(seed);  % ������������ӣ�ʹ������ظ�

%% ��ÿ�����������ѡ��1000��GFP��ֵ������
% ע�⣬������Ҫȷ�� `pop_micro_selectdata` �����ѱ���ȷ����
[EEG, ALLEEG] = pop_micro_selectdata(EEG, ALLEEG, 'datatype', 'spontaneous', 'avgref', 0, 'normalise', 0, ...
'MinPeakDist', 10, 'Npeaks', 1000, 'GFPthresh', 0, 'dataset_idx', 1:num);

% �洢���ݵ�EEG�ṹ��
[ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
eeglab redraw;  % ����EEGLAB����

%% ѡ����GFP��ֵ���ɵ����ݲ��������ڵ�ǰ����ݼ�
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, num, 'retrieve', num + 1, 'study', 0);
eeglab redraw;  % ����EEGLAB����

rng(seed);  % ����������������ӣ�ȷ��������ظ�

%% ����GFP��ֵ���ɵ�ģ�����΢״̬����
EEG = pop_micro_segment(EEG, 'algorithm', 'modkmeans', 'sorting', 'Global explained variance', ...
    'Nmicrostates', 5, 'verbose', 1, 'normalise', 0, 'Nrepetitions', 100, 'max_iterations', 1000, ...
    'threshold', 1e-06, 'fitmeas', 'CV', 'optimised', 1);

% �洢���ݵ�EEG�ṹ��
[ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
figure;MicroPlotTopo(EEG,'plot_range',[]);
template_filename = fullfile(output_path, 'Microstate_Template.set');
pop_saveset(EEG, 'filename', 'Microstate_Template.set', 'filepath', output_path);
fprintf('? ΢״̬ģ���ѱ�����: %s\n', template_filename);
%% ѡ��΢״̬����ĿΪ4��
%EEG = pop_micro_selectNmicro(EEG, 'Nmicro', 5);
%EEG=pop_micro_selectNmicro(EEG);%�鿴GEV

% �洢���ݵ�EEG�ṹ��
[ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

LZC_values = [];
%% ����GFP��ֵģ������΢״̬���ж�ÿһ���������ݵ�ÿһ��ʱ���������һ��΢״̬
for i = 1:length(EEGFiles)  % ����ÿһ������
    fprintf('Importing prototypes and backfitting for dataset %i\n', i);

    % ��ȡ��ǰ���ݼ�
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve', i, 'study', 0);

    % ����΢״̬ģ��
    EEG = pop_micro_import_proto(EEG, ALLEEG, num + 1);
    
    % ΢״̬���
    EEG = pop_micro_fit(EEG, 'polarity', 0);

    % ƽ������
    EEG = pop_micro_smooth(EEG, 'label_type', 'backfit', 'smooth_type', 'reject segments', 'minTime', mi, 'polarity', 0);

    
    
    
    % ��ȡ΢״̬ͳ������
    EEG = pop_micro_stats(EEG, 'label_type', 'backfit', 'polarity', 0);
    input_str = num2str(EEG.microstate.fit.labels);  % �� double ����ת��Ϊ�ַ���
    output_str = remove_consecutive_duplicates(input_str);
    LZC_value = calculate_LZC(output_str);  % ����LZC
    % �洢ͳ������
    GEV(i) = EEG.microstate.stats.GEVtotal;  % ȫ�ֽ��ͷ���
    Duration(i, :) = EEG.microstate.stats.Duration;  % ƽ������ʱ��
    Coverage(i, :) = EEG.microstate.stats.Coverage;  % ������
    Occurence(i, :) = EEG.microstate.stats.Occurence;  % ����Ƶ��
    eee=EEG.microstate.stats.TP';
    TP(i, :) = eee(: ); % ת������
    MspatCorr(i, :) = EEG.microstate.stats.MspatCorr;  % �ռ����ϵ��
    LZC_values = [LZC_values; LZC_value];
    % �洢����
    [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    
end
Common_Index = [GEV', Duration, Coverage, Occurence, TP];  % �����н��ƴ����һ��
% ����洢�󣬿��Ը�����Ҫ������Щͳ�����ݲ����н�һ������


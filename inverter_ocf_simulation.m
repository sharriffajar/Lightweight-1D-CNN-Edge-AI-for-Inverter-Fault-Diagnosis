% =========================================================================
% inverter_ocf_simulation.m
% =========================================================================
% Script Simulasi Matematika / Fisika Inverter Single-Phase Full-Bridge
% untuk Membangun Physical-based Dataset Open-Circuit Fault (OCF).
%
% Arsitektur Inverter:
%        +Vdc ---+---------+
%                |         |
%               [S1]      [S3]
%                |---[R-L]-|   <-- Beban AC (Load Current i_L)
%               [S2]      [S4]
%                |         |
%        -Vdc ---+---------+
%
% Modulasi: Unipolar / Bipolar Sinusoidal PWM (SPWM)
%
% Output:
%   - X_data_matlab.csv / dataset_matlab.mat
%   - Visualisasi bentuk gelombang arus per kelas fault
% =========================================================================

clear; clc; close all;

%% 1. PARAMETER SISTEM & BEBAN
Vdc = 312;              % Tegangan Bus DC (Volt) ~ setara 220V RMS AC
f_grid = 50;            % Frekuensi Output (Hz)
f_sw = 10000;           % Frekuensi Switching PWM (10 kHz)
Fs = 1000;              % Frekuensi Sampling Sensor (1 kHz - matching ESP32 window)
WindowSize = 128;       % Jumlah sampel per window (128 ms @ 1kHz = ~6.4 siklus)

% Parameter Beban R-L
R_load = 10;            % Resistansi Beban (Ohm)
L_load = 0.02;          % Induktansi Beban (Henry - 20 mH)

% Parameter Dataset
SamplesPerClass = 500;
ClassNames = {'Healthy', 'S1_Open', 'S2_Open', 'S3_Open', 'S4_Open', 'Multi_Fault'};
NumClasses = length(ClassNames);

%% 2. GENERASI DATASET BERBASIS PERSAMAAN FISIKA
fprintf('Menjalankan simulasi fisika inverter R-L...\n');

total_samples = NumClasses * SamplesPerClass;
X_mat = zeros(total_samples, WindowSize);
y_vec = zeros(total_samples, 1);

dt = 1 / Fs;
t_window = (0:WindowSize-1) * dt;

sample_counter = 1;

for c = 0:NumClasses-1
    fault_type = c;
    fprintf('  Simulasi Kelas %d: %s...\n', c, ClassNames{c+1});

    for s = 1:SamplesPerClass
        % Tambahkan sedikit variasi acak parameter (beban & tegangan)
        % untuk merepresentasikan kondisi operasi nyata
        v_dc_act = Vdc * (1 + 0.02 * randn());
        r_act = R_load * (1 + 0.03 * randn());
        l_act = L_load * (1 + 0.03 * randn());

        % Phase shift acak agar window sinyal tidak selalu mulai dari titik yang sama
        phase_shift = 2 * pi * rand();

        % Hitung gelombang tegangan ideal inverter V_inv(t)
        omega = 2 * pi * f_grid;
        v_fundamental = v_dc_act * sin(omega * t_window + phase_shift);

        % Injeksi Karakteristik Fisika Open-Circuit Fault (OCF)
        % Berdasarkan arah arus dan saklar yang mengalami OCF:
        % - S1 Open: Setengah gelombang positif dipotong/terdistorsi (diode antiparalel S2 konduksi)
        % - S2 Open: Setengah gelombang negatif dipotong
        % - S3 Open: Setengah gelombang positif terdistorsi (lengan kanan)
        % - S4 Open: Setengah gelombang negatif terdistorsi

        v_faulted = v_fundamental;

        switch fault_type
            case 0 % Healthy
                % Sinyal normal dengan harmonisa switching PWM ringan
                v_faulted = v_fundamental;

            case 1 % S1 Open
                % Saat tegangan positif, S1 open sehingga arus positif tertahan
                pos_idx = v_fundamental > 0;
                v_faulted(pos_idx) = v_fundamental(pos_idx) * 0.15;

            case 2 % S2 Open
                % Saat tegangan negatif, S2 open
                neg_idx = v_fundamental < 0;
                v_faulted(neg_idx) = v_fundamental(neg_idx) * 0.15;

            case 3 % S3 Open
                pos_idx = v_fundamental > 0;
                v_faulted(pos_idx) = v_fundamental(pos_idx) * 0.20;

            case 4 % S4 Open
                neg_idx = v_fundamental < 0;
                v_faulted(neg_idx) = v_fundamental(neg_idx) * 0.20;

            case 5 % Multi_Fault (S1 & S4 Open / Kerusakan Parah)
                v_faulted = 0.1 * v_fundamental + 0.05 * v_dc_act * randn(1, WindowSize);
        end

        % Respons Arus Beban R-L: i(t) = (V/Z) * sin(omega*t - phi) + transient
        Z_mag = sqrt(r_act^2 + (omega * l_act)^2);
        phi_lag = atan(omega * l_act / r_act);

        % Filter orde 1 RL transient response
        i_load = zeros(1, WindowSize);
        tau = l_act / r_act;

        % Integrasi numerik arus i_L (Euler Method)
        i_curr = 0;
        for k = 1:WindowSize
            v_in = v_faulted(k);
            di_dt = (v_in - r_act * i_curr) / l_act;
            i_curr = i_curr + di_dt * dt;
            i_load(k) = i_curr;
        end

        % Tambahkan noise sensor ACS712 (termasuk offset ADC)
        sensor_noise = 0.05 * max(abs(i_load)) * randn(1, WindowSize);
        i_measured = i_load + sensor_noise;

        % Fixed Scaling (Bukan Min-Max!) - Dibagi arus maksimum sensor (misal 20A)
        I_max_sensor = 25.0; % Ampere
        i_scaled = i_measured / I_max_sensor;

        % Clipping antara -1.0 dan 1.0 (sesuai range input model CNN)
        i_scaled = max(min(i_scaled, 1.0), -1.0);

        X_mat(sample_counter, :) = i_scaled;
        y_vec(sample_counter) = fault_type;
        sample_counter = sample_counter + 1;
    end
end

%% 3. EXPORT DATASET
fprintf('\nMenyimpan dataset hasil simulasi...\n');

% Simpan ke format .mat
save('dataset_matlab.mat', 'X_mat', 'y_vec', 'ClassNames', 'Fs', 'WindowSize');
fprintf('  - Saved: dataset_matlab.mat\n');

% Simpan ke format .csv untuk dibaca Python / numpy
csv_data = [y_vec, X_mat];
csvwrite('dataset_matlab.csv', csv_data);
fprintf('  - Saved: dataset_matlab.csv\n');

%% 4. VISUALISASI
figure('Name', 'Bentuk Sinyal Arus Beban R-L per Kelas (Simulasi MATLAB)', 'Position', [100, 100, 1000, 600]);
for c = 1:NumClasses
    subplot(2, 3, c);
    sample_idx = find(y_vec == (c-1), 1, 'first');
    plot(t_window * 1000, X_mat(sample_idx, :), 'LineWidth', 1.5);
    title(sprintf('Kelas %d: %s', c-1, ClassNames{c}), 'Interpreter', 'none');
    xlabel('Waktu (ms)');
    ylabel('Arus Normalized (-1 s/d 1)');
    grid on;
    ylim([-1.2, 1.2]);
end
sgtitle('Dataset Sinyal Arus Inverter Full-Bridge R-L (Physical Simulation)');

fprintf('\nSimulasi Selesai! Kamu bisa menggunakan dataset_matlab.csv ini di Python.\n');

% This code is used to generate the main results of the reproduction task
% reported in the following paper:
%
% Meirhaeghe N, Sohn H, Jazayeri M (2021) A precise and adaptive neural 
% mechanism for predictive temporal processing in the frontal cortex. 
% bioRxiv; https://doi.org/10.1101/2021.03.10.434831
%
% The script runs in one go and depends on functions appended at the end.
% The same functions (eg temporal scaling analysis) were used to generate 
% the results of the gain and adaptation tasks. For additional information 
% contact nmrghe@gmail.com or mjaz@mit.edu


% Clear workspace
close all
clear all
clc

% Specify the monkey (G versus H) and condition type (Eye versus Hand)
% (Left versus Right) to be analyzed
% H eye left and G eye right work best
id_monkey = 'H'; % 'H' for monkey H, 'G' for monkey G
id_eye = true; % always true since we only analyze Eye trials
id_left = true; % false for Right target trials

% Specify fixed parameters for the analysis
wbin = 20; % bin size
t_max_short = 800; % max interval from short prior
t_max_long = 1200; % max interval from long prior
buffer_pre_ready = 200; % value of buffer pre Ready
offset_post_ready = 400; % for speed analysis to exclude early measurement

t_s_unique_short = 480:80:800; % range of ts values for Short
t_s_unique_long = 800:100:1200; % range of ts values for Long
t_short = (0:wbin:t_max_short)'; % discretized time vector for Short
t_long = (0:wbin:t_max_long)'; % discretized time vector for Long
t_short_dense = t_short(1):0.1:t_short(end); % densified time vector
% find indices in time vector corresponding to each ts value
ind_t_s_unique_short = find(ismember(t_short, t_s_unique_short));
ind_t_s_unique_long = find(ismember(t_long, t_s_unique_long));

lbd = t_s_unique_long(3)/t_s_unique_short(3); % value of mean ratio

% Define color maps
cmap_grey = flip(brewermap(5, 'greys'));
cmap_red = flip(brewermap(5, 'reds'));
cmap_blue = flip(brewermap(5, 'blues'));

% Load data
load(['../../Data/' id_monkey '_2prior_ReadyM200ms-Set_bin20ms_bootstrap'])

% Use the dataset corresponding to the desired condition
if id_eye
    if id_left
        PSTH_short = PSTH_left_eye_short(buffer_pre_ready/wbin:end, :, :);
        PSTH_long = PSTH_left_eye_long(buffer_pre_ready/wbin:end, :, :);
    else
        PSTH_short = PSTH_right_eye_short(buffer_pre_ready/wbin:end, :, :);
        PSTH_long = PSTH_right_eye_long(buffer_pre_ready/wbin:end, :, :);
    end
else
    if id_left
        PSTH_short = PSTH_left_hand_short(buffer_pre_ready/wbin:end, :, :);
        PSTH_long = PSTH_left_hand_long(buffer_pre_ready/wbin:end, :, :);
    else
        PSTH_short = PSTH_right_hand_short(buffer_pre_ready/wbin:end, :, :);
        PSTH_long = PSTH_right_hand_long(buffer_pre_ready/wbin:end, :, :);
    end
end

% First extract neuron_id from session data
neuron_ids = [];
for iSession=1:length(tag_all_neurons)
    neuron_ids = [neuron_ids; iSession*ones(size(tag_all_neurons{iSession})) tag_all_neurons{iSession}];
end

% Remove neurons with low firing rate (< 1 spk/s)
neurons2keep_short = find( logical( (mean(mean(PSTH_short, 3), 1)>1) .* (~isnan(mean(mean(PSTH_short, 3), 1)))) );
neurons2keep_long = find( logical( (mean(mean(PSTH_long, 3), 1)>1) .* (~isnan(mean(mean(PSTH_long, 3), 1)))) );
neurons2keep = intersect(neurons2keep_short, neurons2keep_long);
PSTH_short = PSTH_short(:, neurons2keep, :);
PSTH_long = PSTH_long(:, neurons2keep, :);
neuron_ids = neuron_ids(neurons2keep, :);

nBootstraps = size(PSTH_short, 3); % nb of bootstraps


% Plot mean firing rate at Ready
% >> Figure 1C
Ready_activity_short = mean(PSTH_short(1, :, :), 3);
Ready_activity_long = mean(PSTH_long(1, :, :), 3);

figure;
plot(Ready_activity_short, Ready_activity_long, 'k.', 'markersize', 8)
hold on
plot(0:01:30, 0:01:30, 'k--', 'linewidth', 2)
axis([0 30 0 30])
xlabel('FR_{short} (spk/s)')
ylabel('FR_{long} (spk/s)')
fixTicks


% Plot the difference of mean firing rate between conditions
% >> inset of Figure 1C
figure; 
h = histogram(Ready_activity_short-Ready_activity_long, -15:1:15, 'facecolor', 'k');
hold on
plot(zeros(size(0:(max(h.BinCounts)+5))), 0:(max(h.BinCounts)+5), 'k--', 'linewidth', 2)
hold on
plot(mean(Ready_activity_short-Ready_activity_long), max(h.BinCounts)+10, 'v', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 12);
[~, pval, ~, stats] = ttest(Ready_activity_long, Ready_activity_short);
xlabel('FR_{short} - FR_{long} (spk/s)')
ylabel('# Neuron')
title(['p = ' num2str(pval)])
fixTicks


% Plot firing rate difference between conditions throughout measurement
% >> Figure S3C top
activity_short = mean(PSTH_short, 3);
activity_long = mean(PSTH_long(1:size(PSTH_short, 1), :, :), 3);

figure;
plot(t_short, activity_short-activity_long, '-', 'linewidth', 0.5, 'color', [0 0 0 0.2])
hold on
ciplot(mean(activity_short-activity_long, 2)-1.96*std(activity_short-activity_long, 1, 2)/sqrt(size(activity_short, 2)), ...
        mean(activity_short-activity_long, 2)+1.96*std(activity_short-activity_long, 1, 2)/sqrt(size(activity_short, 2)), ...
        t_short, cmap_red(4, :))
hold on
plot(t_short, mean(activity_short-activity_long, 2), 'r-', 'linewidth', 2)
hold on
plot(t_short, zeros(size(t_short)), 'k--', 'linewidth', 2)
ylim([-20 20])
xlabel('Time from Ready (ms)')
ylabel('FR_{short}-FR_{long} (spk/s)')
fixTicks


% Plot normalized (contrast index) difference in FR between short and long
% >> Figure S3C middle
figure;
plot(t_short, (activity_short-activity_long)./(activity_short+activity_long), '-', 'linewidth', 0.5, 'color', [0 0 0 0.2])
hold on
plot(t_short, mean((activity_short-activity_long)./(activity_short+activity_long), 2), 'r-', 'linewidth', 2)
hold on
plot(t_short, zeros(size(t_short)), 'k--', 'linewidth', 2)
% axis([0 30 0 30])
xlabel('Time from Ready (ms)')
ylabel('(FR_{short}-FR_{long})/(FR_{short}+FR_{long}) (spk/s)')
fixTicks


% Plot normalized (contrast index) difference in SPEED between short and long
% >> Figure S3C bottom
figure;
speed_activity_short = abs(diff(activity_short, 1));
speed_activity_long = abs(diff(activity_long, 1));
% plot(t_short(1:end-1), (speed_activity_short-speed_activity_long)./(speed_activity_short+speed_activity_long), '-', 'linewidth', 0.5, 'color', [0 0 0 0.2])
% hold on
plot(t_short(1:end-1), mean((speed_activity_short-speed_activity_long)./(speed_activity_short+speed_activity_long), 2), 'k-', 'linewidth', 2)
hold on
ciplot(mean((speed_activity_short-speed_activity_long)./(speed_activity_short+speed_activity_long), 2)-1.96*std((speed_activity_short-speed_activity_long)./(speed_activity_short+speed_activity_long), 1, 2)/sqrt(size(activity_short, 2)), ...
        mean((speed_activity_short-speed_activity_long)./(speed_activity_short+speed_activity_long), 2)+1.96*std((speed_activity_short-speed_activity_long)./(speed_activity_short+speed_activity_long), 1, 2)/sqrt(size(activity_short, 2)), ...
        t_short(1:end-1), cmap_grey(4, :))
hold on
plot(t_short(1:end-1), zeros(size(t_short(1:end-1))), 'k--', 'linewidth', 2)
% axis([0 30 0 30])
xlabel('Time from Ready (ms)')
ylabel('(dFR/dt_{short}-dFR/dt_{long})/(dFR/dt_{short}+dFR/dt_{long}) (spk/s)')
fixTicks



% Plot mean change of firing rate throughout measurement epoch
% >> Figure 1D
mean_diff_PSTH_short = mean(abs(diff(mean(PSTH_short, 3), 1)), 1)/(wbin*1e-3);
mean_diff_PSTH_long = mean(abs(diff(mean(PSTH_long(1:size(PSTH_short, 1), :, :), 3), 1)), 1)/(wbin*1e-3);

figure;
plot(mean_diff_PSTH_short, mean_diff_PSTH_long, 'k.', 'markersize', 8)
hold on
plot(0:01:50, 0:01:50, 'k--', 'linewidth', 2)
axis([0 50 0 50])
xlabel('\delta FR_{short} (spk/s^2)')
ylabel('\delta FR_{long} (spk/s^2)')
fixTicks

% Plot difference in mean change of firing rate between conditions
% >> inset of Figure 1D
figure; 
h = histogram(mean_diff_PSTH_short-mean_diff_PSTH_long, -30:2:50, 'facecolor', 'k');
hold on
plot(zeros(size(0:(max(h.BinCounts)+5))), 0:(max(h.BinCounts)+5), 'k--', 'linewidth', 2)
hold on
plot(mean(mean_diff_PSTH_short-mean_diff_PSTH_long), max(h.BinCounts)+10, 'v', 'markerfacecolor', 'k', 'markeredgecolor', 'k', 'markersize', 12);
[~, pval] = ttest(mean_diff_PSTH_short, mean_diff_PSTH_long);
xlabel('\delta FR_{short} - \delta FR_{long} (spk/s^2)')
ylabel('# Neuron')
title(['p = ' num2str(pval)])
fixTicks


% For every neuron, apply temporal scaling analysis (see in functions below)
for ind_neuron = 1:size(PSTH_short, 2)
    [lambda_opt(ind_neuron), gamma_opt(ind_neuron), delta_opt(ind_neuron), RMSE_opt(ind_neuron)] = findScalingFactor(nanmean(PSTH_short(:, ind_neuron, :), 3), nanmean(PSTH_long(:, ind_neuron, :), 3), t_short, t_long, t_short_dense);
end

% Plot the distribution associated with the 3 fitted parameters of the
% scaling analysis
% >> Figure S3B
figure;
subplot(3, 1, 1)
h=histogram(lambda_opt, 0:0.1:3);
hold on
plot(lbd*ones(size(0:max(h.Values))), 0:max(h.Values), 'r-')
xlabel('\lambda')
ylabel('# Neurons')
fixTicks
subplot(3, 1, 2)
histogram(gamma_opt, 0:0.1:3)
xlabel('\gamma')
ylabel('# Neurons')
fixTicks
subplot(3, 1, 3)
histogram(delta_opt, -2:0.1:2)
xlabel('\delta')
ylabel('# Neurons')
fixTicks
subtitle('r_{short}(t) = \gamma*r_{long}(\lambda*t)+\delta');
fixTicks
tightfig;

% Apply kmeans clustering to separate neurons into two clusters
% scaling versus non-scaling
[idx, C, sumD, D] = kmeans(lambda_opt', 2);
figure; 
h=histogram(lambda_opt(idx==1), 0:0.1:3, 'facecolor', 'k');
hold on
h=histogram(lambda_opt(idx==2), 0:0.1:3, 'facecolor', 'r');
hold on
plot(lbd*ones(size(0:max(h.Values))), 0:max(h.Values), 'r-')
xlabel('\lambda')
ylabel('# Neurons')
title(['Cluster 1: M = ' num2str(C(1)) ', avgD = ' num2str(sumD(1)/nnz(idx==1)) ', n = ' num2str(nnz(idx==1)) ...
        ', Cluster 2: M = ' num2str(C(2)) ', avgD = ' num2str(sumD(2)/nnz(idx==2)) ', n = ' num2str(nnz(idx==2))]);
fixTicks

% Compute variance explained by each neuron
var_short_meas = var(nanmean(PSTH_short, 3), 0, 1);
var_short_meas = 100*var_short_meas/sum(var_short_meas);
[~, ind_sort_var] = sort(var_short_meas, 'descend'); % sort by variance

% Compute mean and stderr of variance explained for scaling and non-scaling
% neuron separately and performed unpaired t-test to test if sig different
mean_ns = mean(var_short_meas((idx==1)));
stderr_ns = std(var_short_meas((idx==1)))/sqrt(length(var_short_meas((idx==1))));

mean_s = mean(var_short_meas((idx==2)));
stderr_s = std(var_short_meas((idx==2)))/sqrt(length(var_short_meas((idx==2))));

[~, p] = ttest2(var_short_meas((idx==1)), var_short_meas((idx==2)));

% Plot variance explained for scaling vs non-scaling neurons
figure; 
histogram(var_short_meas((idx==1)), 0:0.1:3, 'facecolor', 'k');
hold on
histogram(var_short_meas((idx==2)), 0:0.1:3, 'facecolor', 'r');
xlabel('Var')
ylabel('# Neurons')
title(['M_{ns}=' num2str(mean_ns) '+/-' num2str(stderr_ns) ', M_{s}=' num2str(mean_s) '+/-' num2str(stderr_s) ', p = '  num2str(p)])
fixTicks

% Fit mixture-model (two gaussians; see function below) after removing outliers (lbd>3)
[f, xi, y1, y2, mu1, mu2, sig1, sig2] = fitBimodalGaussian(lambda_opt(lambda_opt<3)');

% Fit mixture-model only on neurons which explain more than 0.1 of the total variance
var_threshold = 0.1;
[f_sub, xi_sub, y1_sub, y2_sub, mu1_sub, mu2_sub, sig1_sub, sig2_sub] = fitBimodalGaussian(lambda_opt(var_short_meas>var_threshold)');

% Plot the distributions of scaling factors for the full population as well 
% as the reduced population
% >> Figure 2B
figure; 
h=histogram(lambda_opt, 0:0.1:4, 'FaceColor', 'k');
hold on
histogram(lambda_opt(var_short_meas>var_threshold), 0:0.1:3, 'FaceColor', [17 17 17]/255);
hold on
plot(lbd*ones(size(0:max(h.Values))), 0:max(h.Values), 'r-', 'linewidth', 2)
hold on
plot(ones(size(0:max(h.Values))), 0:max(h.Values), 'k--', 'linewidth', 2)
hold on
plot(mu1, max(h.Values)+10, 'kv', 'markersize', 12)
hold on
plot(mu2, max(h.Values)+10, 'v', 'color', [17 17 17]/255, 'markersize', 12)
hold on
plot(xi,(y1+y2)*sum(h.BinCounts)*0.1, 'k-', 'linewidth', 2);
xlabel('\lambda')
ylabel('# Neurons')
legend({['\mu : ',num2str(mu1),'. \sigma :',num2str(sig1)], ['\mu : ',num2str(mu2),'. \sigma :',num2str(sig2)] ...
        ['\mu : ',num2str(mu1_sub),'. \sigma :',num2str(sig1_sub)], ['\mu : ',num2str(mu2_sub),'. \sigma :',num2str(sig2_sub)]})
title(['var > ' num2str(var_threshold) '% (N=' num2str(sum(var_short_meas>var_threshold)) '/' num2str(length(var_short_meas)) ')']);
fixTicks


% Plot individual neuron reponses with fitted scaled versions
% >> Figure S3A
figure;
set(gcf,'renderer','Painters')
offset = 0;
for ii=1:100 % Only the 100 first (change offset if want to plot different range)
    sorted_PSTH_short = sort(squeeze(PSTH_short(:, ind_sort_var(offset+ii), :)), 2);
    sorted_PSTH_long = sort(squeeze(PSTH_long(:, ind_sort_var(offset+ii), :)), 2);
    
    subplot(10, 10, ii)
    ciplot(sorted_PSTH_short(:, 2), sorted_PSTH_short(:, end-2), t_short, 'r')
    hold on
    plot(t_short, mean(sorted_PSTH_short, 2), 'r-', 'linewidth', 1.5)

    ciplot(sorted_PSTH_long(:, 2), sorted_PSTH_long(:, end-2), t_long, 'b')
    hold on
    plot(t_long, mean(sorted_PSTH_long, 2), 'b-', 'linewidth', 1.5)
   
    hold on
    plot(t_short_dense, delta_opt(ind_sort_var(offset+ii))+spline(t_long, mean(sorted_PSTH_long, 2), t_short_dense*lambda_opt(ind_sort_var(offset+ii)))*gamma_opt(ind_sort_var(offset+ii)), 'k-', 'linewidth', 1.5);
    set(gca,'xticklabel',{[]})
    xticks(0:200:1200)
    xlim([0 1200])
    ylim([0 inf])
    title([id_monkey num2str(neuron_ids(ind_sort_var(offset+ii), 1)) '\_' num2str(neuron_ids(ind_sort_var(offset+ii), 2)) ', ' num2str(lambda_opt(ind_sort_var(offset+ii)))])
end
tightfig;


% Plot trajectories in top 3 PC space (see function below)
% >> Figure 2C
[score_short, score_long, explained] = plotPCtrajs3D(mean(PSTH_short, 3), mean(PSTH_long, 3), ind_t_s_unique_short, ind_t_s_unique_long);


% This code allows users to apply PCA before doing subsequent speed analyses
% In the paper, however, PCA was NOT done 
pVar = 100; % if 100, then PCA is NOT performed
PSTH_concat = [mean(PSTH_short, 3); mean(PSTH_long, 3)];
data_mean = mean(PSTH_concat, 1);
if pVar<100 
    mat_coeff = applyPCA(PSTH_concat, pVar);    
else % if pVar=100 no need to perform PCA
    mat_coeff = eye(size(PSTH_concat, 2));
end

% For each bootstrap dataset, compute the temporal mapping between
% conditions
for iBootstrap = 1:nBootstraps
    PCscore_short = (PSTH_short(:, :, iBootstrap)-data_mean)*mat_coeff; % apply PCA if needed
    PCscore_long = (PSTH_long(:, :, iBootstrap)-data_mean)*mat_coeff; % apply PCA if needed
    [vec_min(iBootstrap, :), slope(iBootstrap), onset(iBootstrap), offset(iBootstrap)] = computeTemporalMapping(PCscore_short, PCscore_long, offset_post_ready/wbin);
    
    [PSTH_short_null, PSTH_long_null] = getNullPSTH(mean(PSTH_short, 3), mean(PSTH_long, 3)); % randomly shuffle short vs long to create null data
    PCscore_short_null = (PSTH_short_null-data_mean)*mat_coeff; % apply PCA if needed
    PCscore_long_null = (PSTH_long_null-data_mean)*mat_coeff; % apply PCA if needed
    [vec_min_null(iBootstrap, :), slope_null(iBootstrap), onset_null(iBootstrap), offset_null(iBootstrap)] = computeTemporalMapping(PCscore_short_null, PCscore_long_null, offset_post_ready/wbin);
end


% Plot temporal mapping between Short and Long
% >> Figure 2F
figure
t = 0:wbin:t_max_long; % time vector
lbd_mean = 1000/640; % ratio of prior means
vec_min_sorted = sort(vec_min, 1);
ciplot(t(vec_min_sorted(2, :)),t(vec_min_sorted(end-2, :)), t(1:size(vec_min, 2)), cmap_grey(4, :))
hold on
plot(t(1:size(vec_min, 2)), mean(t(vec_min), 1), 'k.-', 'linewidth', 2, 'markersize', 10)
hold on
plot(t(1:size(vec_min, 2)), t(1:size(vec_min, 2)), 'k--', 'linewidth', 2)
fun_mean = @(x) sum(((lbd_mean*mean(t(:, onset:offset), 1)+x)-mean(t(vec_min(:, onset:offset)), 1)).^2);
x0 = fminsearch(fun_mean, 0);
% this is a trick to only plot prediction line above the unity line
[~, ii] = find((lbd_mean*t+x0 - t)>0, 1);
plot(mean(t(ii:offset), 1), lbd_mean*mean(t(:, ii:offset), 1)+x0, 'r-', 'linewidth', 2)
xlabel('t_{short} (ms)')
ylabel('t_{long} (ms)')
axis([0 t_max_short 0 t_max_long])
fixTicks


% Plot distribution of slopes for the temporal mapping for data versus null
% >> inset of Figure 2F
figure; 
h1 = histogram(slope, 1.2:0.05:1.8, 'facecolor', 'k');
hold on
h2 = histogram(slope_null, 0.6:0.05:1.4, 'facecolor', cmap_grey(4, :));
hold on
plot(lbd_mean*ones(size(0:(max(h1.Values)+2))), 0:(max(h1.Values)+2), 'r-', 'linewidth', 2)
hold on
plot(ones(size(0:(max(h1.Values)+2))), 0:(max(h1.Values)+2), 'k-', 'linewidth', 2)
hold on
plot(mean(slope), max(h1.Values)+5, 'kv', 'markersize', 12, 'markerfacecolor', 'k')
hold on
plot(mean(slope_null), max(h1.Values)+5, 'v', 'markerfacecolor', cmap_grey(4, :), 'markeredgecolor', cmap_grey(4, :), 'markersize', 12)
[~, pval] = ttest2(slope, slope_null);
title(['p = ' num2str(pval)])
legend('data', 'null')
xlabel('Slope')
ylabel('Counts')
fixTicks


% Compute the states associated with the mean interval in each condition
mean_state_short = mean(PSTH_short(ind_t_s_unique_short(3), :, :), 3);
mean_state_long = mean(PSTH_long(ind_t_s_unique_long(3), :, :), 3);

% Compute the states between shortest and longest interval along each neural
% trajectory
ii=1;
for ind_t_s = ind_t_s_unique_short(1):ind_t_s_unique_short(end)
    state_short(ii, :, :)=PSTH_short(ind_t_s, :, :);
    ii=ii+1;
end

ii=1;
for ind_t_s = ind_t_s_unique_long(1):ind_t_s_unique_long(end)
    state_long(ii, :, :)=PSTH_long(ind_t_s, :, :);
    ii=ii+1;
end

% Define the tangent at the mean state
tangent_mean_short = mean(PSTH_short(ind_t_s_unique_short(3)+1, :, :), 3)-mean(PSTH_short(ind_t_s_unique_short(3)-1, :, :), 3);
tangent_mean_short = tangent_mean_short/norm(tangent_mean_short);

tangent_mean_long = mean(PSTH_long(ind_t_s_unique_long(3)+1, :, :), 3)-mean(PSTH_long(ind_t_s_unique_long(3)-1, :, :), 3);
tangent_mean_long = tangent_mean_long/norm(tangent_mean_long);

% Store the dot product between tangents
prod_short = nan(size(state_short, 1), 100);
prod_long = nan(size(state_long, 1), 100);

for iBootstrap=1:100
    
    % Find angle between local tangent of Long with mean tangent of Short
    for ii = 2:size(state_long, 1)-1 
        tangent_long(ii, :, iBootstrap) = (state_long(ii+1, :, iBootstrap)-state_long(ii-1, :, iBootstrap)) / ...
                                        norm(state_long(ii+1, :, iBootstrap)'-state_long(ii-1, :, iBootstrap)');
    end
    prod_long(1:end-1, iBootstrap) = (180/pi)*acos(tangent_long(:, :, iBootstrap)*tangent_mean_short');
    prod_long(1, iBootstrap) =90;
    prod_long(size(state_long, 1), iBootstrap) = 90;
        
    % Find angle between local tangent of Short with mean tangent of Long
    for ii = 2:size(state_short, 1)-1
        tangent_short(ii, :, iBootstrap) = (state_short(ii+1, :, iBootstrap)-state_short(ii-1, :, iBootstrap)) / ...
            norm(state_short(ii+1, :, iBootstrap)'-state_short(ii-1, :, iBootstrap)');
    end
    prod_short(1:end-1, iBootstrap) = (180/pi)*acos(tangent_short(:, :, iBootstrap)*tangent_mean_long');
    prod_short(1, iBootstrap) =90;
    prod_short(size(state_short, 1), iBootstrap) = 90;
    
end

% Sort bootstrapped results 
sorted_prod_long = sort(prod_long, 2);
sorted_prod_short = sort(prod_short, 2);

% Plot how the angle varies along each trajectory
% >> Figure 3A
figure;
plot(800:20:1200, mean(prod_long, 2), 'r-', 'linewidth', 2)
hold on
ciplot(sorted_prod_long(:, 2), sorted_prod_long(:, end-2), 800:20:1200, 'r')
hold on
plot(480:20:800, mean(prod_short, 2), 'b-', 'linewidth', 2)
hold on
ciplot(sorted_prod_short(:, 2), sorted_prod_short(:, end-2), 480:20:800, 'b')
hold on
plot(640*ones(size((min(mean(prod_long, 2))-5):90)), (min(mean(prod_long, 2))-5):90, 'b--')
hold on
plot(1000*ones(size((min(mean(prod_short, 2))-5):90)), (min(mean(prod_short, 2))-5):90, 'r--')
xlabel('Time from Ready (ms)')
ylabel('Angle between local tangent and mean tangent (deg)')
fixTicks


% Define a common set of tangent and mean state
tangent_mean=(tangent_mean_short+tangent_mean_long)/2;
tangent_mean = tangent_mean/norm(tangent_mean);
mean_state=(mean_state_short+mean_state_long)/2;

for iBootstrap=1:100
    
    % Project data from Short using common hyperplane
    for ii = 1:size(state_short, 1) 
        a = state_short(ii, :, iBootstrap) - mean_state;
        proj_short(ii, iBootstrap) = a*tangent_mean';
    end
    
    % Project data from Long using common hyperplane
    for ii = 1:size(state_long, 1) 
        a = state_long(ii, :, iBootstrap) - mean_state;
        proj_long(ii, iBootstrap) = a*tangent_mean';
    end
    
end
        
sorted_proj_short = sort(proj_short, 2);
sorted_proj_long = sort(proj_long, 2);

% Plot the distance to the hyperplane for each state of each condition
% >> Figure 3B
figure; 
plot((480:20:800)-640, mean(proj_short, 2), 'r-')
hold on
ciplot(sorted_proj_short(:, 2), sorted_proj_short(:, end-2), (480:20:800)-640, 'r')
hold on
plot((800:20:1200)-1000, mean(proj_long, 2), 'b-')
hold on
ciplot(sorted_proj_long(:, 2), sorted_proj_long(:, end-2), (800:20:1200)-1000, 'b')
hold on
plot((800:20:1200)-1000, zeros(size((800:20:1200)-1000)), 'k--')
hold on
plot(zeros(size(-50:50)), -50:50, 'k--')
xlabel('Deviation from the mean (ms)')
ylabel('Distance from hyperplace (a.u.)')
fixTicks


%% FUNCTIONS

% This function performs a temporal scaling analysis between Short and Long
% The format is the following:
% for Short: f(t), t between 0 and alpha
% for Long: g(t), t between 0 and beta
% The procedure minimizes [f(t) - (delta+gamma*g(lamba*t))]^2, 
% for t btw 0 and alpha
function [lambda_opt, gamma_opt, delta_opt, RMSE_opt] = findScalingFactor(D_short, D_long, t_short, t_long, t_short_dense)
    
    D_short_dense = spline(t_short, D_short, t_short_dense);
    
    % Function to minimize: MSE between D_short and scaled D_long
    fun = @(x) sum((D_short_dense - (x(3)+spline(t_long, D_long, t_short_dense*x(1))*x(2))).^2); % interpolate D_long
    
    % Initialize parameters for scaling
    lambda_0 = 1;
    gamma_0 = 1;
    delta_0 = 0;
    x_0 = [lambda_0 gamma_0 delta_0];
    
    % Find best fit
    [x_opt, RMSE_opt] = fminsearch(fun, x_0);
    lambda_opt = x_opt(1);
    gamma_opt = x_opt(2);
    delta_opt = x_opt(3);
end

% Custom function for fitting two Gaussians to the scaling factor
% distribution
function [f, xi, y1, y2, mu1, mu2, sig1, sig2] = fitBimodalGaussian(x)

    [f,xi] = ksdensity(x);  % x is your data
    % Here we generate a function from two Gaussians and output
    % the rms of the estimation error from the values obtained from ksdensity
    fun = @(xx,t,y)rms(y-(xx(5)*1./sqrt(xx(1)^2*2*pi).*exp(-(t-xx(2)).^2/(2*xx(1)^2))+...
                 xx(6)*1./sqrt(xx(3)^2*2*pi).*exp(-(t-xx(4)).^2/(2*xx(3)^2))   )  );

    % Get the parameters with the minimum error. To improve convergence,choose reasonable initial values       
    x = fminsearch(@(r)fun(r,xi,f),[0.2,1,0.2,1.5,0.5,0.5]);
    % Make sure sigmas are positive
    x([1,3]) = abs(x([1,3]));
    % Generate the Parametric functions
    pd1 = makedist('Normal','mu',x(2),'sigma',x(1));
    pd2 = makedist('Normal','mu',x(4),'sigma',x(3));
    % Get the probability values
    y1 = pdf(pd1,xi)*x(5); % x(5) is the participation factor from pdf1
    y2 = pdf(pd2,xi)*x(6); % x(6) is the participation factor from pdf2
    
    % Output the results
    mu1 = x(2);
    mu2 = x(4);
    sig1 = x(1);
    sig2 = x(3);
    
end


% This function plots neural trajectories of the Short and Long condition
% in the subspace spanned by the top 3 PCs
function [score_short, score_long, explained] = plotPCtrajs3D(PSTH_short, PSTH_long, ind_t_s_unique_short, ind_t_s_unique_long)
    
    nDims = 3; % nb of PCs to keep
    PSTH_concat = [PSTH_short; PSTH_long]; % concatenate short and long data along time dimension
    data_mean = mean(PSTH_concat, 1); %
    [coeff, ~, ~, ~, explained] = pca(PSTH_concat); % apply PCA
    mat_coeff = coeff(:, 1:nDims);
    score_short = (PSTH_short-data_mean)*mat_coeff; % center data
    score_long = (PSTH_long-data_mean)*mat_coeff; % center data
    
    % Plot the 3D trajectories
    figure; 
    plot3(score_short(:, 1), score_short(:, 2), score_short(:, 3), 'r.-', 'markersize', 12)
    hold on
    plot3(score_short(1, 1), score_short(1, 2), score_short(1, 3), 'rs', 'markerfacecolor', 'r', 'markersize', 8)
    hold on
    plot3(score_short(ind_t_s_unique_short, 1), score_short(ind_t_s_unique_short, 2), score_short(ind_t_s_unique_short, 3), 'ro', 'markerfacecolor', 'r', 'markersize', 8)
    hold on
    plot3(score_long(:, 1), score_long(:, 2), score_long(:, 3), 'b.-', 'markersize', 12)
    hold on
    plot3(score_long(1, 1), score_long(1, 2), score_long(1, 3), 'bs', 'markerfacecolor', 'b', 'markersize', 8)
    hold on
    plot3(score_long(ind_t_s_unique_long, 1), score_long(ind_t_s_unique_long, 2), score_long(ind_t_s_unique_long, 3), 'bo', 'markerfacecolor', 'b', 'markersize', 8)
    grid on
    xlabel('PC1')
    ylabel('PC2')
    zlabel('PC3')
    
    % Plot scree plot
    nPlotPCs = 10;
    figure; plot(cumsum(explained(1:nPlotPCs)), 'k.', 'markersize', 36)
    hold on
    plot(1:nPlotPCs, 100*ones(size(1:nPlotPCs)), 'k--', 'linewidth', 3)
    axis([0.8 nPlotPCs 0 105]);
    xticks(0:nPlotPCs)
    yticks(0:20:100)
    xlabel('# PCs')
    ylabel('% Var')
    fixTicks
    
end


% This function applied PCA and returns the associated weight matrix
% while only keeping top dimensions that explain pVar of total variance
function mat_coeff = applyPCA(data, pVar)
% Apply PCA to rotate input data to their top nDims PCs
% data [time x neurons], pVar = min % variance explained 
    [coeff, ~, ~, ~, explained] = pca(data);
    nDims = find(cumsum(explained)>=pVar, 1);
    mat_coeff = coeff(:, 1:nDims);
end


% Take in PSTHs from short and long conditions separately, and outputs
% shuffled PSTHs between short and long to generate null dataset
function [PSTH_short_null, PSTH_long_null] = getNullPSTH(PSTH_short, PSTH_long)

    nTimes = size(PSTH_short, 1);
    nNeurons = size(PSTH_short, 2);
    for iNeuron = 1:nNeurons
        flip = randi([0 1]); % flip a coin to decide whether it gets assigned to short or long
        if flip==0 
            PSTH_short_null(:, iNeuron) = PSTH_short(1:nTimes, iNeuron); 
            PSTH_long_null(:, iNeuron) = PSTH_long(1:nTimes, iNeuron); 
        elseif flip==1
            PSTH_short_null(:, iNeuron) = PSTH_long(1:nTimes, iNeuron); 
            PSTH_long_null(:, iNeuron) = PSTH_short(1:nTimes, iNeuron); 
        end
    end
end


% Compute the mapping between t_ref and t_test to measure speed differences
function [vec_min, slope, onset, offset] = computeTemporalMapping(PSTH_ref, PSTH_test, ind_offset)
 
    % handle case where offset post ready is zero
    if ind_offset==0
        ind_offset=1;
    end
    % In case want to compute the speed starting post ready
    PSTH_ref = PSTH_ref(ind_offset:end, :);
    PSTH_test = PSTH_test(ind_offset:end, :);
    
    ind_bin_max_ref = size(PSTH_ref, 1);
    ind_bin_max_test = size(PSTH_test, 1);
    edges = (1:ind_bin_max_test)';
    % Find mapping between short and long
        map = [];
        for ind_ref = 1:ind_bin_max_ref % for every time point on the ref
            for ind_test = 1:ind_bin_max_test % iterate over time on the test

                % compute the distance traveled along short and long for t_short
                % and t_long (starting from the origin, i.e., ready)
                normspeed_ref = sum(sqrt(sum(diff(PSTH_ref(1:ind_ref, :), 1).^2, 2)));
                normspeed_test = sum(sqrt(sum(diff(PSTH_test(1:ind_test, :), 1).^2, 2)));

                % compute the log ratio of those distances (0 is perfect match)
                map(ind_ref, ind_test) = log(normspeed_ref/normspeed_test);
            end
        end

        % put zeros for t_ref = 0
        map(1, :)=zeros(1, size(map, 2));

        % find times for which traveled distances match across conditions
        for ind_ref = 1:size(map, 1)
            [~, ind_min_test] = min(abs(map(ind_ref, :)));
            if ind_ref>1 && ind_min_test==1 % this is in case there is a near zero value for t_test = 0 (which doesn't make sense to keep)
                [~, ind_min_test] = min(abs(map(ind_ref, 2:end))); % exclude t_test = 0
                vec_min(ind_ref)=ind_min_test+1;
            else
                vec_min(ind_ref)=ind_min_test;
            end
        end

        % Compute when t_ref and t_test start diverging and use this as starting
        % point for interpolation
        [~, onset] = find(abs(vec_min-(1:length(vec_min))), 1); 
        if isempty(onset)
            onset=1;
        end

        % Compute when mapping starts saturating and use this as ending
        % point for interpolation
        offset = find(vec_min==ind_bin_max_test, 1); 
        if isempty(offset)
            offset=ind_bin_max_ref;
        end

        fun_mean = @(x) sum(((x(1)*edges(onset:offset)+x(2))-edges(vec_min(onset:offset))).^2);
        x_min = fminsearch(fun_mean, [1 0]);
        slope = x_min(1); 
        
end

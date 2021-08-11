% This code is used to generate the main results of the gain task
% reported in the following paper:
%
% Meirhaeghe N, Sohn H, Jazayeri M (2021) A precise and adaptive neural 
% mechanism for predictive temporal processing in the frontal cortex. 
% bioRxiv; https://doi.org/10.1101/2021.03.10.434831
%
% The script runs in one go and depends on functions appended at the end.
% Functions are the same as for the reproduction task. For additional 
% information contact nmrghe@gmail.com or mjaz@mit.edu


% Clear workspace
close all
clear all
clc

id_monkey = 'J'; % C for monkey 'Carmen' or 'J' for monkey 'Jiggy'
pVar = 100; % minimum percent variance to be explained (100 if no PCA)

wbin = 20; % bin size
t_s_unique_gain1 = linspace(500, 1000, 7); % ts for condition gain1
t_s_unique_gain2 = linspace(500, 1000, 7); % ts for condition gain2
t_max_gain1 = 980; % max interval from gain1
t_max_gain2 = 980; % max interval from gain2
t_gain1 = 0:wbin:t_max_gain1; % time vector for gain1
t_gain2 = 0:wbin:t_max_gain2; % time vector for gain2
t_gain1_dense = t_gain1(1):0.1:t_gain1(end);
% find indices in time vector corresponding to each ts value
[~, ind_t_s_unique_gain1] = min(abs(t_gain1-t_s_unique_gain1'), [], 2);
[~, ind_t_s_unique_gain2] = min(abs(t_gain2-t_s_unique_gain2'), [], 2);

buffer_pre_ready = 0; % value of buffer pre Ready
offset_post_ready = 0; % for speed analysis to keep early measurement

% Define color maps
cmap_grey = flip(brewermap(5, 'greys'));
cmap_red = flip(brewermap(5, 'reds'));
cmap_blue = flip(brewermap(5, 'blues'));

% Load data
load(['../../Data/' id_monkey '_gain_Ready-Set_bin20ms_bootstrap'])

% Remove neurons with low firing rate (< 1 spk/s)
neurons2keep_gain1 = find( logical( (mean(mean(PSTH_gain1, 3), 1)>1) .* (~isnan(mean(mean(PSTH_gain1, 3), 1)))) );
neurons2keep_gain2 = find( logical( (mean(mean(PSTH_gain2, 3), 1)>1) .* (~isnan(mean(mean(PSTH_gain2, 3), 1)))) );
neurons2keep = intersect(neurons2keep_gain1, neurons2keep_gain2);
PSTH_gain1 = PSTH_gain1(:, neurons2keep, :);
PSTH_gain2 = PSTH_gain2(:, neurons2keep, :);

nBootstraps = size(PSTH_gain1, 3); % nb of bootstraps

% Plot trajectories in top 3 PC space (see function below)
% >> Figure 4D
[score_gain1, score_gain2, explained] = plotPCtrajs3D(mean(PSTH_gain1, 3), mean(PSTH_gain2, 3), ind_t_s_unique_gain1, ind_t_s_unique_gain2);

% This code allows users to apply PCA before doing subsequent speed analyses
% In the paper, however, PCA was NOT done (pVar=100)
PSTH_concat = [mean(PSTH_gain1, 3); mean(PSTH_gain2, 3)];
data_mean = mean(PSTH_concat, 1);
if pVar<100 
    mat_coeff = applyPCA(PSTH_concat, pVar);    
else % if pVar=100 no need to perform PCA
    mat_coeff = eye(size(PSTH_concat, 2));
end

% For each bootstrap dataset, compute the temporal mapping between
% conditions
for iBootstrap = 1:nBootstraps
    PCscore_gain1 = (PSTH_gain1(:, :, iBootstrap)-data_mean)*mat_coeff; % apply PCA if needed
    PCscore_gain2 = (PSTH_gain2(:, :, iBootstrap)-data_mean)*mat_coeff; % apply PCA if needed
    [vec_min(iBootstrap, :), slope(iBootstrap), onset(iBootstrap), offset(iBootstrap)] = computeTemporalMapping(PCscore_gain1, PCscore_gain2, offset_post_ready/wbin);
    
    [PSTH_gain1_null, PSTH_gain2_null] = getNullPSTH(mean(PSTH_gain1, 3), mean(PSTH_gain2, 3)); % randomly shuffle gain1 vs gain2 to create null data
    PCscore_gain1_null = (PSTH_gain1_null-data_mean)*mat_coeff; % apply PCA if needed
    PCscore_gain2_null = (PSTH_gain2_null-data_mean)*mat_coeff; % apply PCA if needed
    [vec_min_null(iBootstrap, :), slope_null(iBootstrap), onset_null(iBootstrap), offset_null(iBootstrap)] = computeTemporalMapping(PCscore_gain1_null, PCscore_gain2_null, offset_post_ready/wbin);

    speed_gain1(iBootstrap, :) = computeInstantaneousSpeed(PCscore_gain1);
    speed_gain2(iBootstrap, :) = computeInstantaneousSpeed(PCscore_gain2);
end

% Plot instantaneous speed
% >> Figure S5B top
figure;
t = 0:wbin:min(t_s_unique_gain1);
speed_gain1_sorted = sort(speed_gain1, 1);
ciplot(speed_gain1_sorted(2, 1:length(t)), speed_gain1_sorted(end-2, 1:length(t)), t, cmap_red(4, :))
hold on
plot(t, mean(speed_gain1(:, 1:length(t)), 1), 'r-', 'linewidth', 3)
hold on
speed_gain2_sorted = sort(speed_gain2, 1);
ciplot(speed_gain2_sorted(2, 1:length(t)), speed_gain2_sorted(end-2, 1:length(t)), t, cmap_blue(4, :))
hold on
plot(t, mean(speed_gain2(:, 1:length(t)), 1), 'b-', 'linewidth', 3)
xlabel('Time from Ready (ms)')
ylabel('Speed (a.u.)')
fixTicks

% Compute 95% CI for speed and perform statistical test
a=sort(mean(speed_gain1(:, 1:length(t)), 2));
b=sort(mean(speed_gain2(:, 1:length(t)), 2));
CI_gain1=[a(2) a(end-2)];
CI_gain2=[b(2) b(end-2)];

[~, p, ~, stats] = ttest2(mean(speed_gain1(:, 1:length(t)), 2), mean(speed_gain2(:, 1:length(t)), 2));


% Plot temporal mapping between conditions
% >> Figure 4E and S5B bottom
figure
t = 0:wbin:t_max_gain2;
vec_min_sorted = sort(vec_min, 1);
ciplot(t(vec_min_sorted(2, :)),t(vec_min_sorted(end-2, :)), t(1:size(vec_min, 2)), cmap_grey(4, :))
hold on
plot(t(1:size(vec_min, 2)), mean(t(vec_min), 1), 'k.-', 'linewidth', 2, 'markersize', 10)
hold on
plot(t(1:size(vec_min, 2)), t(1:size(vec_min, 2)), 'k--', 'linewidth', 2)
xlabel('t_{gain1} (ms)')
ylabel('t_{gain2} (ms)')
axis([0 t_max_gain1 0 t_max_gain2])
fixTicks


% Plot distribution of slopes for the temporal mapping for data versus null
% >> inset of 4E and S5B bottom
figure; 
h1 = histogram(slope, 0.6:0.05:1.4, 'facecolor', 'k');
hold on
h2 = histogram(slope_null, 0.6:0.05:1.4, 'facecolor', cmap_grey(4, :));
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

% Compute 95% CI for mapping slope
a = sort(slope);
CI_slope = [a(2) a(end-2)];


% Plot PSTHs for all neurons
% >> Figure 4B
t = 0:wbin:t_max_gain2;
iter = 1;
figure;
for iNeuron = 1:length(neurons2keep)
    if mod(iNeuron, 49)==0
        tightfig;
        figure;
        iter = 1;
    end
    subplot(7, 7, iter)
    % Short
    sorted_PSTH_gain1 = sort(squeeze(PSTH_gain1(:, iNeuron, :)), 2);
    ciplot(sorted_PSTH_gain1(:, 2), sorted_PSTH_gain1(:, end-2), t(1:size(PSTH_gain1, 1)), 'r')
    hold on
    plot(t(1:size(PSTH_gain1, 1)), mean(sorted_PSTH_gain1, 2), 'r-', 'linewidth', 2)
    
    % Long
    sorted_PSTH_gain2 = sort(squeeze(PSTH_gain2(:, iNeuron, :)), 2);
    ciplot(sorted_PSTH_gain2(:, 2), sorted_PSTH_gain2(:, end-2), t(1:size(PSTH_gain2, 1)), 'b')
    hold on
    plot(t(1:size(PSTH_gain2, 1)), mean(sorted_PSTH_gain2, 2), 'b-', 'linewidth', 2)
    
    iter=iter+1;
    xticks(0:200:1200)
    xlim([0 1200])
    title([num2str(neuron_ids(iNeuron, 1)) '-' num2str(neuron_ids(iNeuron, 2)) ]) 
    set(gca, 'XTickLabel', [])
    fixTicks
end


% For every neuron, apply temporal scaling analysis (see in functions below)
for ind_neuron = 1:size(PSTH_gain1, 2)
    [lambda_opt(ind_neuron), gamma_opt(ind_neuron), delta_opt(ind_neuron), RMSE_opt(ind_neuron)] = findScalingFactor(nanmean(PSTH_gain1(:, ind_neuron, :), 3), nanmean(PSTH_gain2(:, ind_neuron, :), 3), t_gain1, t_gain2, t_gain1_dense);
end


% mean ratio
lbd=1;


% Plot the distributions of scaling factors for the full population
% >> Figure 4C
figure; 
h=histogram(lambda_opt, 0:0.1:1.5);
hold on
plot(lbd*ones(size(0:max(h.Values))), 0:max(h.Values), 'r-')
xlabel('\lambda')
ylabel('# Neurons')
title('r_{gain1}(t) = \gamma*r_{gain2}(\lambda*t)+\delta');
fixTicks


% Plot the distribution associated with the 3 fitted parameters of the
% scaling analysis
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
subtitle('r_{gain1}(t) = \gamma*r_{gain2}(\lambda*t)+\delta');
fixTicks
tightfig;


%% FUNCTIONS

% Compute the speed as sqrt of squared difference of firing rate in consecutive
% bins
function speed = computeInstantaneousSpeed(PSTH)
% PSTH [time x neurons]

    speed = sqrt(sum(diff(PSTH, 1).^2, 2));
    
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
    % Find mapping between gain1 and gain2
        map = [];
        for ind_ref = 1:ind_bin_max_ref % for every time point on the ref
            for ind_test = 1:ind_bin_max_test % iterate over time on the test

                % compute the distance traveled again2 gain1 and gain2 for t_gain1
                % and t_gain2 (starting from the origin, i.e., ready)
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


% This function applied PCA and returns the associated weight matrix
% while only keeping top dimensions that explain pVar of total variance
function mat_coeff = applyPCA(data, pVar)
% Apply PCA to rotate input data to their top nDims PCs
% data [time x neurons], pVar = min % variance explained 
    [coeff, ~, ~, ~, explained] = pca(data);
    nDims = find(cumsum(explained)>=pVar, 1);
    mat_coeff = coeff(:, 1:nDims);
end


% Take in PSTHs from gain1 and gain2 conditions separately, and outputs
% shuffled PSTHs between gain1 and gain2 to generate null dataset
function [PSTH_gain1_null, PSTH_gain2_null] = getNullPSTH(PSTH_gain1, PSTH_gain2)

    nTimes = size(PSTH_gain1, 1);
    nNeurons = size(PSTH_gain1, 2);
    for iNeuron = 1:nNeurons
        flip = randi([0 1]); % flip a coin to decide whether it gets assigned to gain1 or gain2
        if flip==0 
            PSTH_gain1_null(:, iNeuron) = PSTH_gain1(1:nTimes, iNeuron); 
            PSTH_gain2_null(:, iNeuron) = PSTH_gain2(1:nTimes, iNeuron); 
        elseif flip==1
            PSTH_gain1_null(:, iNeuron) = PSTH_gain2(1:nTimes, iNeuron); 
            PSTH_gain2_null(:, iNeuron) = PSTH_gain1(1:nTimes, iNeuron); 
        end
    end
end


% This function plots neural trajectories of the two conditions
% in the subspace spanned by the top 3 PCs
function [score_gain1, score_gain2, explained] = plotPCtrajs3D(PSTH_gain1, PSTH_gain2, ind_t_s_unique_gain1, ind_t_s_unique_gain2)
    
    nDims = 3; % nb of PCs to keep
    PSTH_concat = [PSTH_gain1; PSTH_gain2];
    data_mean = mean(PSTH_concat, 1);
    [coeff, ~, ~, ~, explained] = pca(PSTH_concat);
    mat_coeff = coeff(:, 1:nDims);
    score_gain1 = (PSTH_gain1-data_mean)*mat_coeff;
    score_gain2 = (PSTH_gain2-data_mean)*mat_coeff;
    
    figure; 
    plot3(score_gain1(:, 1), score_gain1(:, 2), score_gain1(:, 3), 'r.-', 'markersize', 12)
    hold on
    plot3(score_gain1(1, 1), score_gain1(1, 2), score_gain1(1, 3), 'rs', 'markerfacecolor', 'r', 'markersize', 8)
    hold on
    plot3(score_gain1(ind_t_s_unique_gain1, 1), score_gain1(ind_t_s_unique_gain1, 2), score_gain1(ind_t_s_unique_gain1, 3), 'ro', 'markerfacecolor', 'r', 'markersize', 8)
    hold on
    plot3(score_gain2(:, 1), score_gain2(:, 2), score_gain2(:, 3), 'b.-', 'markersize', 12)
    hold on
    plot3(score_gain2(1, 1), score_gain2(1, 2), score_gain2(1, 3), 'bs', 'markerfacecolor', 'b', 'markersize', 8)
    hold on
    plot3(score_gain2(ind_t_s_unique_gain2, 1), score_gain2(ind_t_s_unique_gain2, 2), score_gain2(ind_t_s_unique_gain2, 3), 'bo', 'markerfacecolor', 'b', 'markersize', 8)
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


% This function performs a temporal scaling analysis between conditions
% The format is the following:
% for Short: f(t), t between 0 and alpha
% for Long: g(t), t between 0 and beta
% The procedure minimizes [f(t) - (delta+gamma*g(lamba*t))]^2, 
% for t btw 0 and alpha
function [lambda_opt, gamma_opt, delta_opt, RMSE_opt] = findScalingFactor(D_gain1, D_gain2, t_gain1, t_gain2, t_gain1_dense)
    
    D_gain1_dense = spline(t_gain1, D_gain1, t_gain1_dense);
    
    % Function to minimize: MSE between D_gain1 and scaled D_gain2
    fun = @(x) sum((D_gain1_dense - (x(3)+spline(t_gain2, D_gain2, t_gain1_dense*x(1))*x(2))).^2); % interpolate D_gain2
    
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


% Code to predict the structure characteristcs of phase-separated 
% microstructures from the corresponding scattering
% data using GPR. The code trains and deploys four different models, making
% use of the SE, RQ, Mat5/2 and exponential kernel functions. The code
% repeats this process over many different combinations of training data

clear all

phis = [25]; % list of average blend compositions
target_names = {'normvol','normsur','normcur','normeul'}; % list of structure characteristics to predict
for z = 1:1:size(phis,2) % loop over average blend compositions
    phi = phis(z);
    for y = 1:1:size(target_names,2) % loop over structure characteristics
        
        % load relevant data 
        data = load(['../Data/Input/log10(SD)_' target_names{y} '_phi=' int2str(phi) '.txt'],'-ascii');
        inputs = data(:,1:128); % scattering data to predict from - I(q), i.e. intensity as a function of wavenumber
        targets = data(:,129); % structure characteristic to predict 
    
        % initialise run parameters and storage arrays
        n_runs = 100; % number of repeats to calculate statistics from
        size_train_set = 30; % number of scattering measurements in training set
        size_test_set = 7; % number of scattering measurements in testing set

        for a = 1:1:n_runs % loop over repeats, i.e. different combinations of training and testing data

            % Partition data into training and validation set
            train_inputs = zeros(size_train_set,128); % array to store scattering data used for training
            train_output = zeros(size_train_set,1); % array to store targets used for training
            test_inputs = zeros(size_test_set,128); % array to store scattering data used for testing
            test_output = zeros(size_test_set,1); % array to store targets used for testing
            train_inds = randperm(37,size_train_set)+2; % randomly determine data to put in training set (ignore first two measurements based on preliminary studies)
            j = 1;
            k = 1;
            for i = 3:1:39 % loop over data
                if ismember(i,train_inds)
                    train_inputs(j,:) = inputs(i,:); % make ith row of inputs the jth column of train_inputs
                    train_output(j) = targets(i);
                    j = j + 1;
                else 
                    test_inputs(k,:) = inputs(i,:);
                    test_output(k) = targets(i);
                    k = k + 1;
                end
            end

            % Define, train and deploy GPR models
            meanfunc = []; % empty: all models to have 0 mean function
            likfunc = @likGauss; % all models to use Gaussian likelihood function
            inffunc = @infGaussLik; % perform exact inference on all models

            % Kernel function a - Squared Exponential
            covfunc_a = @covSEiso; % define kernel function
            hyp_a = struct('mean', [], 'cov', [0 0], 'lik', log(0.1)); % initialise hyperparameters for given covariance function
            hyp_a2 = minimize(hyp_a, @gp, -100, inffunc, [], covfunc_a, likfunc, train_inputs, train_output); % optimise hyperparameters 
            [target_2d_array_test_a, s2_a] = gp(hyp_a2, inffunc, meanfunc, covfunc_a, likfunc, train_inputs, train_output, test_inputs); % make predictions 
            % Save predictions and true values to analyse in another script
            writematrix([target_2d_array_test_a, test_output],['../Data/Output/kern=SE_phi=' int2str(phi) '_target=' target_names{y} '_test_size=' int2str(size_train_set) '_run=' int2str(a) '.txt'],'Delimiter','tab')

            % Kernel function b - Rational Quadratic
            covfunc_b = @covRQiso; % define kernel function
            hyp_b = struct('mean', [], 'cov', [0 0 0], 'lik', log(0.1)); % initialise hyperparameters for given covariance function
            hyp_b2 = minimize(hyp_b, @gp, -100, inffunc, [], covfunc_b, likfunc, train_inputs, train_output); % optimise hyperparameters 
            [target_2d_array_test_b, s2_b] = gp(hyp_b2, inffunc, meanfunc, covfunc_b, likfunc, train_inputs, train_output, test_inputs); % make predictions
            % Save predictions and true values 
            writematrix([target_2d_array_test_b, test_output],['../Data/Output/kern=RQ_phi=' int2str(phi) '_target=' target_names{y} '_test_size=' int2str(size_train_set) '_run=' int2str(a) '.txt'],'Delimiter','tab')

            % Kernel function c - Matern_5/2
            covfunc_c = {@covMaterniso, 5}; % define kernel function
            hyp_c = struct('mean', [], 'cov', [0 0], 'lik', log(0.1)); % initialise hyperparameters for given covariance function
            hyp_c2 = minimize(hyp_c, @gp, -100, inffunc, [], covfunc_c, likfunc, train_inputs, train_output); % optimise hyperparameters 
            [target_2d_array_test_c, s2_c] = gp(hyp_c2, inffunc, meanfunc, covfunc_c, likfunc, train_inputs, train_output, test_inputs); % make predictions
            % Save predictions and true values 
            writematrix([target_2d_array_test_c, test_output],['../Data/Output/kern=Mat_phi=' int2str(phi) '_target=' target_names{y} '_test_size=' int2str(size_train_set) '_run=' int2str(a) '.txt'],'Delimiter','tab')

            % Kernel function d - Exponential (Matern_1/2)
            covfunc_d = {@covMaterniso, 1}; % define kernel function
            hyp_d = struct('mean', [], 'cov', [0 0], 'lik', log(0.1)); % initialise hyperparameters for given covariance function
            hyp_d2 = minimize(hyp_d, @gp, -100, inffunc, [], covfunc_d, likfunc, train_inputs, train_output); % optimise hyperparameters 
            [target_2d_array_test_d, s2_d] = gp(hyp_d2, inffunc, meanfunc, covfunc_d, likfunc, train_inputs, train_output, test_inputs); % make predictions
            % Save predictions and true values 
            writematrix([target_2d_array_test_d, test_output],['../Data/Output/kern=Exp_phi=' int2str(phi) '_target=' target_names{y} '_test_size=' int2str(size_train_set) '_run=' int2str(a) '.txt'],'Delimiter','tab')
        end
    end
end
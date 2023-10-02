% Code to produce figures comparing the best predictions of each structure
% characteristic made by each GPR model with the true values. For each GPR 
% model, a figure containing a subplot for each structure characteristic is
% produced

clear all

set(0,'DefaultTextInterpreter','Latex') % Latex formatting in figures

phis = [25];  % list of average blend compositions
target_names = {'normvol','normsur','normcur','normeul'}; % list of structure characteristics 
kerns = {'SE','RQ','Mat','Exp'}; % list of kernel functions
n_runs = 100; % number of repeats
size_train_set = 30; % number of scattering measurements in training set

for i = 1:1:size(kerns,2) % loop over kernel functions (models)
    kern = kerns{i};
    fig = figure('Units','Centimeters','Position',[2.00,2.35,17,25]); % create figure for each kernel function
    for y = 1:1:size(target_names,2) % loop over structure characteristics
        target_name = target_names{y};
        for z = 1:1:size(phis,2) % loop over average blend compositions
            phi = phis(z);
            RMSEs = zeros(n_runs,1); % array to store RMSE associated with each combination of training data for current structure characteristic - kernel function pair
            for a = 1:1:n_runs % loop over repeats and calculate RMSE
                data = load(['../Data/Output/kern=' kern '_phi=' int2str(phi) '_target=' target_names{y} '_test_size=' int2str(size_train_set) '_run=' int2str(a) '.txt']);
                err = ((data(:,2)-data(:,1))./data(:,2)); % calculate relative error: true-pred/true
                err_ref = err(~isinf(err)); % refine relative error values in case data contains instance of inf and/or nan
                err_ref(isnan(err_ref))=0;
                RMSEs(a) = sqrt(sum(err_ref.^2)/length(err_ref)); % calc RMSE using refined relative errors
            end
            
            % calc max and min values of RMSEs and corresponding repeat number
            [max_RMSE, i_max] = max(RMSEs); 
            [min_RMSE, i_min] = min(RMSEs);
            
            % load best predicitons (lowest RMSE)
            data_min = load(['../Data/Output/kern=' kern '_phi=' int2str(phi) '_target=' target_names{y} '_test_size=' int2str(size_train_set) '_run=' int2str(i_min) '.txt']);
            
            % plot best predicitions of each structure characteristic in different subplot
            if phi <= 25
                subplot(4,1,y);
                plot(data_min(:,2),data_min(:,1),'.','MarkerSize',14)
                hold on
            elseif phi > 25
                subplot(4,1,y);
                plot(data_min(:,2),data_min(:,1),'x','MarkerSize',8)
                hold on
            end
            
            % add titles to each subplot
            subplot(4,1,y);
            str = ["squared exponential", "rational quadratic", "Matern 5/2", "exponential"];
            if y == 1 
                title('Volume','interpreter','latex');
            elseif y == 2
                title('Surface Area','interpreter','latex');
            elseif y == 3
                title('Curvature','interpreter','latex');
            elseif y == 4 
                title('Connectivity','interpreter','latex');
            end
            refline(1,0); % add reference line to show where pred=true
            xlabel('True'); % label axes
            ylabel('Prediction');
            if y == 1 % add legend to first subplot
                legend('$\bar{\phi}$=0.25','Location','northwest','Interpreter','Latex','fontsize',8)
            end
            % format axes
            set(gca,'TickLabelInterpreter','latex','FontSize',12);
            ax = gca;
            ax.XAxis.FontSize = 10;
            ax.YAxis.FontSize = 10;
        end
    end
    sgtitle(['\bf{Best predictions made by ' kern ' model}'],'FontSize',16) % add main figure title
    %set(gcf,'visible','off');
    % specify figure size and save
    set(gcf, 'units', 'centimeters', 'Position', [0,0,17,25], 'paperunits', 'centimeters', 'papersize', [17,25]);
    saveas(fig,['../Figures/Pred_v_true_best_kern=' kern '_phi=' int2str(phi) '_target=' target_names{y} '_test_size=' int2str(size_train_set) '.pdf']);
end
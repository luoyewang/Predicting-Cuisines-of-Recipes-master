BASE_CS = readmatrix('./baseline/baseline_BASE_CS.txt');
BASE_ACS = readmatrix('./baseline/baseline_BASE_ACS.txt');
BASE_JS = readmatrix('./baseline/baseline_BASE_JS.txt');
BASE_PMI = readmatrix('./baseline/baseline_BASE_PMI.txt');

PCA_CS = readmatrix('./baseline/baseline_PCA_CS.txt');
PCA_ACS = readmatrix('./baseline/baseline_PCA_ACS.txt');
PCA_JS = readmatrix('./baseline/baseline_PCA_JS.txt');
PCA_PMI = readmatrix('./baseline/baseline_PCA_PMI.txt');

x_ax = BASE_CS(1:6,1);

figure(1);
grid on;
hold on;
plot(x_ax, BASE_CS(1:6,2), 'LineStyle', '--', 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 9, 'Color', '#0072BD');
plot(x_ax, BASE_ACS(1:6,2), 'LineStyle', '--', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 9, 'Color', '#EDB120');
plot(x_ax, BASE_JS(1:6,2), 'LineStyle', '--', 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 9, 'Color', '#D95319');
plot(x_ax, BASE_PMI(1:6,2), 'LineStyle', '--', 'LineWidth', 2, 'Marker', 'd', 'MarkerSize', 9, 'Color', '#77AC30');
plot(x_ax, PCA_CS(1:6,2), 'LineStyle', '-', 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 9, 'Color', '#0072BD');
plot(x_ax, PCA_ACS(1:6,2), 'LineStyle', '-', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 9, 'Color', '#EDB120');
plot(x_ax, PCA_JS(1:6,2), 'LineStyle', '-', 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 9, 'Color', '#D95319');
plot(x_ax, PCA_PMI(1:6,2), 'LineStyle', '-', 'LineWidth', 2, 'Marker', 'd', 'MarkerSize', 9, 'Color', '#77AC30');
hold off;

xlabel('K');
title('Recall@10');

legend('CS', 'ACS', 'JS', 'PMI', 'CS(PCA)', 'ACS(PCA)', 'JS(PCA)', 'PMI(PCA)');

set(gca,'FontSize',20);
set(gcf, 'position', [0 0 900 600]);
xlim([10, 85]);

% x_val=get(gca,'XTick');
% x_str=num2str(x_val');
% set(gca,'XTickLabel',x_str);

figure(2);
grid on;
hold on;
plot(x_ax, BASE_CS(1:6,3), 'LineStyle', '--', 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 9, 'Color', '#0072BD');
plot(x_ax, BASE_ACS(1:6,3), 'LineStyle', '--', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 9, 'Color', '#EDB120');
plot(x_ax, BASE_JS(1:6,3), 'LineStyle', '--', 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 9, 'Color', '#D95319');
plot(x_ax, BASE_PMI(1:6,3), 'LineStyle', '--', 'LineWidth', 2, 'Marker', 'd', 'MarkerSize', 9, 'Color', '#77AC30');
plot(x_ax, PCA_CS(1:6,3), 'LineStyle', '-', 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 9, 'Color', '#0072BD');
plot(x_ax, PCA_ACS(1:6,3), 'LineStyle', '-', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 9, 'Color', '#EDB120');
plot(x_ax, PCA_JS(1:6,3), 'LineStyle', '-', 'LineWidth', 2, 'Marker', 's', 'MarkerSize', 9, 'Color', '#D95319');
plot(x_ax, PCA_PMI(1:6,3), 'LineStyle', '-', 'LineWidth', 2, 'Marker', 'd', 'MarkerSize', 9, 'Color', '#77AC30');
hold off;

xlabel('K');
title('Median Rank');

legend('CS', 'ACS', 'JS', 'PMI', 'CS(PCA)', 'ACS(PCA)', 'JS(PCA)', 'PMI(PCA)');

set(gca,'FontSize',20);
set(gcf, 'position', [0 0 900 600]);
xlim([10, 85]);



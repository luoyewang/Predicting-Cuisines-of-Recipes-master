BASE_CS = readmatrix('./conca/BASE.txt');

PCA_CS = readmatrix('./conca/PCA.txt');

After_CS = readmatrix('./conca/After.txt');

x_ax = BASE_CS(:,1);
x_for_after = After_CS(:,1);

figure(1);
grid on;
hold on;
plot(x_ax, BASE_CS(:,2), 'LineStyle', '--', 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 9, 'Color', '#0072BD');
plot(x_ax, PCA_CS(:,2), 'LineStyle', '-', 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 9, 'Color', '#D95319');
plot(x_for_after, After_CS(:,2), 'LineStyle', '-', 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 9, 'Color', '#EDB120')
hold off;

xlabel('Multiply Factor');
title('Recall@10');

legend('CS', 'CS(before-PCA)', 'CS(after-PCA)');

set(gca,'FontSize',20);
set(gcf, 'position', [0 0 900 600]);
xlim([10, 200]);

% x_val=get(gca,'XTick');
% x_str=num2str(x_val');
% set(gca,'XTickLabel',x_str);

figure(2);
grid on;
hold on;
plot(x_ax, BASE_CS(:,3), 'LineStyle', '--', 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 9, 'Color', '#0072BD');
plot(x_ax, PCA_CS(:,3), 'LineStyle', '-', 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 9, 'Color', '#D95319');
plot(x_for_after, After_CS(:,3), 'LineStyle', '-', 'LineWidth', 2, 'Marker', '^', 'MarkerSize', 9, 'Color', '#EDB120')
hold off;

xlabel('Multiply Factor');
title('Median Rank');

legend('CS', 'CS(before-PCA)', 'CS(after-PCA)');

set(gca,'FontSize',20);
set(gcf, 'position', [0 0 900 600]);
xlim([10, 200]);
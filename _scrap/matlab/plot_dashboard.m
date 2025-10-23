function hFig = plot_dashboard(data)
% PLOT_DASHBOARD
    arguments
        data table
    end
    hFig = figure('Position',[100 100 1200 900]); 
    tl = tiledlayout(hFig,2,1,'TileSpacing','compact','Padding','compact');

    %% π and Eπ
    axTop = nexttile(tl,1);
    plot(axTop, data.DATE, data.pi, 'r-', 'LineWidth',2, 'DisplayName', 'Inflation \pi_t'); hold(axTop,'on');
    plot(axTop, data.DATE, data.Epi, '--', 'Color', [1 0.5 0], 'LineWidth',2, 'DisplayName', 'Expected Inflation E\pi_t');
    xlabel(axTop,'Year','FontSize',12,'FontWeight','bold');
    ylabel(axTop,'Inflation rate','FontSize',12,'FontWeight','bold');
    title(axTop,'Inflation and Expected Inflation','FontSize',14);
    grid(axTop,'on'); axTop.GridLineStyle='--'; axTop.GridAlpha=0.6;
    legend(axTop,'Location','best','FontSize',9);
    hold(axTop,'off');

    %% 
    tl2 = tiledlayout(tl,2,2,'TileSpacing','compact','Padding','compact');
    tl2.Layout.Tile = 2;

    % (1) Inverse of HHI and Markup
    ax1 = nexttile(tl2,1);
    yyaxis(ax1,'left');
    plot(ax1, data.DATE, data.N, 'b-', 'LineWidth',2, 'DisplayName','Inverse of HHI (N_t)');
    ylabel(ax1,'Inverse of HHI','FontSize',12,'FontWeight','bold','Color','b');
    yyaxis(ax1,'right');
    plot(ax1, data.DATE, data.markup, 'LineStyle','-.', 'Color',[0 0.6 0], 'LineWidth',2, 'DisplayName','Markup (\mu_t)');
    ylabel(ax1,'Markup','FontSize',12,'FontWeight','bold','Color',[0 0.6 0]);
    xlabel(ax1,'Year'); title(ax1,'Inverse of HHI and Markup','FontSize',14);
    grid(ax1,'on'); ax1.GridLineStyle='--'; ax1.GridAlpha=0.6;
    legend(ax1,'Location','northwest','FontSize',9);

    % (2) Output gap and Unemployment gap
    ax2 = nexttile(tl2,2);
    yyaxis(ax2,'left');
    plot(ax2, data.DATE, data.output_gap_BN, 'k-', 'LineWidth',2, 'DisplayName','Output gap');
    ylabel(ax2,'Output gap','FontSize',12,'FontWeight','bold','Color','k');
    yyaxis(ax2,'right');
    plot(ax2, data.DATE, data.unemp_gap, 'm-.', 'LineWidth',2, 'DisplayName','Unemployment gap');
    ylabel(ax2,'Unemployment gap','FontSize',12,'FontWeight','bold','Color','m');
    xlabel(ax2,'Year'); title(ax2,'Output gap and Unemployment gap','FontSize',14);
    grid(ax2,'on'); ax2.GridLineStyle='--'; ax2.GridAlpha=0.6;
    legend(ax2,'Location','northwest','FontSize',9);

    % (3) Output Gap and Inverse Markup
    ax3 = nexttile(tl2,3);
    yyaxis(ax3,'left');
    plot(ax3, data.DATE, data.output_gap_BN, 'k-', 'LineWidth',2, 'DisplayName','Output gap');
    ylabel(ax3,'Output gap','FontSize',12,'FontWeight','bold');
    yyaxis(ax3,'right');
    plot(ax3, data.DATE, 1./data.markup, 'LineStyle','-.', 'Color',[0 0.6 0], 'LineWidth',2, 'DisplayName','1/markup');
    ylabel(ax3,'Inverse of Markup','FontSize',12,'FontWeight','bold','Color',[0 0.6 0]);
    xlabel(ax3,'Year'); title(ax3,'Output Gap (BN) and Inverse of Markup','FontSize',14);
    grid(ax3,'on'); ax3.GridLineStyle='--'; ax3.GridAlpha=0.6;
    legend(ax3,'Location','northwest','FontSize',9);

    % (4) Output gap (BN) and Inverse of markup (BN)
    ax4 = nexttile(tl2,4);
    yyaxis(ax4,'left');
    plot(ax4, data.DATE, data.output_gap_BN, 'k-', 'LineWidth',2, 'DisplayName','Output gap (BN)');
    ylabel(ax4,'Output gap (BN)','FontSize',12,'FontWeight','bold');
    yyaxis(ax4,'right');
    plot(ax4, data.DATE, data.markup_BN_inv, 'LineStyle','-.', 'Color',[0 0.6 0], 'LineWidth',2, 'DisplayName','Inverse of markup (BN)');
    ylabel(ax4,'Inverse of markup (BN)','FontSize',12,'FontWeight','bold','Color',[0 0.6 0]);
    xlabel(ax4,'Year'); title(ax4,'Output gap (BN) and Inverse of markup (BN)','FontSize',14);
    grid(ax4,'on'); ax4.GridLineStyle='--'; ax4.GridAlpha=0.6;
    legend(ax4,'Location','northwest','FontSize',9);
end
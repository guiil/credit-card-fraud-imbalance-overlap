
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns


matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def generate_graphics_from_gridsearchcv_results(
    dataset_name,
    validation_score_summary,
    test_score_summary,
    scoring
):
    def change_width(ax, new_value):
        for patch in ax.patches:
            current_width = patch.get_width()
            diff = current_width - new_value

            # we change the bar width
            patch.set_width(new_value)

            # we recenter the bar
            patch.set_x(patch.get_x() + diff * .5)

    def change_linestyle(ax, new_value):
        for i, patch in enumerate(ax.patches):

            if i < len(ax.patches)/2:
                # change linestyle
                patch.set_linestyle(new_value)

    _, ax = plt.subplots(len(scoring), 1, sharex='all', figsize=(6.29707, 7))

    for i, score in enumerate(scoring):

        train_varname = f'mean_train_{score}_score'
        validation_varname = f'mean_test_{score}_score'

        scor = validation_score_summary[
            ['Caractere', train_varname, validation_varname]
        ]

        scor = scor.rename(
            columns={
                'Caractere': 'Estimador',
                train_varname: 'Treino',
                validation_varname: 'Validação'
            }
        )

        scor = scor.melt(
            id_vars=['Estimador'],
            value_vars=['Treino', 'Validação'],
            var_name='Passo',
            value_name=score
        )

        # if i == 0:
        #     plot_order = scor.loc[
        #         scor['Passo'] == 'Validação',
        #         ['Estimador', score]
        #     ].groupby('Estimador').mean().sort_values(
        #         score,
        #         ascending=False
        #     ).index

        sns.barplot(
            data=scor, ax=ax[i],
            # order=plot_order,
            x="Estimador",
            y=score,
            hue='Passo',
            palette='dark:white',
            edgecolor='black',
            linewidth=.8,
            capsize=.15,
            err_kws={'linewidth': .8}
        )

        test_scor = test_score_summary.loc[
            :,
            ['Caractere', score]
        ]
        test_scor.rename(
            columns={score: 'Teste'},
            inplace=True
        )

        sns.lineplot(
            data=test_scor,
            marker='o',
            sort=False,
            ax=ax[i],
            lw=1,
            label='Teste',
            markersize=4,
            palette=['r'],
            legend=False
        )

        change_width(ax[i], .36)
        change_linestyle(ax[i], '--')
        ax[i].set(xlabel=None)
        ax[i].get_legend().remove()

    ax[0].legend()
    ax[-1].set(xlabel='Estimador')
    plt.savefig(
        f'results/{dataset_name}/imgs/estimators_metrics.pgf',
        bbox_inches="tight"
    )
    plt.savefig(
        f'results/{dataset_name}/imgs/estimators_metrics.png',
        bbox_inches="tight"
    )
    plt.show()

    scor = validation_score_summary.loc[
        :,
        [
            'estimator',
            'mean_test_AUPRC_score',
            'mean_test_AUROC_score',
            'mean_test_f1_score'
        ]
    ]

    scor[['0', '1']] = scor['estimator'].apply(
        lambda x: pd.Series(str(x).split(" | "))
    )
    scor['Classificador'] = scor.apply(
        lambda x: x['0'] if x['1'] is np.nan else x['1'], axis=1
    )
    scor['Amostragem'] = scor.apply(
        lambda x: None if x['0'] == x['Classificador'] else x['0'], axis=1
    )

    scor.loc[
        scor['Amostragem'].isna(),
        'Amostragem'
    ] = 'Sem Amostragem'
    scor.drop(columns=['estimator', '0', '1'], inplace=True)
    scor = scor[
        [
            'Amostragem', 'Classificador',
            'mean_test_AUPRC_score',
            'mean_test_AUROC_score',
            'mean_test_f1_score'
        ]
    ].rename(columns={
        'mean_test_AUPRC_score': 'AUPRC validação',
        'mean_test_AUROC_score': 'AUROC validação',
        'mean_test_f1_score': 'f1 validação',
    })

    scor = scor.melt(
        id_vars=['Amostragem', 'Classificador'],
        value_vars=['AUPRC validação', 'AUROC validação', 'f1 validação'],
        var_name='Métrica',
        value_name='Valor'
    )

    test_scor = test_score_summary.loc[
        :,
        [
            'estimator',
            'AUPRC',
        ]
    ]

    test_scor[['0', '1']] = test_scor['estimator'].apply(
        lambda x: pd.Series(str(x).split(" | "))
    )
    test_scor['Classificador'] = test_scor.apply(
        lambda x: x['0'] if x['1'] is np.nan else x['1'], axis=1
    )
    test_scor['Amostragem'] = test_scor.apply(
        lambda x: None if x['0'] == x['Classificador'] else x['0'], axis=1
    )
    test_scor.loc[
        test_scor['Amostragem'].isna(),
        'Amostragem'
    ] = 'Sem Amostragem'
    test_scor.drop(columns=['estimator', '0', '1'], inplace=True)
    test_scor = test_scor[
        [
            'Amostragem', 'Classificador',
            'AUPRC'
        ]
    ].rename(columns={
        'AUPRC': 'Valor',
    })

    classificadores = pd.unique(scor['Classificador'])

    plot_order = [
        'Sem Amostragem', 'SMOTE',
        'BorderlineSMOTE', 'NearMiss',
        'ClusterCentroids'
    ]

    _, ax = plt.subplots(
        len(classificadores),
        1, sharex='all', figsize=(6.29707, 9)
    )

    for i, clf in enumerate(classificadores):

        plot_data = scor.loc[
            scor['Classificador'] == clf,
            :
        ]

        sns.barplot(
            data=plot_data, ax=ax[i], order=plot_order,
            x="Amostragem", y='Valor', hue='Métrica',
            palette='dark:white', edgecolor='black', linewidth=.8,
            capsize=0.10, err_kws={'linewidth': .8}
        )

        plot_data = test_scor.loc[
            test_scor['Classificador'] == clf,
            ['Amostragem', 'Valor']
        ]

        sns.lineplot(
            data=plot_data, ax=ax[i],
            x='Amostragem', y='Valor',
            marker='o',  # sort=False,
            lw=1, label='AUPRC teste',
            markersize=4, color='r', legend=False
        )

        if i == 0:
            ax[i].xaxis.set_label_position('top')

        ax[i].title.set_text(clf)
        ax[i].set(ylabel=None)
        ax[i].get_legend().remove()
        ax[i].yaxis.set_ticks(np.arange(0, 1.01, 0.25))

    # Shrink current axis by 20%
    box = ax[2].get_position()
    ax[2].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(
        f'results/{dataset_name}/imgs/sampling_compared.pgf',
        bbox_inches="tight"
    )
    plt.savefig(
        f'results/{dataset_name}/imgs/sampling_compared.png',
        bbox_inches="tight"
    )
    plt.show()

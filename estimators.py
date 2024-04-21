import os
import logging

import builtins
import json
import itertools
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import \
    roc_curve, RocCurveDisplay, \
    auc, f1_score, \
    precision_recall_curve, \
    PrecisionRecallDisplay, \
    average_precision_score, \
    confusion_matrix

from sklearn.model_selection import GridSearchCV, StratifiedKFold

import pickle

from metrics import RValue

logging.config.fileConfig("logging.conf")
global L
L = logging.getLogger('root')


class EstimatorSelectionHelper:
    # https://www.davidsbatista.net/blog/2018/02/23/model_optimization/

    def generate_params_for_gridsearchcv(self, key, d: dict):
        if d is None:
            return None

        if isinstance(d, list):
            aux_d = []
            for i in d:
                aux_d.append(self.generate_params_for_gridsearchcv(key, i))

        if isinstance(d, dict):
            aux_d = {}
            for k, v in d.items():
                aux_d[f'{key}__{k}'] = v

        return aux_d

    def now(self): return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def __init__(
            self,
            transformers,
            models,
            dataset_name
    ):
        self.transformers = transformers
        self.models = models
        self.dataset_name = dataset_name

        self.estimator_product = itertools.product(
            self.transformers,
            self.models
        )

        # initiate variables
        self.grid_searches = {}
        self.test_results = {}

    def fit_predict(
        self,
        x_train, x_test, y_train, y_test,
        cv=5,
        n_jobs=3, verbose=1,
        scoring=None, refit=False,
        pre_dispatch='2*n_jobs',
    ):

        r_values_path = f'results/{self.dataset_name}/r_values_all_data.json'

        r_measuring = RValue()

        x = np.concatenate((x_train, x_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        if not os.path.exists(r_values_path):
            self.r_values_all_data = r_measuring.generate_all_metrics(x, y)
            with open(r_values_path, 'w') as f:
                f.write(json.dumps(self.r_values_all_data))
        else:
            with open(r_values_path, 'r') as f:
                self.r_values_all_data = json.loads(f.read())

        metrics_json = 'Cálculo de R-Valor:' + \
            json.dumps(self.r_values_all_data, indent=2)

        print(metrics_json)

        self.scoring = scoring

        for transformer, model in self.estimator_product:
            # string representations
            transformer__name__ = transformer.__class__.__name__
            model__name__ = model.__class__.__name__

            transformer_params = self.transformers.get(transformer, {})
            model_params = self.models.get(model, {})

            # create pipeline
            pipe = imbpipeline(
                steps=[
                    ['StandardScaler', StandardScaler()],
                    [model__name__, model]
                ]
            )

            estimator = estimator__name__ = model__name__
            if transformer:
                estimator = (
                    transformer__name__ + ' | ' + estimator
                    if transformer__name__ != ''
                    else estimator
                )
                estimator__name__ = transformer__name__ + '__' + model__name__
                pipe.steps.insert(1, [transformer__name__, transformer])

            print(
                f'[{self.now()}] Running GridSearchCV({cv}) for {estimator}\n',
                ' '*21, f'Pre-processing: {transformer__name__}\n',
                ' '*21, f'Pre-processing params: {transformer_params}\n',
                ' '*21, f'Model: {model__name__}\n',
                ' '*21, f'Model params: {model_params}\n',
            )

            # set params for GridSearchCV
            params = self.generate_params_for_gridsearchcv(
                model__name__,
                model_params
            )
            transformer_params = self.generate_params_for_gridsearchcv(
                transformer__name__,
                transformer_params
            )

            # add transformer params to model params for GridSearchCV
            match type(transformer_params):
                case builtins.dict:
                    params.update(transformer_params)
                case builtins.list:
                    for item in params:
                        item.update(transformer_params)
                case None:
                    params.update()

            pickle_path = f'results/{self.dataset_name}/pkls/' \
                          f'GridSearchCV_{estimator__name__}.pkl'

            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as p:
                    gs = pickle.load(p)
                with open(
                    (
                        f'results/{self.dataset_name}/'
                        f'test_results/{estimator__name__}.json'
                    ),
                    'r'
                ) as f:
                    self.test_results[estimator__name__] = json.load(f)

                gridsearchcv_fit_time = timedelta(
                    seconds=self.test_results[
                        estimator__name__
                    ]['gridsearchcv_fit_time']
                )

            else:

                gs = GridSearchCV(
                    estimator=pipe, param_grid=params,
                    cv=cv, n_jobs=n_jobs,
                    verbose=verbose, scoring=scoring, refit=refit,
                    pre_dispatch=pre_dispatch, return_train_score=True
                )

                t1 = datetime.now()
                gs.fit(x_train, y_train)
                t2 = datetime.now()

                # fit time
                gridsearchcv_fit_time = t2 - t1

                # save to file
                with open(pickle_path, 'wb') as p:
                    pickle.dump(gs, p)

            print(
                f'[{self.now()}] Fit execution time is '
                f'{gridsearchcv_fit_time} '
                f'({gridsearchcv_fit_time.total_seconds()} seconds).'
            )

            print('Melhores parâmetros: ', gs.best_params_)
            print('Melhor AUPRC: ', gs.best_score_)

            t1 = datetime.now()
            y_pred = gs.predict(x_test)
            y_probas = gs.predict_proba(x_test)
            y_score = y_probas[:, 1]
            t2 = datetime.now()

            # predict_time
            gridsearchcv_predict_time = t2 - t1

            print(
                f'[{self.now()}] Predict execution time is '
                f'{gridsearchcv_predict_time}'
                f'({gridsearchcv_predict_time.total_seconds()} '
                'seconds).'
            )

            confusion_mtx = confusion_matrix(y_test, y_pred)
            cmn = (confusion_mtx.astype('float') /
                   confusion_mtx.sum(axis=1)[:, np.newaxis])

            group_counts = [
                f'{v:0.0f}' for v in confusion_mtx.flatten()
            ]

            group_percentages = [
                f'{v:.2%}'
                for v in
                cmn.flatten()
            ]
            labels = [
                f'{v1}\n{v2}'
                for v1, v2 in
                zip(group_counts, group_percentages)
            ]
            labels = [labels[:2], labels[2:]]
            ax = sns.heatmap(cmn, annot=labels, fmt='s')
            for t in ax.texts:
                t.set_text(t.get_text() + " %")
            plt.ylabel('Real')
            plt.xlabel('Previsto')
            plt.show(block=False)

            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            print(f'AUROC: {roc_auc}')

            roc_display = RocCurveDisplay(
                fpr=fpr,
                tpr=tpr,
                roc_auc=roc_auc,
                estimator_name=estimator,
            ).plot()

            roc_display.ax_.set_title(f"ROC {estimator}")
            plt.savefig(
                f'results/{self.dataset_name}/imgs/pgf/'
                f'roc_curve_{estimator__name__}.pgf', bbox_inches="tight"
            )
            plt.savefig(
                f'results/{self.dataset_name}/imgs/png/'
                f'roc_curve_{estimator__name__}.png', bbox_inches="tight"
            )
            plt.show()

            precision, recall, _ = precision_recall_curve(y_test, y_score)
            average_precision = average_precision_score(y_test, y_score)
            print(f'AUPRC: {average_precision}')

            prc_display = PrecisionRecallDisplay(
                precision=precision,
                recall=recall,
                average_precision=average_precision,
                estimator_name=estimator
            ).plot()

            prc_display.ax_.set_title(
                f"Curva Precisão-Recall {estimator}"
            )

            plt.savefig(
                f'results/{self.dataset_name}/imgs/pgf/'
                f'pr_curve_{estimator__name__}.pgf', bbox_inches="tight"
            )
            plt.savefig(
                f'results/{self.dataset_name}/imgs/png/'
                f'pr_curve_{estimator__name__}.png', bbox_inches="tight"
            )
            plt.show()

            f1 = f1_score(y_test, y_pred)
            print(f'f1-score: {f1}')

            self.test_results[estimator__name__] = {
                'dataset_name': self.dataset_name,
                'best_params_': gs.best_params_,
                'confusion_matrix': confusion_mtx.tolist(),
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'roc_auc': roc_auc,
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'auprc': average_precision,
                'f1-score': f1,
                'gridsearchcv_fit_time':
                    gridsearchcv_fit_time.total_seconds(),
                'gridsearchcv_predict_time':
                    gridsearchcv_predict_time.total_seconds(),
            }

            with open(
                (
                    f'results/{self.dataset_name}/'
                    f'test_results/{estimator__name__}.json'
                ),
                'w'
            ) as f:
                json.dump(self.test_results[estimator__name__], f, indent=4)

    def generate_metadata_summary(self):

        # initiate empty DataFrame
        df_metadata = pd.DataFrame()

        # iterate through grid searches
        for estimator in self.test_results:

            pickle_path = f'results/{self.dataset_name}/pkls/' \
                          f'GridSearchCV_{estimator}.pkl'

            with open(pickle_path, 'rb') as p:
                gs = pickle.load(p)

            # fetch estimator cv results
            cv_results = gs.cv_results_

            # create dataframe with cv metadata
            metadata_cols = [
                'mean_fit_time',
                'std_fit_time',
                'mean_score_time',
                'std_score_time'
            ]
            metadata = [cv_results.get(m) for m in metadata_cols]
            df_metadata_estimator = pd.DataFrame(
                data=list(zip(*metadata)),
                columns=metadata_cols
            )

            # add estimator index
            df_metadata_estimator['estimator'] = estimator
            df_metadata_estimator.set_index(
                'estimator',
                append=True,
                inplace=True
            )
            df_metadata_estimator = df_metadata_estimator.swaplevel(0, 1)

            # concat with all estimators
            df_metadata = pd.concat([df_metadata, df_metadata_estimator])

        return df_metadata

    def generate_validation_score_summary(self):

        # initiate empty DataFrame
        df_scores = pd.DataFrame()

        # initiate empty dict
        self.all_cv_results = {}

        for estimator in self.test_results:

            pickle_path = f'results/{self.dataset_name}/pkls/' \
                          f'GridSearchCV_{estimator}.pkl'

            with open(pickle_path, 'rb') as p:
                gs = pickle.load(p)

            # fetch estimator cv results
            cv_results = gs.cv_results_

            # get params from cv_results
            params = cv_results['params']

            # validate and create variable for split ranges
            if isinstance(gs.cv, StratifiedKFold):
                splits_range = range(gs.cv.n_splits)
            else:
                splits_range = gs.cv

            # instantiate new dataframes
            df_estimator_scores = pd.DataFrame.from_records(params)
            df_estimator_scores['estimator'] = estimator

            # iterate through scoring options
            for score_name in self.scoring.keys():

                df_split_scores = pd.DataFrame()

                # iterate through split ranges
                for s in splits_range:

                    for score_type in ['train', 'test']:
                        # set key as split i for test set
                        # with the scoring option
                        split_key = f"split{s}_{score_type}_{score_name}"

                        # get results for it
                        r = gs.cv_results_[split_key]

                        # append to dataframe
                        df_split_scores[split_key] = r

                        df_estimator_scores[
                            f'min_{score_type}_{score_name}_score'
                        ] = df_split_scores.apply(np.min, axis=1)

                        df_estimator_scores[
                            f'max_{score_type}_{score_name}_score'
                        ] = df_split_scores.apply(np.max, axis=1)

                        df_estimator_scores[
                            f'mean_{score_type}_{score_name}_score'
                        ] = df_split_scores.apply(np.mean, axis=1)

                        df_estimator_scores[
                            f'std_{score_type}_{score_name}_score'
                        ] = df_split_scores.apply(np.std, axis=1)

            df_scores = pd.concat([df_scores, df_estimator_scores])

            self.all_cv_results[estimator] = gs.cv_results_

        score_cols = [
            [
                f'min_train_{s}_score', f'mean_train_{s}_score',
                f'max_train_{s}_score', f'std_train_{s}_score',
                f'min_test_{s}_score', f'mean_test_{s}_score',
                f'max_test_{s}_score', f'std_test_{s}_score',
            ]
            for s in self.scoring.keys()
        ]

        columns = ['estimator'] + [
            s for nestedlist in score_cols for s in nestedlist
        ]
        columns = columns + [c for c in df_scores.columns if c not in columns]

        score_summary = df_scores[columns].reset_index(drop=True)

        return score_summary

    def generate_test_score_summary(self):

        df_test_score = pd.DataFrame()
        for estimator, result in self.test_results.items():
            df_test_score = pd.concat([
                df_test_score,
                pd.DataFrame(
                    {
                        'estimator': [estimator],
                        'AUPRC': [result['auprc']],
                        'AUROC': [result['roc_auc']],
                        'f1': [result['f1-score']]
                    }
                )
            ])

        score_summary = df_test_score.reset_index(drop=True)
        return score_summary

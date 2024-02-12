import CompareMetrics from '../components/compare-metrics.js';
import Placeholder from '../components/placeholder.js';
import RunMetrics from '../components/run-metrics.js';
import RunParameters from '../components/run-parameters.js';
import Overview from './overview.js';
import SingleRun from './single-run.js';
import MultipleRuns from './multiple-runs.js';

export const routes = [
    { path: '/', name: 'index', component: Overview },
    {
        path: '/run/:id',
        name: 'single-run',
        component: SingleRun,
        props: true,
        children: [
            { path: '', redirect: { name: 'single-run-metrics' } },
            { path: 'metrics', name: 'single-run-metrics', component: RunMetrics },
            { path: 'parameters', name: 'single-run-params', component: RunParameters },
            { path: 'pipeline', name: 'single-run-pipeline', component: Placeholder },
        ],
    },
    {
        path: '/compare',
        name: 'compare-runs',
        component: MultipleRuns,
        children: [
            { path: '', redirect: { name: 'compare-runs-metrics' } },
            { path: 'metrics', name: 'compare-runs-metrics', component: CompareMetrics },
            { path: 'diff', name: 'compare-runs-diff', component: Placeholder },
        ],
    },
];

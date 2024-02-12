import Placeholder from '../components/placeholder.js';
import RunMetrics from '../components/run-metrics.js';
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
            { path: 'parameters', name: 'single-run-params', component: Placeholder },
            { path: 'pipeline', name: 'single-run-pipeline', component: Placeholder },
        ],
    },
    { path: '/compare', name: 'compare-runs', component: MultipleRuns },
];
